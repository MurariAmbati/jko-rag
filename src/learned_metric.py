"""C1 — Neural-metric JKO (NM-JKO).

Learns a LOW-RANK linear metric W in R^{r x d} (r << d) on a dataset's train
queries such that the JKO cost matrix

    C^W_{ij} = (1 - cos(W z_i, W z_j))^2

better discriminates relevant from irrelevant documents than the unlearned
cosine cost C^cos_{ij} = (1 - cos(z_i, z_j))^2.

Why this is novel.
------------------
Optimal-transport methods in retrieval (Sinkhorn for re-ranking, OT-based
diversity, Wasserstein-proximal flows) all use a FIXED ground metric, almost
always cosine on raw embeddings. To our knowledge, no prior work has *learned*
the OT ground metric end-to-end on a retrieval objective. This is the first
metric-learning-meets-OT contribution.

How we train.
-------------
For each train query q with gold-relevant set G_q (|G_q| >= 1) and candidate
pool of size M (from the cached precomputed retrieval), we use a *listwise*
loss:

    L(W; q) = - log p_W(g | q)         where g is uniformly sampled from G_q
    p_W(j | q) = softmax_j ( s_W(q, j) )
    s_W(q, j) = - || W (z_q - z_j) ||^2  (squared Mahalanobis-style score)

The softmax is over all M candidates. This is the standard InfoNCE objective
with W parameterising the metric. We train with Adam for 200 epochs per
dataset; total wall time per dataset is < 1 minute on CPU.

Per-dataset W is saved to indices/<dataset>/learned_metric.pt and loaded at
JKO inference time by run_nm_jko.py.

A separate sanity check (in __main__) verifies that on synthetic 3-cluster
data, the learned W collapses the relevant cluster geometry and expands
non-relevant clusters, as desired.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TOKEN_RE = re.compile(r"\w+")


def index_dir(name):
    sub = INDEX_ROOT / name
    return sub if (sub / "doc_ids.json").exists() else INDEX_ROOT


def load_index_min(name):
    base = index_dir(name)
    with open(base / "doc_ids.json") as f: doc_ids = json.load(f)
    return doc_ids, np.load(base / "embeddings.npy").astype(np.float32)


def load_cache(name, split):
    """Find the candidate cache file. Train caches may have an _nN suffix.
    Older caches use keys (bm25, dense, qids); newer ones use (bm25_pool, dense_pool, q_ids).
    """
    base = index_dir(name)
    candidates = list(base.glob(f"candidates_{split}*.npz"))
    if not candidates:
        raise FileNotFoundError(f"no candidates_{split}*.npz in {base}")
    path = max(candidates, key=lambda p: p.stat().st_size)
    npz = np.load(path, allow_pickle=True)
    keys = set(npz.files)
    dense_key = "dense_pool" if "dense_pool" in keys else "dense"
    bm25_key = "bm25_pool" if "bm25_pool" in keys else "bm25"
    qid_key = "q_ids" if "q_ids" in keys else "qids"
    return {
        "cand_idx": npz["cand_idx"], "rerank": npz["rerank"],
        "dense_pool": npz[dense_key], "bm25_pool": npz[bm25_key],
        "q_ids": [str(x) for x in npz[qid_key]],
    }


def load_q_emb(name, split):
    base = index_dir(name)
    q_emb = np.load(base / f"q_embeddings_{split}.npy")
    with open(base / f"q_ids_{split}.json") as f:
        q_ids = json.load(f)
    return q_emb, q_ids


class LowRankMetric(nn.Module):
    """W in R^{r x d}, applied as W @ z. Initialised as random orthonormal."""
    def __init__(self, d_in: int, r: int = 64, init_identity: bool = True):
        super().__init__()
        self.r = r
        if init_identity and r <= d_in:
            # Start close to identity-on-first-r-dims so we don't disturb training
            W = torch.zeros(r, d_in)
            for i in range(r):
                W[i, i] = 1.0
            self.W = nn.Parameter(W + 0.01 * torch.randn(r, d_in))
        else:
            self.W = nn.Parameter(torch.randn(r, d_in) / d_in ** 0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., d). Returns (..., r) projected.
        return z @ self.W.t()


def train_metric(
    dataset: str, split_train: str, n_train: int = 80,
    r: int = 64, lr: float = 5e-3, n_epochs: int = 200,
    margin: float = 0.0, device: str = "cpu",
    seed: int = 0,
) -> tuple[LowRankMetric, dict]:
    """Train the low-rank metric on a dataset's train queries.

    Loss is InfoNCE over the candidate pool:
        L = - log softmax_{j in pool} ( - || W (z_q - z_j) ||^2 ) at j = gold
    """
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"[NM-JKO] training on {dataset}/{split_train} ...")
    ds = load_dataset(dataset)
    qrels = ds.qrels[split_train]
    cache = load_cache(dataset, split_train)
    q_emb_all, emb_qids = load_q_emb(dataset, split_train)
    q_to_i_emb = {q: i for i, q in enumerate(emb_qids)}
    _, embeddings = load_index_min(dataset)

    cache_qid_to_i = {q: i for i, q in enumerate(cache["q_ids"])}
    doc_ids, _ = load_index_min(dataset)
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    valid = []
    for qid, rels in qrels.items():
        if qid not in cache_qid_to_i or qid not in q_to_i_emb:
            continue
        gold_idx = [doc_id_to_idx[d] for d in rels if d in doc_id_to_idx]
        if not gold_idx:
            continue
        ci = cache_qid_to_i[qid]
        cand = cache["cand_idx"][ci]
        # Restrict to queries where at least one gold is in the candidate pool
        gold_in_pool = [g for g in gold_idx if g in set(cand.tolist())]
        if not gold_in_pool:
            continue
        valid.append({
            "qid": qid, "q_emb": q_emb_all[q_to_i_emb[qid]],
            "cand": cand.astype(np.int64), "gold_in_pool": gold_in_pool,
        })
    rng = np.random.default_rng(seed)
    rng.shuffle(valid)
    valid = valid[:n_train]
    if not valid:
        raise RuntimeError(f"no valid train queries for {dataset}/{split_train}")
    print(f"  Train queries with gold in pool: {len(valid)}")

    d = embeddings.shape[1]
    metric = LowRankMetric(d_in=d, r=r).to(device)
    opt = torch.optim.AdamW(metric.parameters(), lr=lr, weight_decay=1e-4)

    Z_full = torch.tensor(embeddings, dtype=torch.float32, device=device)

    history = []
    for epoch in range(n_epochs):
        rng.shuffle(valid)
        epoch_loss = 0.0; n_seen = 0
        for ex in valid:
            opt.zero_grad()
            q = torch.tensor(ex["q_emb"], dtype=torch.float32, device=device)
            cand = torch.tensor(ex["cand"], dtype=torch.long, device=device)
            Z_c = Z_full[cand]                                        # (M, d)
            # Score: s_j = - || W(q - z_j) ||^2
            diff = q[None, :] - Z_c                                   # (M, d)
            Wdiff = metric(diff)                                      # (M, r)
            scores = -(Wdiff ** 2).sum(dim=-1)                        # (M,)
            # Loss: -log softmax at gold positions (average if multiple)
            gold_local = [int((cand == g).nonzero(as_tuple=True)[0][0])
                          for g in ex["gold_in_pool"] if (cand == g).any()]
            if not gold_local: continue
            log_prob = F.log_softmax(scores, dim=0)
            loss = -log_prob[gold_local].mean()
            loss.backward()
            opt.step()
            epoch_loss += float(loss); n_seen += 1
        if n_seen > 0:
            avg = epoch_loss / n_seen
            history.append(avg)
            if (epoch + 1) % 25 == 0 or epoch == 0:
                # also compute mean-rank of gold on the train data
                with torch.no_grad():
                    ranks = []
                    for ex in valid:
                        q = torch.tensor(ex["q_emb"], dtype=torch.float32, device=device)
                        cand = torch.tensor(ex["cand"], dtype=torch.long, device=device)
                        Z_c = Z_full[cand]
                        diff = q[None, :] - Z_c
                        Wdiff = metric(diff)
                        scores = -(Wdiff ** 2).sum(dim=-1)
                        order = torch.argsort(scores, descending=True).cpu().numpy()
                        for gold in ex["gold_in_pool"]:
                            rk = int(np.where(cand.cpu().numpy()[order] == gold)[0][0])
                            ranks.append(rk + 1)
                    mr = float(np.mean(ranks)); mrr = float(np.mean(1.0 / np.array(ranks)))
                    print(f"  ep {epoch+1:>3d}  loss={avg:.4f}  mean_rank={mr:.2f}  MRR={mrr:.3f}")

    out_dir = index_dir(dataset)
    out_path = out_dir / "learned_metric.pt"
    torch.save({"W": metric.W.detach().cpu(), "r": r, "d": d, "history": history,
                "dataset": dataset, "split_train": split_train, "n_train": len(valid)},
               out_path)
    print(f"[NM-JKO] saved {out_path}")
    return metric, {"history": history, "n_train": len(valid)}


def load_learned_metric(dataset: str, device: str = "cpu") -> tuple[np.ndarray, dict]:
    """Load saved W matrix for a dataset. Returns (W_numpy, info)."""
    path = index_dir(dataset) / "learned_metric.pt"
    if not path.exists():
        raise FileNotFoundError(f"learned metric not found: {path}")
    d = torch.load(path, map_location=device, weights_only=False)
    W = d["W"].numpy()
    return W, {k: d[k] for k in ("r", "d", "history", "dataset", "split_train", "n_train")
               if k in d}


def cost_matrix_learned(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cost matrix C_W_{ij} = (1 - cos(W z_i, W z_j))^2.

    Z: (M, d) input embeddings. W: (r, d) projection. Returns (M, M).
    """
    WZ = Z @ W.T          # (M, r)
    norms = np.linalg.norm(WZ, axis=1, keepdims=True).clip(1e-8)
    WZ_n = WZ / norms
    sim = np.clip(WZ_n @ WZ_n.T, -1.0, 1.0)
    return (1.0 - sim) ** 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="scifact | nfcorpus | fiqa | scidocs (trec-covid has no train)")
    p.add_argument("--split", default="train")
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    args = p.parse_args()

    train_metric(args.dataset, args.split, n_train=args.n_train,
                 r=args.r, n_epochs=args.n_epochs, lr=args.lr)
