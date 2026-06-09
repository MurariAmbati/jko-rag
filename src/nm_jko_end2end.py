"""D3 -- End-to-end NM-JKO training.

The InfoNCE training of learned_metric.py optimises a query-doc similarity
proxy, not the JKO output directly. Here we train W by UNROLLING JKO and
backpropagating a differentiable ranking loss (LambdaLoss-style pairwise
logistic) on p_T.

Setup:
  - Initialise W from the InfoNCE-pretrained metric (warm start).
  - For each train query: build candidate pool of M from precomputed cache;
    compute cost C_W = (1 - cos(WZ, WZ))^2 (differentiable in W);
    unroll T_unroll JKO steps (using inner_steps_unroll Adam steps with
    create_graph=True for the LAST iteration only -- envelope-theorem style);
    obtain p_T;
  - Differentiable ranking loss:
        L = sum_{gold g, neg n}  log(1 + exp(-(p_T[g] - p_T[n]) / tau))
    over pairs in pool.
  - Adam on W. lr=1e-3, weight_decay=1e-4. n_epochs=20.

Notes:
  - Unrolling Sinkhorn through autograd is expensive. We use T_unroll=1 and
    inner_steps_unroll=8 with n_iter_grad_sinkhorn=4 for tractability.
  - Each training step is ~1-2 seconds for M=200, total ~20 epochs * 80 queries
    * 1.5s = 40 minutes per dataset.
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
from jko import log_sinkhorn_loss
from learned_metric import LowRankMetric, load_learned_metric, index_dir, load_index_min, load_q_emb, load_cache

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def cost_matrix_differentiable(Z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """C_{ij} = (1 - cos(W z_i, W z_j))^2.  Fully differentiable in W."""
    WZ = Z @ W.t()                                          # (M, r)
    WZ_norm = WZ / WZ.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = WZ_norm @ WZ_norm.t()
    sim = sim.clamp(-1.0, 1.0)
    return (1.0 - sim) ** 2


def unrolled_jko_step(
    p_prev: torch.Tensor, energy: torch.Tensor,
    C: torch.Tensor, K: torch.Tensor,
    h: float, lam: float, rho: float,
    eps_sinkhorn: float, n_inner: int = 8,
    inner_lr: float = 0.1, n_iter_grad_sinkhorn: int = 4,
) -> torch.Tensor:
    """ONE differentiable JKO step.  Inner optimisation is unrolled Adam-lite
    (we use gradient steps with manual gradient computation so the whole
    thing is in the autograd graph).
    """
    log_p_prev = torch.log(p_prev.clamp_min(1e-30))
    theta = log_p_prev.clone().detach().requires_grad_(True)

    # We need to retain graph between inner steps; use create_graph
    # for the LAST step only (envelope theorem at the optimum).
    for k in range(n_inner):
        log_p = F.log_softmax(theta, dim=0)
        p = log_p.exp()
        loss = (p * energy).sum()
        loss = loss + lam * (p * log_p).sum()
        loss = loss + 0.5 * rho * (p @ (K @ p))
        w2 = log_sinkhorn_loss(
            log_p, log_p_prev, C, eps=eps_sinkhorn,
            n_iter=30, n_iter_grad=n_iter_grad_sinkhorn,
        )
        loss = loss + (1.0 / (2.0 * h)) * w2

        create_graph = (k == n_inner - 1)
        grad = torch.autograd.grad(loss, theta, create_graph=create_graph)[0]
        # plain SGD step (so we can keep graph)
        theta = theta - inner_lr * grad
    p_next = F.softmax(theta, dim=0)
    return p_next


def pairwise_logistic_loss(p_T: torch.Tensor, gold_mask: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """L = mean over (g in gold, n in neg) of log(1 + exp(-(p_T[g] - p_T[n]) / tau))."""
    M = p_T.shape[0]
    gold_idx = torch.where(gold_mask)[0]
    neg_idx = torch.where(~gold_mask)[0]
    if len(gold_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, requires_grad=True)
    g_vals = p_T[gold_idx]                  # (G,)
    n_vals = p_T[neg_idx]                   # (N,)
    diff = g_vals[:, None] - n_vals[None, :]     # (G, N)
    return torch.log1p(torch.exp(-diff / tau)).mean()


def train_e2e(dataset: str, split_train: str = "train", n_train: int = 80,
              r: int = 64, n_epochs: int = 20, lr: float = 1e-3, seed: int = 0,
              h_jko: float = 2.0, lam_jko: float = 0.1, rho_jko: float = 0.05,
              eps_sinkhorn: float = 0.2, tau_loss: float = 0.5, tau0: float = 1.0,
              n_inner: int = 6, T_unroll: int = 1, device: str = "cpu",
              warmstart_from_infonce: bool = True):
    """Train W end-to-end through JKO unroll."""
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"[NM-JKO E2E] training {dataset}/{split_train}, n_train={n_train}, T_unroll={T_unroll}, n_inner={n_inner}")

    ds = load_dataset(dataset)
    qrels = ds.qrels[split_train]
    cache = load_cache(dataset, split_train)
    q_emb_all, emb_qids = load_q_emb(dataset, split_train)
    q_to_i_emb = {q: i for i, q in enumerate(emb_qids)}
    doc_ids, embeddings = load_index_min(dataset)
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    cache_qid_to_i = {q: i for i, q in enumerate(cache["q_ids"])}
    valid = []
    for qid, rels in qrels.items():
        if qid not in cache_qid_to_i or qid not in q_to_i_emb:
            continue
        gold_idx = [doc_id_to_idx[d] for d in rels if d in doc_id_to_idx]
        if not gold_idx: continue
        ci = cache_qid_to_i[qid]
        cand = cache["cand_idx"][ci]
        cand_set = set(cand.tolist())
        gold_in_pool = [g for g in gold_idx if g in cand_set]
        if not gold_in_pool: continue
        valid.append({
            "qid": qid, "q_emb": q_emb_all[q_to_i_emb[qid]],
            "cand": cand.astype(np.int64), "gold_in_pool": set(gold_in_pool),
            "dense": cache["dense_pool"][ci].astype(np.float32),
            "rerank": cache["rerank"][ci].astype(np.float32),
        })
    rng = np.random.default_rng(seed); rng.shuffle(valid); valid = valid[:n_train]
    print(f"  {len(valid)} train queries with gold in pool")

    d = embeddings.shape[1]
    metric = LowRankMetric(d_in=d, r=r).to(device)
    if warmstart_from_infonce:
        try:
            W_init, _ = load_learned_metric(dataset)
            with torch.no_grad():
                metric.W.copy_(torch.tensor(W_init, dtype=torch.float32))
            print(f"  Warm-started from InfoNCE-trained W")
        except FileNotFoundError:
            print(f"  No InfoNCE W found; using random init")

    opt = torch.optim.AdamW(metric.parameters(), lr=lr, weight_decay=1e-4)
    Z_full = torch.tensor(embeddings, dtype=torch.float32, device=device)

    history = []
    for epoch in range(n_epochs):
        rng.shuffle(valid)
        epoch_loss = 0.0; epoch_top_gold = 0.0; n_seen = 0
        for ex in tqdm(valid, desc=f"e2e ep{epoch+1}/{n_epochs}", leave=False):
            opt.zero_grad()
            cand = torch.tensor(ex["cand"], dtype=torch.long, device=device)
            Z_c = Z_full[cand]                                       # (M, d)

            # compute differentiable C from W
            C = cost_matrix_differentiable(Z_c, metric.W)            # (M, M)

            # detached K (redundancy kernel from cosine, doesn't depend on W)
            with torch.no_grad():
                Z_n = Z_c / Z_c.norm(dim=1, keepdim=True).clamp_min(1e-8)
                K = torch.clamp(Z_n @ Z_n.t(), min=0.0)

            # energy from blended scores
            dense = torch.tensor(ex["dense"], dtype=torch.float32, device=device)
            rerank = torch.tensor(ex["rerank"], dtype=torch.float32, device=device)
            dense_n = (dense - dense.min()) / (dense.max() - dense.min() + 1e-8)
            rerank_n = (rerank - rerank.min()) / (rerank.max() - rerank.min() + 1e-8)
            energy = -(0.4 * dense_n + 0.6 * rerank_n)

            # initial distribution
            with torch.no_grad():
                p0 = F.softmax(-energy / tau0, dim=0)

            # Unrolled JKO (typically T=1 for tractability)
            p_T = p0
            for _ in range(T_unroll):
                p_T = unrolled_jko_step(
                    p_T, energy, C, K,
                    h=h_jko, lam=lam_jko, rho=rho_jko,
                    eps_sinkhorn=eps_sinkhorn, n_inner=n_inner,
                )

            # pairwise loss on gold vs non-gold in pool
            gold_mask = torch.zeros(len(cand), dtype=torch.bool, device=device)
            cand_np = cand.cpu().numpy()
            for g in ex["gold_in_pool"]:
                pos = np.where(cand_np == g)[0]
                if len(pos): gold_mask[int(pos[0])] = True
            loss = pairwise_logistic_loss(p_T, gold_mask, tau=tau_loss)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(metric.parameters(), 5.0)
            opt.step()

            with torch.no_grad():
                top_idx = int(torch.argmax(p_T))
                top_is_gold = float(gold_mask[top_idx])
                epoch_loss += float(loss); epoch_top_gold += top_is_gold; n_seen += 1
        if n_seen > 0:
            avg = epoch_loss / n_seen
            top1 = epoch_top_gold / n_seen
            history.append({"epoch": epoch + 1, "loss": avg, "top1_acc": top1})
            print(f"  ep {epoch+1:>3d}  loss={avg:.4f}  top1_acc={top1:.3f}")

    out_dir = index_dir(dataset)
    out_path = out_dir / "learned_metric_e2e.pt"
    torch.save({"W": metric.W.detach().cpu(), "r": r, "d": d, "history": history,
                "dataset": dataset, "split_train": split_train, "n_train": len(valid),
                "training": "end-to-end JKO-unroll, T={}, inner={}".format(T_unroll, n_inner)},
               out_path)
    print(f"[NM-JKO E2E] saved {out_path}")
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--T-unroll", type=int, default=1)
    p.add_argument("--n-inner", type=int, default=6)
    args = p.parse_args()
    train_e2e(args.dataset, args.split, n_train=args.n_train,
              n_epochs=args.n_epochs, lr=args.lr,
              T_unroll=args.T_unroll, n_inner=args.n_inner)
