"""Determinantal Point Process (DPP) retrieval baseline.

We implement greedy MAP-DPP retrieval: select k items that approximately
maximise det(K_S), where K is a relevance-quality-diversity kernel.

Kernel construction (standard L-ensemble formulation):
  K_{ij} = r_i * r_j * cos(z_i, z_j)
  K_{ii} += δ   (regularisation for numerical PSD)

where r_i = min-max normalised relevance score and z_i is the embedding.

Greedy MAP via Cholesky updates (Kulesza & Taskar 2012, Alg. 3):
  At each step, add the item i with largest conditional marginal:
    Δ(i) = K_{ii} - K_{iS} (K_{SS})^{-1} K_{iS}^T

This is O(k * M) per step using incremental Cholesky.

We compare DPP-MAP against MMR, JKO-RAG, KL-prox, and noprox
on the same candidate pool (using cached candidates_test.npz).

Usage:
  python dpp_retrieval.py --dataset scifact --split test \
      --config-file results/best_hparams_scifact.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset
from methods import Candidates
from retrieval import (
    Indices, cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax,
)
from jko import JKOConfig, run_jko
from evaluation import (
    ndcg_at_k, recall_at_k, precision_at_k, reciprocal_rank,
    semantic_diversity, bootstrap_ci, paired_bootstrap_diff,
)

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TOKEN_RE = re.compile(r"\w+")


def tokenize(t): return TOKEN_RE.findall(t.lower())


def index_dir(name):
    sub = INDEX_ROOT / name
    return sub if (sub / "doc_ids.json").exists() else INDEX_ROOT


def load_index(name: str) -> Indices:
    base = index_dir(name)
    with open(base / "doc_ids.json") as f: doc_ids = json.load(f)
    with open(base / "doc_texts.json") as f: doc_texts = json.load(f)
    embeddings = np.load(base / "embeddings.npy")
    with open(base / "bm25.pkl", "rb") as f: bm25_data = pickle.load(f)
    return Indices(doc_ids=doc_ids, doc_id_to_idx={d: i for i, d in enumerate(doc_ids)},
                   doc_texts=doc_texts, embeddings=embeddings,
                   bm25=bm25_data["bm25"], bm25_tokenized=bm25_data["tokenized"])


def load_cache(name: str, split: str = "test"):
    npz = np.load(index_dir(name) / f"candidates_{split}.npz", allow_pickle=True)
    return {
        "cand_idx": npz["cand_idx"], "bm25_pool": npz["bm25_pool"],
        "dense_pool": npz["dense_pool"], "rerank": npz["rerank"],
        "q_ids": [str(x) for x in npz["q_ids"]],
    }


# ---------------------------------------------------------------------------
# DPP greedy MAP selection
# ---------------------------------------------------------------------------

def dpp_greedy_topk(embeddings: np.ndarray, relevance: np.ndarray,
                    k: int, delta: float = 1e-3) -> list[int]:
    """Greedy MAP-DPP selection via Gram-Schmidt residuals.

    Implements the L-ensemble DPP with kernel L_{ij} = r_i * r_j * z_i^T z_j.
    We maintain residual vectors R[i, :] = V_i - (projection onto selected directions),
    so ||R_i||^2 + delta is the Schur complement / conditional marginal for item i.
    Greedy selection picks the item with largest conditional marginal at each step.

    Time: O(k * M * d).  Space: O(M * d).  For M=200, d=384, k=20: ~1.5M ops.

    Parameters
    ----------
    embeddings : (M, d) normalised embeddings of candidate items
    relevance  : (M,) normalised relevance scores in [0, 1]
    k          : number of items to select
    delta      : regularisation for numerical PSD guarantee

    Returns
    -------
    selected : list of k indices into the candidate pool
    """
    r = np.asarray(relevance, dtype=np.float64).clip(1e-8, None)
    Z = np.asarray(embeddings, dtype=np.float64)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-8)
    M = len(r)

    # V_i = r_i * z_i  (the quality-scaled embedding)
    R = (r[:, None] * Z).copy()    # residuals, initially = V, shape (M, d)
    d_arr = np.sum(R ** 2, axis=1) + delta  # conditional marginals, shape (M,)

    selected: list[int] = []
    selected_mask = np.zeros(M, dtype=bool)

    for _ in range(k):
        d_arr[selected_mask] = -np.inf
        if d_arr.max() <= 0:
            break
        best = int(np.argmax(d_arr))
        selected.append(best)
        selected_mask[best] = True

        sqrt_d = float(np.sqrt(d_arr[best]))
        if sqrt_d < 1e-12:
            break

        # New orthonormal direction u = R[best] / ||R[best]||
        u = R[best] / sqrt_d                         # shape (d,)
        proj = R @ u                                  # projections onto u, shape (M,)
        R -= proj[:, None] * u[None, :]              # Gram-Schmidt deflation
        d_arr = np.sum(R ** 2, axis=1) + delta       # update marginals

    # Pad if needed (e.g., all remaining marginals went to 0)
    if len(selected) < k:
        remaining = sorted([i for i in range(M) if i not in set(selected)],
                           key=lambda i: -r[i])
        selected.extend(remaining[:k - len(selected)])

    return selected[:k]


# ---------------------------------------------------------------------------
# Other baselines (copied / adapted from run_full_dataset.py)
# ---------------------------------------------------------------------------

def bm25_topk_full(idx, q, k):
    s = idx.bm25.get_scores(tokenize(q)).astype(np.float32)
    return np.argsort(-s)[:k].tolist()


def dense_topk_full(idx, q_emb, k):
    s = (idx.embeddings @ q_emb).astype(np.float32)
    return np.argsort(-s)[:k].tolist()


def hybrid_rrf_pool(c, k):
    rrf = np.zeros(len(c.cand_idx), dtype=np.float64); K = 60
    for r, i in enumerate(np.argsort(-c.bm25_scores)): rrf[i] += 1.0 / (K + r + 1)
    for r, i in enumerate(np.argsort(-c.dense_scores)): rrf[i] += 1.0 / (K + r + 1)
    return c.cand_idx[np.argsort(-rrf)[:k]].tolist()


def rerank_topk_pool(c, k):
    return c.cand_idx[np.argsort(-c.rerank_scores)[:k]].tolist()


def mmr_pool(c, idx, k, lam=0.5):
    Z = idx.embeddings[c.cand_idx]
    rel = normalize_minmax(c.rerank_scores); sim = Z @ Z.T
    selected = []; remaining = set(range(len(c.cand_idx)))
    while len(selected) < min(k, len(c.cand_idx)):
        best_i, best_score = -1, -1e18
        for i in remaining:
            if not selected: score = rel[i]
            else:
                div = max(sim[i, j] for j in selected)
                score = lam * rel[i] - (1 - lam) * div
            if score > best_score: best_score, best_i = score, i
        selected.append(best_i); remaining.discard(best_i)
    return c.cand_idx[selected].tolist()


def dpp_topk(c, idx, k, blend_alpha=0.4, blend_gamma=0.6, delta=1e-3):
    """DPP-MAP selection from candidate pool."""
    Z = idx.embeddings[c.cand_idx]
    rel = normalize_minmax(blend_alpha * normalize_minmax(c.dense_scores)
                           + blend_gamma * normalize_minmax(c.rerank_scores))
    selected_local = dpp_greedy_topk(Z, rel, k=k, delta=delta)
    return c.cand_idx[selected_local].tolist()


def jko_topk(c, idx, mode, alpha, gamma, k, cfg):
    Z = idx.embeddings[c.cand_idx]
    C = cost_matrix_cosine(Z); K = redundancy_kernel(Z)
    energy = -(alpha * normalize_minmax(c.dense_scores) + gamma * normalize_minmax(c.rerank_scores))
    p0 = softmax_np(-energy, tau=cfg["tau0"])
    jcfg = JKOConfig(h=cfg["h"], lam=cfg["lam"], rho=cfg["rho"],
                     sinkhorn_eps=cfg["sinkhorn_eps"], T=cfg["T"],
                     inner_steps=cfg["inner_steps"], mode=mode)
    p_T, _ = run_jko(p0, energy, C, K, jcfg)
    return c.cand_idx[np.argsort(-p_T)[:k]].tolist()


def evaluate(topk_idx, idx, qrels, embeddings):
    dids = [idx.doc_ids[i] for i in topk_idx]
    return {
        "ndcg@10":      ndcg_at_k(dids, qrels, 10),
        "recall@5":     recall_at_k(dids, qrels, 5),
        "recall@10":    recall_at_k(dids, qrels, 10),
        "recall@20":    recall_at_k(dids, qrels, 20),
        "precision@5":  precision_at_k(dids, qrels, 5),
        "mrr@10":       reciprocal_rank(dids, qrels, 10),
        "diversity@10": semantic_diversity(topk_idx[:10], embeddings),
    }


METHOD_NAMES = ["rerank", "mmr", "dpp_map", "noprox_blend", "kl_blend", "jko_blend"]


def main():
    global INDEX_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--config-file", default=None)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--index-root", default=None)
    args = p.parse_args()

    if args.index_root:
        INDEX_ROOT = Path(args.index_root).resolve()

    print(f"=== DPP retrieval: {args.dataset}/{args.split} ===")
    ds = load_dataset(args.dataset)
    qrels = ds.qrels[args.split]
    idx = load_index(args.dataset)
    cache = load_cache(args.dataset, args.split)
    q_ids = cache["q_ids"]

    base_cfg = {"h": 0.5, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
                "T": 3, "inner_steps": 25, "tau0": 0.1}
    if args.config_file:
        loaded = json.loads(Path(args.config_file).read_text())
        if "best" in loaded: base_cfg.update(loaded["best"]["cfg"])
        else: base_cfg.update(loaded)
    print(f"Config: {base_cfg}")

    q_emb_all = np.load(INDEX_ROOT / args.dataset / f"q_embeddings_{args.split}.npy"
                        if (INDEX_ROOT / args.dataset).exists()
                        else INDEX_ROOT / f"q_embeddings_{args.split}.npy")
    qid_path = (INDEX_ROOT / args.dataset / f"q_ids_{args.split}.json"
                if (INDEX_ROOT / args.dataset).exists()
                else INDEX_ROOT / f"q_ids_{args.split}.json")
    with open(qid_path) as f: emb_qids = json.load(f)
    q_to_i_emb = {q: i for i, q in enumerate(emb_qids)}

    per_query = {m: defaultdict(list) for m in METHOD_NAMES}
    t0 = time.time()
    k = 20
    for qi, qid in enumerate(tqdm(q_ids, desc=f"dpp/{args.dataset}")):
        if qid not in qrels: continue
        if qid not in q_to_i_emb: continue
        q_emb = q_emb_all[q_to_i_emb[qid]]
        c = Candidates(cand_idx=cache["cand_idx"][qi], bm25_scores=cache["bm25_pool"][qi],
                       dense_scores=cache["dense_pool"][qi], rerank_scores=cache["rerank"][qi])
        tops = {
            "rerank":        rerank_topk_pool(c, k),
            "mmr":           mmr_pool(c, idx, k, lam=0.5),
            "dpp_map":       dpp_topk(c, idx, k, blend_alpha=0.4, blend_gamma=0.6),
            "noprox_blend":  jko_topk(c, idx, "noproximal",  0.4, 0.6, k, base_cfg),
            "kl_blend":      jko_topk(c, idx, "kl",          0.4, 0.6, k, base_cfg),
            "jko_blend":     jko_topk(c, idx, "wasserstein", 0.4, 0.6, k, base_cfg),
        }
        for m, doc_indices in tops.items():
            ms = evaluate(doc_indices, idx, qrels[qid], idx.embeddings)
            for metric, v in ms.items():
                per_query[m][metric].append(v)
    elapsed = time.time() - t0

    summary = {}
    for m, d in per_query.items():
        for metric, scores in d.items():
            mean, lo, hi = bootstrap_ci(scores)
            summary.setdefault(m, {})[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                                                  "n": len(scores)}

    paired = {}
    for pair in [("jko_blend", "dpp_map"), ("jko_blend", "mmr"),
                 ("dpp_map", "mmr"), ("jko_blend", "rerank")]:
        a, b = pair
        paired[f"{a}_vs_{b}"] = {}
        for metric in ("ndcg@10", "recall@10", "diversity@10"):
            diff, lo, hi = paired_bootstrap_diff(per_query[a][metric], per_query[b][metric])
            paired[f"{a}_vs_{b}"][metric] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}

    out = {
        "dataset": args.dataset, "split": args.split, "config": base_cfg,
        "n_queries": sum(1 for q in q_ids if q in qrels),
        "elapsed_sec": elapsed, "summary": summary, "paired": paired,
        "per_query": {m: dict(d) for m, d in per_query.items()},
    }
    out_name = f"dpp_{args.dataset}{args.out_suffix}.json"
    (RESULTS_DIR / out_name).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {RESULTS_DIR / out_name}")

    print(f"\n=== DPP vs baselines ({args.dataset}/{args.split}) ===")
    print(f"{'method':<18s}  nDCG@10  Recall@10  Recall@20  Div@10")
    for m in METHOD_NAMES:
        s = summary[m]
        print(f"{m:<18s}  {s['ndcg@10']['mean']:.3f}[{s['ndcg@10']['ci_lo']:.3f},{s['ndcg@10']['ci_hi']:.3f}]"
              f"  {s['recall@10']['mean']:.3f}  {s['recall@20']['mean']:.3f}"
              f"  {s['diversity@10']['mean']:.3f}")

    print("\nPaired diffs:")
    for k_pair, mvals in paired.items():
        sig = "*" if (mvals['ndcg@10']['ci_lo'] > 0 or mvals['ndcg@10']['ci_hi'] < 0) else " "
        print(f"  {k_pair}: ndcg@10 diff={mvals['ndcg@10']['diff']:+.4f}"
              f"  [{mvals['ndcg@10']['ci_lo']:+.4f},{mvals['ndcg@10']['ci_hi']:+.4f}] {sig}")


if __name__ == "__main__":
    main()
