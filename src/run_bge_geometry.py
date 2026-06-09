"""BGE Geometry Ablation.

Tests whether replacing the MiniLM cost matrix with BGE-small-en-v1.5 embeddings
improves JKO-RAG performance — WITHOUT rerunning retrieval or reranking.

Protocol:
  - Candidate pool + scores (bm25/dense/rerank) from indices/<dataset>/ (MiniLM)
  - Cost matrix C_{ij} = (1 - cos(z_i, z_j))^2 uses embeddings from indices_bge/
  - Compare: jko_minilm_geom (old C) vs jko_bge_geom (new C) vs kl_blend (no geometry)

This isolates the contribution of semantic geometry quality in the cost matrix,
independent of pool construction or reranker quality.

Note: For BGE-small-en-v1.5, embeddings are in 384-d (same as MiniLM),
      but trained with MSMARCO + sentence-transformers contrastive pairs.
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
    ndcg_at_k, recall_at_k, semantic_diversity, bootstrap_ci, paired_bootstrap_diff,
)

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
BGE_ROOT   = Path(__file__).resolve().parents[1] / "indices_bge"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TOKEN_RE = re.compile(r"\w+")


def tokenize(t): return TOKEN_RE.findall(t.lower())


def index_dir(name, root=None):
    root = root or INDEX_ROOT
    sub = root / name
    return sub if (sub / "doc_ids.json").exists() else root


def load_index(name: str, root=None) -> Indices:
    base = index_dir(name, root)
    with open(base / "doc_ids.json") as f: doc_ids = json.load(f)
    with open(base / "doc_texts.json") as f: doc_texts = json.load(f)
    embeddings = np.load(base / "embeddings.npy")
    with open(base / "bm25.pkl", "rb") as f: bm25_data = pickle.load(f)
    return Indices(doc_ids=doc_ids, doc_id_to_idx={d: i for i, d in enumerate(doc_ids)},
                   doc_texts=doc_texts, embeddings=embeddings,
                   bm25=bm25_data["bm25"], bm25_tokenized=bm25_data["tokenized"])


def load_cache(name: str, split: str = "test", root=None):
    npz = np.load(index_dir(name, root) / f"candidates_{split}.npz", allow_pickle=True)
    return {
        "cand_idx": npz["cand_idx"], "bm25_pool": npz["bm25_pool"],
        "dense_pool": npz["dense_pool"], "rerank": npz["rerank"],
        "q_ids": [str(x) for x in npz["q_ids"]],
    }


def load_bge_embeddings(name: str) -> np.ndarray:
    """Load BGE-small corpus embeddings (aligned to same doc_ids as MiniLM index)."""
    path = BGE_ROOT / name / "embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"BGE embeddings not found at {path}")
    return np.load(path).astype(np.float32)


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


def jko_topk_with_emb(c, embeddings, mode, alpha, gamma, k, cfg):
    """JKO with explicit embeddings matrix (can be MiniLM or BGE)."""
    Z = embeddings[c.cand_idx]
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
        "recall@10":    recall_at_k(dids, qrels, 10),
        "recall@20":    recall_at_k(dids, qrels, 20),
        "diversity@10": semantic_diversity(topk_idx[:10], embeddings),
    }


# (dataset, bge_available) — BGE embeddings available for all 4
METHOD_NAMES = ["rerank", "mmr", "kl_blend", "noprox_blend",
                "jko_minilm_geom", "jko_bge_geom"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--config-file", default=None)
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    print(f"=== BGE geometry ablation: {args.dataset}/{args.split} ===")
    ds = load_dataset(args.dataset)
    qrels = ds.qrels[args.split]
    idx_ml = load_index(args.dataset)                # MiniLM index
    cache   = load_cache(args.dataset, args.split)
    q_ids   = cache["q_ids"]

    # Load BGE corpus embeddings (aligned to MiniLM doc_ids)
    bge_emb = load_bge_embeddings(args.dataset)
    print(f"MiniLM emb shape: {idx_ml.embeddings.shape}, BGE emb shape: {bge_emb.shape}")

    base_cfg = {"h": 0.5, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
                "T": 3, "inner_steps": 25, "tau0": 0.1}
    if args.config_file:
        loaded = json.loads(Path(args.config_file).read_text())
        if "best" in loaded: base_cfg.update(loaded["best"]["cfg"])
        else: base_cfg.update(loaded)
    # Strip non-JKO keys
    jko_keys = {"h", "lam", "rho", "sinkhorn_eps", "T", "inner_steps", "tau0"}
    jko_cfg = {k: v for k, v in base_cfg.items() if k in jko_keys}
    print(f"JKO config: {jko_cfg}")

    # Query embeddings (MiniLM, for BM25/dense lookups — not used for cost matrix here)
    q_emb_base = index_dir(args.dataset)
    q_emb_all = np.load(q_emb_base / f"q_embeddings_{args.split}.npy")
    with open(q_emb_base / f"q_ids_{args.split}.json") as f: emb_qids = json.load(f)
    q_to_i_emb = {q: i for i, q in enumerate(emb_qids)}

    per_query = {m: defaultdict(list) for m in METHOD_NAMES}
    t0 = time.time()
    k = 20
    for qi, qid in enumerate(tqdm(q_ids, desc=f"bge_geom/{args.dataset}")):
        if qid not in qrels: continue
        if qid not in q_to_i_emb: continue
        c = Candidates(cand_idx=cache["cand_idx"][qi], bm25_scores=cache["bm25_pool"][qi],
                       dense_scores=cache["dense_pool"][qi], rerank_scores=cache["rerank"][qi])
        tops = {
            "rerank":          rerank_topk_pool(c, k),
            "mmr":             mmr_pool(c, idx_ml, k, lam=0.5),
            "kl_blend":        jko_topk_with_emb(c, idx_ml.embeddings, "kl",
                                                  0.4, 0.6, k, jko_cfg),
            "noprox_blend":    jko_topk_with_emb(c, idx_ml.embeddings, "noproximal",
                                                  0.4, 0.6, k, jko_cfg),
            "jko_minilm_geom": jko_topk_with_emb(c, idx_ml.embeddings, "wasserstein",
                                                  0.4, 0.6, k, jko_cfg),
            "jko_bge_geom":    jko_topk_with_emb(c, bge_emb,           "wasserstein",
                                                  0.4, 0.6, k, jko_cfg),
        }
        for m, doc_indices in tops.items():
            ms = evaluate(doc_indices, idx_ml, qrels[qid], idx_ml.embeddings)
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
    for pair in [("jko_bge_geom", "jko_minilm_geom"),
                 ("jko_bge_geom", "kl_blend"),
                 ("jko_minilm_geom", "kl_blend")]:
        a, b = pair
        paired[f"{a}_vs_{b}"] = {}
        for metric in ("ndcg@10", "recall@10", "diversity@10"):
            diff, lo, hi = paired_bootstrap_diff(per_query[a][metric], per_query[b][metric])
            paired[f"{a}_vs_{b}"][metric] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}

    out = {
        "dataset": args.dataset, "split": args.split, "config": jko_cfg,
        "n_queries": sum(1 for q in q_ids if q in qrels),
        "elapsed_sec": elapsed, "summary": summary, "paired": paired,
        "per_query": {m: dict(d) for m, d in per_query.items()},
    }
    out_name = f"bge_geometry_{args.dataset}{args.out_suffix}.json"
    (RESULTS_DIR / out_name).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {RESULTS_DIR / out_name}")

    print(f"\n=== BGE geometry: {args.dataset}/{args.split} ===")
    print(f"{'method':<22s}  nDCG@10  R@10  Div@10")
    for m in METHOD_NAMES:
        s = summary[m]
        print(f"{m:<22s}  {s['ndcg@10']['mean']:.3f}[{s['ndcg@10']['ci_lo']:.3f},"
              f"{s['ndcg@10']['ci_hi']:.3f}]  {s['recall@10']['mean']:.3f}  "
              f"{s['diversity@10']['mean']:.3f}")
    print("\nKey paired diffs:")
    for kk, mvals in paired.items():
        d = mvals['ndcg@10']
        sig = "*" if (d['ci_lo'] > 0 or d['ci_hi'] < 0) else " "
        print(f"  {kk}: {d['diff']:+.4f} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}] {sig}")


if __name__ == "__main__":
    main()
