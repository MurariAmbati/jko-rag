"""Stage 1 on SciFact using the fine-tuned MiniLM retriever (indices/scifact_ft).

Reuses SciFact data (queries, qrels) but loads embeddings + candidate pools
from indices/scifact_ft.
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
FT_DIR = INDEX_ROOT / "scifact_ft"
TOKEN_RE = re.compile(r"\w+")


def tokenize(t): return TOKEN_RE.findall(t.lower())


def load_ft_index() -> Indices:
    with open(FT_DIR / "doc_ids.json") as f: doc_ids = json.load(f)
    with open(FT_DIR / "doc_texts.json") as f: doc_texts = json.load(f)
    embeddings = np.load(FT_DIR / "embeddings.npy")
    with open(FT_DIR / "bm25.pkl", "rb") as f: bm25_data = pickle.load(f)
    return Indices(
        doc_ids=doc_ids, doc_id_to_idx={d: i for i, d in enumerate(doc_ids)},
        doc_texts=doc_texts, embeddings=embeddings,
        bm25=bm25_data["bm25"], bm25_tokenized=bm25_data["tokenized"],
    )


def load_cache():
    npz = np.load(FT_DIR / "candidates_test.npz", allow_pickle=True)
    return {"cand_idx": npz["cand_idx"], "bm25_pool": npz["bm25_pool"],
            "dense_pool": npz["dense_pool"], "rerank": npz["rerank"],
            "q_ids": [str(x) for x in npz["q_ids"]]}


def bm25_topk_full(idx, q, k):
    s = idx.bm25.get_scores(tokenize(q)).astype(np.float32)
    return np.argsort(-s)[:k].tolist()


def dense_topk_full(idx, q_emb, k):
    s = (idx.embeddings @ q_emb).astype(np.float32)
    return np.argsort(-s)[:k].tolist()


def hybrid_rrf(c, k):
    rrf = np.zeros(len(c.cand_idx), dtype=np.float64); K = 60
    for r, i in enumerate(np.argsort(-c.bm25_scores)): rrf[i] += 1.0 / (K + r + 1)
    for r, i in enumerate(np.argsort(-c.dense_scores)): rrf[i] += 1.0 / (K + r + 1)
    return c.cand_idx[np.argsort(-rrf)[:k]].tolist()


def rerank_topk(c, k): return c.cand_idx[np.argsort(-c.rerank_scores)[:k]].tolist()


def mmr(c, idx, k, lam=0.5):
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


METHOD_NAMES = ["bm25", "dense", "hybrid_rrf", "rerank", "mmr",
                "noprox_blend", "kl_blend", "jko_blend",
                "noprox_blend_dense", "kl_blend_dense", "jko_blend_dense"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default=None)
    p.add_argument("--out-suffix", default="_ft")
    args = p.parse_args()

    print("Loading SciFact data + FT index...")
    ds = load_dataset("scifact")
    qrels = ds.qrels["test"]
    idx = load_ft_index()
    cache = load_cache()
    q_ids = cache["q_ids"]

    base_cfg = {"h": 0.5, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
                "T": 3, "inner_steps": 25, "tau0": 0.1}
    if args.config_file:
        c = json.loads(Path(args.config_file).read_text())
        base_cfg.update(c.get("best", {}).get("cfg", c))
    print(f"Base config: {base_cfg}")

    q_emb_all = np.load(FT_DIR / "q_embeddings_test.npy")
    with open(FT_DIR / "q_ids_test.json") as f: emb_qids = json.load(f)
    q_to_i_emb = {q: i for i, q in enumerate(emb_qids)}

    per_query = {m: defaultdict(list) for m in METHOD_NAMES}
    t0 = time.time()
    for qi, qid in enumerate(tqdm(q_ids, desc="ft scifact")):
        if qid not in qrels: continue
        q_emb = q_emb_all[q_to_i_emb[qid]]
        c = Candidates(cand_idx=cache["cand_idx"][qi], bm25_scores=cache["bm25_pool"][qi],
                       dense_scores=cache["dense_pool"][qi], rerank_scores=cache["rerank"][qi])
        k_max = 20
        tops = {
            "bm25":       bm25_topk_full(idx, ds.queries[qid], k_max),
            "dense":      dense_topk_full(idx, q_emb, k_max),
            "hybrid_rrf": hybrid_rrf(c, k_max),
            "rerank":     rerank_topk(c, k_max),
            "mmr":        mmr(c, idx, k_max, 0.5),
        }
        common = dict(k=k_max, cfg=base_cfg)
        for name, mode, alpha, gamma in [
            ("noprox_blend",       "noproximal",  0.4, 0.6),
            ("kl_blend",           "kl",          0.4, 0.6),
            ("jko_blend",          "wasserstein", 0.4, 0.6),
            ("noprox_blend_dense", "noproximal",  0.7, 0.3),
            ("kl_blend_dense",     "kl",          0.7, 0.3),
            ("jko_blend_dense",    "wasserstein", 0.7, 0.3),
        ]:
            tops[name] = jko_topk(c, idx, mode, alpha, gamma, **common)
        for m, doc_indices in tops.items():
            ms = evaluate(doc_indices, idx, qrels[qid], idx.embeddings)
            for metric, v in ms.items():
                per_query[m][metric].append(v)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    summary = {}
    for m, d in per_query.items():
        for metric, scores in d.items():
            mean, lo, hi = bootstrap_ci(scores)
            summary.setdefault(m, {})[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}

    paired = {}
    for pair in [("jko_blend", "kl_blend"), ("jko_blend", "rerank"),
                 ("jko_blend", "hybrid_rrf"), ("jko_blend", "noprox_blend")]:
        a, b = pair
        if a not in per_query or b not in per_query: continue
        paired[f"{a}_vs_{b}"] = {}
        for metric in ("ndcg@10", "recall@10", "recall@20", "diversity@10"):
            diff, lo, hi = paired_bootstrap_diff(per_query[a][metric], per_query[b][metric])
            paired[f"{a}_vs_{b}"][metric] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}

    # Pool recall
    in_pool, total = 0, 0
    for qi, qid in enumerate(q_ids):
        if qid not in qrels: continue
        rel = {d for d, r in qrels[qid].items() if r > 0}
        if not rel: continue
        pool_dids = {idx.doc_ids[int(j)] for j in cache["cand_idx"][qi]}
        in_pool += len(rel & pool_dids); total += len(rel)

    out = {
        "dataset": "scifact_ft", "split": "test",
        "n_queries": sum(1 for q in q_ids if q in qrels),
        "config": base_cfg,
        "pool_recall_micro": in_pool / max(1, total),
        "elapsed_sec": elapsed,
        "summary": summary, "paired": paired,
        "per_query": {m: dict(d) for m, d in per_query.items()},
    }
    out_name = f"stage1_scifact_ft{args.out_suffix}.json"
    (RESULTS_DIR / out_name).write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved {RESULTS_DIR / out_name}")
    print(f"Pool recall (micro): {out['pool_recall_micro']:.4f}")

    print(f"\n=== Fine-tuned retriever results ===")
    print(f"{'method':<22s}  {'nDCG@10':>22s}  {'Recall@10':>22s}  {'Recall@20':>22s}")
    for m in METHOD_NAMES:
        if m not in summary: continue
        s = summary[m]
        def cell(d): return f"{d['mean']:.3f}[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"
        print(f"{m:<22s}  {cell(s['ndcg@10']):>22s}  {cell(s['recall@10']):>22s}  {cell(s['recall@20']):>22s}")

    print("\nPaired diffs:")
    for k, mvals in paired.items():
        print(f"  {k}:")
        for metric, d in mvals.items():
            sig = "*" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else " "
            print(f"    {metric:<14s}  diff={d['diff']:+.4f}  [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  {sig}")


if __name__ == "__main__":
    main()
