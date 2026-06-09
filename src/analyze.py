"""Post-hoc analysis of Stage 1 results.

Computes:
- Pool recall ceiling (oracle: best any pool-restricted method could do)
- Paired W vs KL diffs (decisive ablation) per metric
- Where Wasserstein wins / loses most relative to KL (top examples)
- Distribution-level stats: avg entropy of p_T, top-5 mass concentration
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from download_data import load_scifact
from retrieval import load_indices
from evaluation import paired_bootstrap_diff, bootstrap_ci

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
INDEX_DIR = Path(__file__).resolve().parents[1] / "indices"


def pool_recall_ceiling(idx, qrels: dict, cache_arr: np.ndarray, q_ids: list[str], k: int = 20):
    """For each test query, fraction of relevant docs that lie in the M=200 pool, restricted to top-k order via... actually the ceiling is whether the relevant docs are in the pool at all (top-k chosen by an oracle from the pool)."""
    in_pool, total = 0, 0
    per_query = []
    for i, qid in enumerate(q_ids):
        if qid not in qrels:
            continue
        rel = {d for d, r in qrels[qid].items() if r > 0}
        if not rel:
            continue
        pool_dids = {idx.doc_ids[int(j)] for j in cache_arr[i]}
        hits = len(rel & pool_dids)
        in_pool += hits
        total += len(rel)
        per_query.append(hits / len(rel))
    return {
        "macro_recall_in_pool": float(np.mean(per_query)),
        "micro_recall_in_pool": in_pool / max(1, total),
        "n": len(per_query),
    }


def main():
    s1 = json.loads((RESULTS_DIR / "stage1.json").read_text())
    print(f"Loaded results for n={s1['n_queries']} queries, elapsed={s1['elapsed_sec']:.1f}s\n")

    # Pool recall ceiling
    print("=" * 70)
    print("Pool recall ceiling (max fraction of relevant docs reachable in M=200)")
    print("=" * 70)
    _, _, qrels_test, _ = load_scifact()
    idx = load_indices()
    npz = np.load(INDEX_DIR / "candidates_test.npz", allow_pickle=True)
    q_ids = [str(x) for x in npz["q_ids"]]
    ceil = pool_recall_ceiling(idx, qrels_test, npz["cand_idx"], q_ids)
    print(f"  macro pool recall: {ceil['macro_recall_in_pool']:.4f}")
    print(f"  micro pool recall: {ceil['micro_recall_in_pool']:.4f}")
    print(f"  (this is the upper bound for any method that runs on the pool)\n")

    # Headline table
    print("=" * 70)
    print("HEADLINE RESULTS (mean over n={} queries with 95% bootstrap CI)".format(s1['n_queries']))
    print("=" * 70)
    print(f"{'method':<22s} {'nDCG@10':>20s} {'Recall@10':>20s} {'Recall@20':>20s} {'Div@10':>12s}")
    for m, s in s1["summary"].items():
        def cell(d): return f"{d['mean']:.3f}[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"
        print(f"{m:<22s} {cell(s['ndcg@10']):>20s} {cell(s['recall@10']):>20s} "
              f"{cell(s['recall@20']):>20s} {cell(s['diversity@10']):>12s}")

    # Decisive ablation: W vs KL (same energy)
    print()
    print("=" * 70)
    print("DECISIVE ABLATION: Wasserstein-prox vs KL-prox (same energy, same hyperparams)")
    print("=" * 70)
    if "jko_rerank" in s1["per_query"] and "kl_rerank" in s1["per_query"]:
        for metric in ["ndcg@10", "recall@5", "recall@10", "recall@20", "diversity@10", "mrr@10"]:
            a = s1["per_query"]["jko_rerank"][metric]
            b = s1["per_query"]["kl_rerank"][metric]
            diff, lo, hi = paired_bootstrap_diff(a, b)
            sig = "  *" if (lo > 0 or hi < 0) else "   "
            print(f"  {metric:<14s} jko - kl = {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")

    # Decisive ablation: blended energy variants
    print()
    print("=" * 70)
    print("WASSERSTEIN: rerank-energy vs blended-energy")
    print("=" * 70)
    for variant in ["jko_blend", "jko_blend_dense"]:
        if variant not in s1["per_query"]:
            continue
        print(f"  {variant} vs jko_rerank:")
        for metric in ["ndcg@10", "recall@10", "recall@20"]:
            a = s1["per_query"][variant][metric]
            b = s1["per_query"]["jko_rerank"][metric]
            diff, lo, hi = paired_bootstrap_diff(a, b)
            sig = "  *" if (lo > 0 or hi < 0) else "   "
            print(f"    {metric:<14s} diff = {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")
        print()

    # All paired comparisons vs hybrid (a fair "no reranker" baseline)
    print("=" * 70)
    print("Paired diff vs `hybrid_rrf` on nDCG@10 (does the reranker / WFE matter?)")
    print("=" * 70)
    base = s1["per_query"]["hybrid_rrf"]["ndcg@10"]
    for m in s1["per_query"]:
        if m == "hybrid_rrf":
            continue
        a = s1["per_query"][m]["ndcg@10"]
        diff, lo, hi = paired_bootstrap_diff(a, base)
        sig = "  *" if (lo > 0 or hi < 0) else "   "
        print(f"  {m:<22s} diff = {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")

    # Where does Wasserstein win/lose vs KL?
    print()
    print("=" * 70)
    print("WHERE does Wasserstein beat KL most on nDCG@10? (top 10)")
    print("=" * 70)
    if "jko_rerank" in s1["per_query"] and "kl_rerank" in s1["per_query"]:
        a = np.asarray(s1["per_query"]["jko_rerank"]["ndcg@10"])
        b = np.asarray(s1["per_query"]["kl_rerank"]["ndcg@10"])
        diff = a - b
        # Index back to qid
        order = np.argsort(-diff)
        print("  Largest positive (W > KL):")
        for i in order[:5]:
            print(f"    qid_idx={i}  W_ndcg={a[i]:.3f}  KL_ndcg={b[i]:.3f}  diff={diff[i]:+.3f}")
        print("  Largest negative (W < KL):")
        for i in order[-5:][::-1]:
            print(f"    qid_idx={i}  W_ndcg={a[i]:.3f}  KL_ndcg={b[i]:.3f}  diff={diff[i]:+.3f}")
        print(f"  Queries where W > KL strictly: {(diff > 0).sum()}/{len(diff)}")
        print(f"  Queries where W < KL strictly: {(diff < 0).sum()}/{len(diff)}")
        print(f"  Queries where W = KL:        {(diff == 0).sum()}/{len(diff)}")


if __name__ == "__main__":
    main()
