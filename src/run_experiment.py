"""Stage 1 retrieval experiment on SciFact test set.

Runs every method on each query, computes per-query metrics, saves to
results/stage1.json with mean + bootstrap CIs and paired diffs against the
strongest baseline.

Methods evaluated:
- bm25                : full-corpus BM25 top-k
- dense               : full-corpus dense top-k
- hybrid_rrf          : pool RRF top-k
- rerank              : hybrid + cross-encoder top-k
- mmr                 : MMR over rerank scores
- noprox_rerank       : F = energy(rerank) + lambda*H + redundancy   (no proximal)
- kl_rerank           : KL-proximal version
- jko_rerank          : Wasserstein-proximal (the headline method)
- jko_blend           : energy uses 0.4*dense + 0.6*rerank, W^2 proximal
- jko_blend_dense_eng : same but heavy on dense (claim-style queries)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from download_data import load_scifact
from retrieval import (
    Indices, load_indices, bm25_scores_for_query, dense_scores_for_query,
    cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax,
)
from methods import Candidates, method_bm25, method_dense, method_hybrid_rrf, method_rerank, method_mmr, method_wfe
from jko import JKOConfig, run_jko
from evaluation import (
    ndcg_at_k, recall_at_k, precision_at_k, reciprocal_rank,
    semantic_diversity, shannon_entropy, bootstrap_ci, paired_bootstrap_diff,
)

INDEX_DIR = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_candidates_from_cache(
    cache: dict, i: int, do_rerank: bool = True,
) -> Candidates:
    return Candidates(
        cand_idx=cache["cand_idx"][i],
        bm25_scores=cache["bm25_pool"][i],
        dense_scores=cache["dense_pool"][i],
        rerank_scores=cache["rerank"][i] if do_rerank else None,
    )


def make_energy(c: Candidates, alpha: float, beta: float, gamma: float) -> np.ndarray:
    r = np.zeros(len(c.cand_idx), dtype=np.float32)
    if alpha > 0:
        r = r + alpha * normalize_minmax(c.dense_scores)
    if beta > 0:
        r = r + beta * normalize_minmax(c.bm25_scores)
    if gamma > 0 and c.rerank_scores is not None:
        r = r + gamma * normalize_minmax(c.rerank_scores)
    return -r


def make_p0(c: Candidates, tau0: float, energy: np.ndarray) -> np.ndarray:
    return softmax_np(-energy, tau=tau0)


def jko_method(
    c: Candidates,
    idx: Indices,
    mode: str,
    h: float,
    lam: float,
    rho: float,
    sinkhorn_eps: float,
    T: int,
    inner_steps: int,
    alpha: float,
    beta: float,
    gamma: float,
    tau0: float,
    k: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    Z = idx.embeddings[c.cand_idx]
    C = cost_matrix_cosine(Z)
    K = redundancy_kernel(Z)
    energy = make_energy(c, alpha, beta, gamma)
    p0 = make_p0(c, tau0, energy)
    cfg = JKOConfig(
        h=h, lam=lam, rho=rho, sinkhorn_eps=sinkhorn_eps,
        T=T, inner_steps=inner_steps, mode=mode,
    )
    p_T, _ = run_jko(p0, energy, C, K, cfg)
    order = np.argsort(-p_T)[:k]
    topk = c.cand_idx[order].tolist()
    return topk, p_T, p0


def evaluate_method(
    topk_idx: list[int],
    idx: Indices,
    qrels: dict[str, int],
    embeddings: np.ndarray,
) -> dict[str, float]:
    dids = [idx.doc_ids[i] for i in topk_idx]
    metrics = {
        "ndcg@10":   ndcg_at_k(dids, qrels, 10),
        "recall@5":  recall_at_k(dids, qrels, 5),
        "recall@10": recall_at_k(dids, qrels, 10),
        "recall@20": recall_at_k(dids, qrels, 20),
        "precision@5":  precision_at_k(dids, qrels, 5),
        "mrr@10":    reciprocal_rank(dids, qrels, 10),
        "diversity@10":  semantic_diversity(topk_idx[:10], embeddings),
    }
    return metrics


METHOD_NAMES = [
    "bm25", "dense", "hybrid_rrf", "rerank", "mmr",
    "noprox_rerank", "kl_rerank", "jko_rerank",
    "jko_blend",       # alpha=0.4 (dense), gamma=0.6 (rerank)
    "jko_blend_dense", # alpha=0.7 (dense), gamma=0.3 (rerank)
]


def run_all_methods(
    idx: Indices,
    queries: dict,
    qrels: dict,
    cache: dict,
    q_ids: list[str],
    k_max: int = 20,
) -> tuple[dict, dict, dict]:
    """Returns (per_query_scores[method][metric], retrieved[method][qid], pT[method][qid])."""
    per_query: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in METHOD_NAMES}
    retrieved: dict[str, dict[str, list[str]]] = {m: {} for m in METHOD_NAMES}
    pT_store: dict[str, dict[str, np.ndarray]] = {m: {} for m in METHOD_NAMES}

    q_emb_all = np.load(INDEX_DIR / "q_embeddings_test.npy")
    with open(INDEX_DIR / "q_ids_test.json") as f:
        cached_qids = json.load(f)
    q_to_i = {q: i for i, q in enumerate(cached_qids)}

    for qi, qid in enumerate(tqdm(q_ids, desc="queries")):
        if qid not in qrels:
            continue
        q_emb = q_emb_all[q_to_i[qid]]
        c = make_candidates_from_cache(cache, qi, do_rerank=True)

        # --- non-distributional ---
        tops = {
            "bm25":       method_bm25(idx, queries[qid], q_emb, k=k_max),
            "dense":      method_dense(idx, queries[qid], q_emb, k=k_max),
            "hybrid_rrf": method_hybrid_rrf(c, k=k_max),
            "rerank":     method_rerank(c, k=k_max),
            "mmr":        method_mmr(c, idx, k=k_max, lambda_mmr=0.5),
        }

        # --- distributional ---
        common = dict(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1,
                      T=3, inner_steps=25, tau0=0.1, k=k_max)
        # rerank-only energy: gamma=1, alpha=beta=0
        ids_no, pT_no, _ = jko_method(c, idx, "noproximal", alpha=0, beta=0, gamma=1, **common)
        ids_kl, pT_kl, _ = jko_method(c, idx, "kl",        alpha=0, beta=0, gamma=1, **common)
        ids_jk, pT_jk, _ = jko_method(c, idx, "wasserstein", alpha=0, beta=0, gamma=1, **common)
        # blend energy (0.4 dense + 0.6 rerank), W^2
        ids_bl, pT_bl, _ = jko_method(c, idx, "wasserstein", alpha=0.4, beta=0.0, gamma=0.6, **common)
        # dense-heavy energy
        ids_bd, pT_bd, _ = jko_method(c, idx, "wasserstein", alpha=0.7, beta=0.0, gamma=0.3, **common)

        tops["noprox_rerank"] = ids_no
        tops["kl_rerank"] = ids_kl
        tops["jko_rerank"] = ids_jk
        tops["jko_blend"] = ids_bl
        tops["jko_blend_dense"] = ids_bd

        pT_store["noprox_rerank"][qid] = pT_no
        pT_store["kl_rerank"][qid] = pT_kl
        pT_store["jko_rerank"][qid] = pT_jk
        pT_store["jko_blend"][qid] = pT_bl
        pT_store["jko_blend_dense"][qid] = pT_bd

        for m, doc_indices in tops.items():
            ms = evaluate_method(doc_indices, idx, qrels[qid], idx.embeddings)
            for metric, v in ms.items():
                per_query[m][metric].append(v)
            retrieved[m][qid] = [idx.doc_ids[i] for i in doc_indices[:20]]

    return per_query, retrieved, pT_store


def summarize(per_query: dict) -> dict:
    out: dict = {}
    for m, metrics in per_query.items():
        out[m] = {}
        for metric, scores in metrics.items():
            mean, lo, hi = bootstrap_ci(scores)
            out[m][metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
    return out


def paired_table(per_query: dict, baseline: str = "rerank", metric: str = "ndcg@10") -> dict:
    out: dict = {}
    if baseline not in per_query:
        return out
    base = per_query[baseline][metric]
    for m in per_query:
        if m == baseline:
            continue
        diff, lo, hi = paired_bootstrap_diff(per_query[m][metric], base)
        out[m] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}
    return out


def main():
    print("Loading data and indices...")
    corpus, queries, qrels_test, _ = load_scifact()
    idx = load_indices()
    cache_file = INDEX_DIR / "candidates_test.npz"
    if not cache_file.exists():
        raise SystemExit(f"Run precompute_candidates.py first - missing {cache_file}")
    npz = np.load(cache_file, allow_pickle=True)
    cache = {
        "cand_idx":  npz["cand_idx"],
        "bm25_pool": npz["bm25_pool"],
        "dense_pool": npz["dense_pool"],
        "rerank":    npz["rerank"],
    }
    q_ids = [str(x) for x in npz["q_ids"]]

    print(f"Running {len(METHOD_NAMES)} methods on {len(q_ids)} queries...")
    t0 = time.time()
    per_query, retrieved, pT_store = run_all_methods(
        idx, queries, qrels_test, cache, q_ids
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(q_ids):.2f}s/query)")

    summary = summarize(per_query)
    paired_ndcg = paired_table(per_query, baseline="rerank", metric="ndcg@10")
    paired_recall10 = paired_table(per_query, baseline="rerank", metric="recall@10")
    paired_recall20 = paired_table(per_query, baseline="rerank", metric="recall@20")

    out = {
        "n_queries": len(q_ids),
        "elapsed_sec": elapsed,
        "summary": summary,
        "paired_vs_rerank": {
            "ndcg@10": paired_ndcg,
            "recall@10": paired_recall10,
            "recall@20": paired_recall20,
        },
        "per_query": {m: dict(d) for m, d in per_query.items()},
    }
    with open(RESULTS_DIR / "stage1.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else None)
    print(f"Saved {RESULTS_DIR / 'stage1.json'}")

    # save retrieved + p_T for stability analysis
    np.savez(
        RESULTS_DIR / "stage1_dists.npz",
        **{f"pT_{m}_{qid}": p for m, store in pT_store.items() for qid, p in store.items()}
    )
    with open(RESULTS_DIR / "retrieved.json", "w") as f:
        json.dump(retrieved, f, indent=2)

    print("\n=== Headline metrics (mean, 95% CI) ===")
    print(f"{'method':<22s}  {'nDCG@10':>20s}  {'Recall@10':>20s}  {'Recall@20':>20s}  {'Div@10':>10s}")
    for m in METHOD_NAMES:
        s = summary[m]
        ndcg = s["ndcg@10"]; r10 = s["recall@10"]; r20 = s["recall@20"]; div = s["diversity@10"]
        print(f"{m:<22s}  {ndcg['mean']:.3f} [{ndcg['ci_lo']:.3f},{ndcg['ci_hi']:.3f}]  "
              f"{r10['mean']:.3f} [{r10['ci_lo']:.3f},{r10['ci_hi']:.3f}]  "
              f"{r20['mean']:.3f} [{r20['ci_lo']:.3f},{r20['ci_hi']:.3f}]  "
              f"{div['mean']:.3f}")

    print("\n=== Paired diff vs 'rerank' on nDCG@10 (positive = our method beats rerank) ===")
    for m, d in paired_ndcg.items():
        sig = "*" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else " "
        print(f"  {m:<22s}  diff={d['diff']:+.4f}  [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  {sig}")


if __name__ == "__main__":
    main()
