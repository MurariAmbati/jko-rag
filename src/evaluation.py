"""Evaluation metrics: nDCG@k, Recall@k, MRR@k, plus diversity / entropy."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def dcg_at_k(rels: list[int], k: int) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    rels = [qrels.get(d, 0) for d in retrieved_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    total_rel = sum(1 for v in qrels.values() if v > 0)
    if total_rel == 0:
        return 0.0
    got = sum(1 for d in retrieved_ids[:k] if qrels.get(d, 0) > 0)
    return got / total_rel


def precision_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    got = sum(1 for d in retrieved_ids[:k] if qrels.get(d, 0) > 0)
    return got / k


def reciprocal_rank(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    for r, d in enumerate(retrieved_ids[:k]):
        if qrels.get(d, 0) > 0:
            return 1.0 / (r + 1)
    return 0.0


def semantic_diversity(retrieved_idx: list[int], embeddings: np.ndarray) -> float:
    """Mean pairwise (1 - cosine) over top-k retrieved (embeddings are L2-normalized)."""
    if len(retrieved_idx) < 2:
        return 0.0
    Z = embeddings[retrieved_idx]
    sim = Z @ Z.T
    n = len(retrieved_idx)
    total = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(sim[i, j])
            cnt += 1
    return total / max(1, cnt)


def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def bootstrap_ci(
    per_query_scores: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (mean, lo, hi) with (1-alpha)*100% bootstrap CI."""
    arr = np.asarray(per_query_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def paired_bootstrap_diff(
    a: list[float],
    b: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap on (a - b)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    diff = a_arr - b_arr
    return bootstrap_ci(diff.tolist(), n_boot=n_boot, alpha=alpha, seed=seed)


def summarize_runs(
    runs: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """runs[method][metric] -> list of per-query scores. Returns mean+CI."""
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for m, metrics in runs.items():
        out[m] = {}
        for metric, scores in metrics.items():
            out[m][metric] = bootstrap_ci(scores)
    return out
