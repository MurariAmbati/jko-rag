"""Retrieval methods: BM25, dense, hybrid, reranker, MMR, KL-prox, JKO-RAG."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from retrieval import (
    Indices,
    bm25_scores_for_query,
    dense_scores_for_query,
    hybrid_candidate_pool,
    normalize_minmax,
    cost_matrix_cosine,
    redundancy_kernel,
    softmax_np,
)
from jko import JKOConfig, run_jko


# -----------------------------------------------------------------------------
# Cross-encoder reranker (lazy global to avoid reloading)
# -----------------------------------------------------------------------------
_RERANKER = None


def get_reranker():
    """ms-marco-MiniLM-L-6-v2 is a standard, small cross-encoder for IR."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
        )
    return _RERANKER


def rerank_scores(query: str, doc_texts: list[str], batch_size: int = 32) -> np.ndarray:
    pairs = [(query, t) for t in doc_texts]
    scores = get_reranker().predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)


# -----------------------------------------------------------------------------
# Shared candidate generation
# -----------------------------------------------------------------------------
@dataclass
class Candidates:
    cand_idx: np.ndarray            # (M,) into corpus
    bm25_scores: np.ndarray         # (M,)
    dense_scores: np.ndarray        # (M,)
    rerank_scores: np.ndarray | None = None  # (M,)


def make_candidates(
    idx: Indices,
    query_text: str,
    q_emb: np.ndarray,
    pool_size: int = 200,
    each_n: int = 500,
    do_rerank: bool = True,
) -> Candidates:
    cand, bm25_pool, dense_pool = hybrid_candidate_pool(
        idx, query_text, q_emb, pool_size=pool_size, each_n=each_n
    )
    rr = None
    if do_rerank:
        texts = [idx.doc_texts[idx.doc_ids[i]] for i in cand]
        rr = rerank_scores(query_text, texts)
    return Candidates(cand, bm25_pool, dense_pool, rr)


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
def method_bm25(idx: Indices, query: str, q_emb: np.ndarray, k: int = 10) -> list[int]:
    """Pure BM25 top-k over the full corpus."""
    s = bm25_scores_for_query(idx, query)
    return np.argsort(-s)[:k].tolist()


def method_dense(idx: Indices, query: str, q_emb: np.ndarray, k: int = 10) -> list[int]:
    s = dense_scores_for_query(idx, q_emb)
    return np.argsort(-s)[:k].tolist()


def method_hybrid_rrf(c: Candidates, k: int = 10) -> list[int]:
    """RRF of bm25 and dense ranks on the pool, top-k."""
    bm25_rank = np.argsort(-c.bm25_scores)
    dense_rank = np.argsort(-c.dense_scores)
    rrf = np.zeros(len(c.cand_idx), dtype=np.float64)
    K = 60
    for r, i in enumerate(bm25_rank):
        rrf[i] += 1.0 / (K + r + 1)
    for r, i in enumerate(dense_rank):
        rrf[i] += 1.0 / (K + r + 1)
    order = np.argsort(-rrf)[:k]
    return c.cand_idx[order].tolist()


def method_rerank(c: Candidates, k: int = 10) -> list[int]:
    """Hybrid+cross-encoder: top-k by reranker score."""
    assert c.rerank_scores is not None
    order = np.argsort(-c.rerank_scores)[:k]
    return c.cand_idx[order].tolist()


def method_mmr(
    c: Candidates,
    idx: Indices,
    k: int = 10,
    lambda_mmr: float = 0.5,
) -> list[int]:
    """Maximal Marginal Relevance over the candidate pool.

    Relevance = reranker score if available, else normalized hybrid.
    Diversity = max cosine to already-selected.
    """
    Z = idx.embeddings[c.cand_idx]
    if c.rerank_scores is not None:
        rel = normalize_minmax(c.rerank_scores)
    else:
        rel = 0.5 * normalize_minmax(c.bm25_scores) + 0.5 * normalize_minmax(c.dense_scores)
    M = len(c.cand_idx)
    selected: list[int] = []
    remaining = set(range(M))
    sim = Z @ Z.T
    while len(selected) < min(k, M):
        best_i, best_score = -1, -1e18
        for i in remaining:
            if not selected:
                score = rel[i]
            else:
                diversity = max(sim[i, j] for j in selected)
                score = lambda_mmr * rel[i] - (1 - lambda_mmr) * diversity
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.discard(best_i)
    return c.cand_idx[selected].tolist()


# -----------------------------------------------------------------------------
# Distributional methods (JKO-RAG + ablations)
# -----------------------------------------------------------------------------
@dataclass
class WFEResult:
    p_T: np.ndarray
    cand_idx: np.ndarray
    energy: np.ndarray
    C: np.ndarray
    K: np.ndarray
    topk: list[int]


def _make_energy(
    c: Candidates,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Build relevance, then energy = -relevance. Defaults: pure reranker."""
    r = np.zeros(len(c.cand_idx), dtype=np.float32)
    if alpha > 0:
        r = r + alpha * normalize_minmax(c.dense_scores)
    if beta > 0:
        r = r + beta * normalize_minmax(c.bm25_scores)
    if gamma > 0 and c.rerank_scores is not None:
        r = r + gamma * normalize_minmax(c.rerank_scores)
    return -r


def _make_p0(c: Candidates, tau0: float = 0.1, use_rerank: bool = True) -> np.ndarray:
    if use_rerank and c.rerank_scores is not None:
        a = normalize_minmax(c.rerank_scores)
    else:
        a = 0.5 * normalize_minmax(c.dense_scores) + 0.5 * normalize_minmax(c.bm25_scores)
    return softmax_np(a, tau=tau0)


def method_wfe(
    c: Candidates,
    idx: Indices,
    cfg: JKOConfig,
    k: int = 10,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    tau0: float = 0.1,
) -> WFEResult:
    """Generic Wasserstein/KL/noproximal free-energy retriever.

    `cfg.mode` controls which proximal term is used.
    """
    Z = idx.embeddings[c.cand_idx]
    C = cost_matrix_cosine(Z)
    K = redundancy_kernel(Z)
    energy = _make_energy(c, alpha, beta, gamma)
    p0 = _make_p0(c, tau0=tau0, use_rerank=(gamma > 0 and c.rerank_scores is not None))

    p_T, _ = run_jko(p0, energy, C, K, cfg)
    order = np.argsort(-p_T)[:k]
    topk = c.cand_idx[order].tolist()
    return WFEResult(p_T=p_T, cand_idx=c.cand_idx, energy=energy, C=C, K=K, topk=topk)
