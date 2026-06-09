"""Loading utilities and primitive retrievers (BM25, dense, hybrid)."""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_DIR = Path(__file__).resolve().parents[1] / "indices"


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class Indices:
    doc_ids: list[str]
    doc_id_to_idx: dict[str, int]
    doc_texts: dict[str, str]
    embeddings: np.ndarray  # (N, D), L2-normalized
    bm25: object  # rank_bm25.BM25Okapi
    bm25_tokenized: list[list[str]]


def load_indices() -> Indices:
    with open(INDEX_DIR / "doc_ids.json") as f:
        doc_ids = json.load(f)
    with open(INDEX_DIR / "doc_texts.json") as f:
        doc_texts = json.load(f)
    embeddings = np.load(INDEX_DIR / "embeddings.npy")
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    return Indices(
        doc_ids=doc_ids,
        doc_id_to_idx={d: i for i, d in enumerate(doc_ids)},
        doc_texts=doc_texts,
        embeddings=embeddings,
        bm25=bm25_data["bm25"],
        bm25_tokenized=bm25_data["tokenized"],
    )


def bm25_scores_for_query(idx: Indices, query: str) -> np.ndarray:
    """BM25 score for every doc in the index (length N)."""
    tokens = simple_tokenize(query)
    return idx.bm25.get_scores(tokens).astype(np.float32)


def dense_scores_for_query(idx: Indices, q_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity for every doc (q_emb is L2-normalized)."""
    return idx.embeddings @ q_emb  # (N,)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.argsort(-scores)
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


def reciprocal_rank_fusion(
    rank_lists: list[list[int]], k_const: int = 60
) -> dict[int, float]:
    """RRF score per doc index."""
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for r, doc_idx in enumerate(ranks):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k_const + r + 1)
    return fused


def hybrid_candidate_pool(
    idx: Indices,
    query: str,
    q_emb: np.ndarray,
    pool_size: int = 200,
    each_n: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (candidate_idx, bm25_scores_on_pool, dense_scores_on_pool).

    candidate_idx are indices into the corpus (length M = pool_size).
    """
    bm25_all = bm25_scores_for_query(idx, query)
    dense_all = dense_scores_for_query(idx, q_emb)
    bm25_top = topk_indices(bm25_all, each_n).tolist()
    dense_top = topk_indices(dense_all, each_n).tolist()
    fused = reciprocal_rank_fusion([bm25_top, dense_top])
    items = sorted(fused.items(), key=lambda kv: -kv[1])[:pool_size]
    cand = np.array([i for i, _ in items], dtype=np.int64)
    return cand, bm25_all[cand], dense_all[cand]


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def normalize_zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sigma = float(x.std()) + 1e-9
    return (x - mu) / sigma


def cosine_matrix(Z: np.ndarray) -> np.ndarray:
    """Z is L2-normalized -> cosine sim is Z @ Z.T."""
    return Z @ Z.T


def cost_matrix_cosine(Z: np.ndarray) -> np.ndarray:
    """C_ij = (1 - cos(z_i, z_j))^2, in [0, 4]."""
    sim = np.clip(Z @ Z.T, -1.0, 1.0)
    return (1.0 - sim) ** 2


def redundancy_kernel(Z: np.ndarray) -> np.ndarray:
    """K_ij = max(0, cos(z_i, z_j))."""
    sim = np.clip(Z @ Z.T, -1.0, 1.0)
    return np.maximum(sim, 0.0).astype(np.float32)


def softmax_np(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / max(tau, 1e-9)
    e = np.exp(z)
    return e / e.sum()
