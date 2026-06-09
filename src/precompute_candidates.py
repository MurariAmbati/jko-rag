"""Precompute candidate pools + reranker scores for all test queries.

Why: cross-encoder is expensive. Compute once, reuse across all methods/ablations.

Saved to indices/candidates_test.npz:
- cand_idx_arr   : (Q, M) int64 candidate indices
- bm25_pool_arr  : (Q, M) float32 BM25 score on pool
- dense_pool_arr : (Q, M) float32 dense (cosine) score on pool
- rerank_arr     : (Q, M) float32 cross-encoder score on pool
- q_ids          : (Q,) list of query IDs
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from download_data import load_scifact
from retrieval import load_indices, hybrid_candidate_pool
from methods import rerank_scores

INDEX_DIR = Path(__file__).resolve().parents[1] / "indices"
POOL_SIZE = 200


def main():
    corpus, queries, qrels_test, _ = load_scifact()
    idx = load_indices()
    q_emb_all = np.load(INDEX_DIR / "q_embeddings_test.npy")
    with open(INDEX_DIR / "q_ids_test.json") as f:
        q_ids = json.load(f)
    Q = len(q_ids)

    cand_arr = np.zeros((Q, POOL_SIZE), dtype=np.int64)
    bm25_arr = np.zeros((Q, POOL_SIZE), dtype=np.float32)
    dense_arr = np.zeros((Q, POOL_SIZE), dtype=np.float32)
    rerank_arr = np.zeros((Q, POOL_SIZE), dtype=np.float32)

    print(f"Building candidate pools (M={POOL_SIZE}) for {Q} queries...")
    for i, qid in enumerate(tqdm(q_ids, desc="pools")):
        cand, bm25s, dense_s = hybrid_candidate_pool(
            idx, queries[qid], q_emb_all[i], pool_size=POOL_SIZE, each_n=500
        )
        cand_arr[i] = cand
        bm25_arr[i] = bm25s
        dense_arr[i] = dense_s

    # Batch reranker across all (query, doc) pairs for max throughput.
    print(f"Scoring {Q * POOL_SIZE:,} (query, doc) pairs with cross-encoder...")
    t0 = time.time()
    # Process query-by-query but with large batches
    for i, qid in enumerate(tqdm(q_ids, desc="rerank")):
        texts = [idx.doc_texts[idx.doc_ids[j]] for j in cand_arr[i]]
        rerank_arr[i] = rerank_scores(queries[qid], texts, batch_size=64)
    print(f"Reranking done in {time.time() - t0:.1f}s")

    np.savez(
        INDEX_DIR / "candidates_test.npz",
        cand_idx=cand_arr,
        bm25_pool=bm25_arr,
        dense_pool=dense_arr,
        rerank=rerank_arr,
        q_ids=np.asarray(q_ids),
    )
    print(f"Saved {INDEX_DIR / 'candidates_test.npz'}")


if __name__ == "__main__":
    main()
