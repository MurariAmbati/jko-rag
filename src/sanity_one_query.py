"""Sanity-check all methods on a single SciFact query end-to-end."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from download_data import load_scifact
from retrieval import load_indices
from methods import (
    make_candidates, method_bm25, method_dense, method_hybrid_rrf,
    method_rerank, method_mmr, method_wfe,
)
from jko import JKOConfig

INDEX_DIR = Path(__file__).resolve().parents[1] / "indices"


def main():
    corpus, queries, qrels_test, _ = load_scifact()
    idx = load_indices()
    q_emb_all = np.load(INDEX_DIR / "q_embeddings_test.npy")
    with open(INDEX_DIR / "q_ids_test.json") as f:
        test_qids = json.load(f)
    q_to_i = {q: i for i, q in enumerate(test_qids)}

    # pick a query with at least 1 known relevant doc
    qid = test_qids[0]
    query_text = queries[qid]
    q_emb = q_emb_all[q_to_i[qid]]
    relevant = set(qrels_test[qid].keys())
    print(f"Query [{qid}]: {query_text}")
    print(f"Relevant docs: {relevant}")

    print("\nBuilding candidates (pool=200, with reranker)...")
    cand = make_candidates(idx, query_text, q_emb, pool_size=200, do_rerank=True)
    # how many relevant docs are in the pool?
    pool_dids = {idx.doc_ids[i] for i in cand.cand_idx}
    print(f"Pool size: {len(cand.cand_idx)}, relevant in pool: {len(pool_dids & relevant)} / {len(relevant)}")

    def show(name, doc_indices):
        dids = [idx.doc_ids[i] for i in doc_indices]
        hits = [d for d in dids if d in relevant]
        print(f"  {name:18s} -> top5={dids[:5]} hits={hits}")

    show("bm25",    method_bm25(idx, query_text, q_emb, k=10))
    show("dense",   method_dense(idx, query_text, q_emb, k=10))
    show("hybrid",  method_hybrid_rrf(cand, k=10))
    show("rerank",  method_rerank(cand, k=10))
    show("mmr(0.5)", method_mmr(cand, idx, k=10, lambda_mmr=0.5))

    cfg_w = JKOConfig(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1, T=3, mode="wasserstein")
    cfg_k = JKOConfig(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1, T=3, mode="kl")
    cfg_n = JKOConfig(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1, T=3, mode="noproximal")
    r_w = method_wfe(cand, idx, cfg_w, k=10)
    r_k = method_wfe(cand, idx, cfg_k, k=10)
    r_n = method_wfe(cand, idx, cfg_n, k=10)
    show("jko-wfe",       r_w.topk)
    show("kl-prox",       r_k.topk)
    show("noprox",        r_n.topk)
    print(f"\nWFE p_T entropy: {-(r_w.p_T*np.log(r_w.p_T+1e-30)).sum():.3f}")
    print(f"KL  p_T entropy: {-(r_k.p_T*np.log(r_k.p_T+1e-30)).sum():.3f}")
    print(f"WFE p_T top-5 mass: {np.sort(r_w.p_T)[-5:].sum():.3f}")
    print(f"KL  p_T top-5 mass: {np.sort(r_k.p_T)[-5:].sum():.3f}")


if __name__ == "__main__":
    main()
