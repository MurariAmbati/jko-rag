"""Stage 3 robustness: perturb queries and measure retrieval distribution shift.

For a subset of test queries, build paraphrase-like perturbations via simple
lexical edits (drop a stopword, append a hedge) and re-run each method, then
measure W_C( p_T(q), p_T(q') ) between the original and perturbed retrieval
distributions. Lower instability = more stable.

This is meant to test whether Wasserstein-proximal retrieval is more robust
to query perturbation than KL-proximal or one-shot top-k.

Note: this uses lexical perturbations, not an LLM paraphraser - cheap and
deterministic, but a real evaluation would use an LLM.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from download_data import load_scifact
from retrieval import (
    load_indices, hybrid_candidate_pool, cost_matrix_cosine, redundancy_kernel,
    softmax_np, normalize_minmax,
)
from methods import Candidates, rerank_scores
from jko import JKOConfig, run_jko, log_sinkhorn_loss
from evaluation import ndcg_at_k, recall_at_k

INDEX_DIR = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

STOPWORDS = {"the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by", "from"}
HEDGES = [
    " in some cases.", " under certain conditions.", " according to studies.",
]


def perturb_drop_stopword(q: str, seed: int = 0) -> str:
    toks = q.split()
    rng = np.random.default_rng(seed)
    idxs = [i for i, t in enumerate(toks) if t.lower().strip(".,?!") in STOPWORDS]
    if not idxs:
        return q
    drop = int(rng.choice(idxs))
    out = " ".join(toks[:drop] + toks[drop + 1:])
    return out


def perturb_append_hedge(q: str, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    q = q.rstrip(".")
    return q + HEDGES[int(rng.integers(0, len(HEDGES)))]


def perturb_lower_punct(q: str) -> str:
    return re.sub(r"[.\?!]", "", q).lower()


PERTURBATIONS = [
    ("drop_stop", perturb_drop_stopword),
    ("hedge",     perturb_append_hedge),
    ("lower_nop", lambda q, seed=0: perturb_lower_punct(q)),
]


def encode_one(model, text: str) -> np.ndarray:
    return model.encode([text], normalize_embeddings=True)[0].astype(np.float32)


def build_candidates_for_query(idx, query_text: str, q_emb: np.ndarray, pool_size: int = 200):
    cand, bm25_pool, dense_pool = hybrid_candidate_pool(
        idx, query_text, q_emb, pool_size=pool_size, each_n=500
    )
    texts = [idx.doc_texts[idx.doc_ids[i]] for i in cand]
    rr = rerank_scores(query_text, texts, batch_size=64)
    return Candidates(cand_idx=cand, bm25_scores=bm25_pool, dense_scores=dense_pool, rerank_scores=rr)


def jko_distribution(c: Candidates, idx, mode: str, alpha: float = 0, beta: float = 0, gamma: float = 1.0) -> np.ndarray:
    Z = idx.embeddings[c.cand_idx]
    C = cost_matrix_cosine(Z)
    K = redundancy_kernel(Z)
    r = np.zeros(len(c.cand_idx), dtype=np.float32)
    if alpha > 0:
        r = r + alpha * normalize_minmax(c.dense_scores)
    if beta > 0:
        r = r + beta * normalize_minmax(c.bm25_scores)
    if gamma > 0 and c.rerank_scores is not None:
        r = r + gamma * normalize_minmax(c.rerank_scores)
    energy = -r
    p0 = softmax_np(-energy, tau=0.1)
    cfg = JKOConfig(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1, T=3, inner_steps=40, mode=mode)
    p_T, _ = run_jko(p0, energy, C, K, cfg)
    return p_T, p0, c.cand_idx, Z


def topk_dist(c: Candidates, mode: str = "rerank", k: int = 10) -> np.ndarray:
    """One-shot top-k distribution: spike on the top-k items with uniform mass.
    Used to measure instability of one-shot methods."""
    p = np.zeros(len(c.cand_idx), dtype=np.float32)
    if mode == "rerank":
        order = np.argsort(-c.rerank_scores)
    elif mode == "dense":
        order = np.argsort(-c.dense_scores)
    else:
        raise ValueError(mode)
    p[order[:k]] = 1.0 / k
    return p


def w_distance_on_full_pool(
    pa: np.ndarray, cand_a: np.ndarray,
    pb: np.ndarray, cand_b: np.ndarray,
    embeddings: np.ndarray,
    eps: float = 0.1,
) -> float:
    """Entropic Wasserstein distance between two distributions over (possibly
    different) candidate pools. Lifts both to the union pool with zero mass on
    items absent from each, then computes a single Sinkhorn distance."""
    union = sorted(set(cand_a.tolist()) | set(cand_b.tolist()))
    idx_of = {d: i for i, d in enumerate(union)}
    n = len(union)
    pa_u = np.zeros(n, dtype=np.float32)
    pb_u = np.zeros(n, dtype=np.float32)
    for i, d in enumerate(cand_a):
        pa_u[idx_of[int(d)]] += pa[i]
    for i, d in enumerate(cand_b):
        pb_u[idx_of[int(d)]] += pb[i]
    # Numerical: add tiny floor so log is defined for sinkhorn
    pa_u = (pa_u + 1e-8); pa_u /= pa_u.sum()
    pb_u = (pb_u + 1e-8); pb_u /= pb_u.sum()
    Z = embeddings[np.array(union, dtype=np.int64)]
    sim = np.clip(Z @ Z.T, -1.0, 1.0)
    C = (1.0 - sim) ** 2
    pa_t = torch.tensor(pa_u, dtype=torch.float32)
    pb_t = torch.tensor(pb_u, dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)
    with torch.no_grad():
        w = log_sinkhorn_loss(torch.log(pa_t), torch.log(pb_t), Ct, eps=eps, n_iter=80)
    return float(w)


def main(n_queries: int = 80, seed: int = 0):
    print("Loading...")
    corpus, queries, qrels_test, _ = load_scifact()
    idx = load_indices()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with open(INDEX_DIR / "q_ids_test.json") as f:
        all_qids = json.load(f)

    # Reuse precomputed candidates / reranker scores for the ORIGINAL queries
    cache = np.load(INDEX_DIR / "candidates_test.npz", allow_pickle=True)
    cache_qids = [str(x) for x in cache["q_ids"]]
    qid_to_cache_idx = {q: i for i, q in enumerate(cache_qids)}

    rng = np.random.default_rng(seed)
    qids = list(rng.choice(all_qids, size=min(n_queries, len(all_qids)), replace=False))
    print(f"Running stability over {len(qids)} queries x {len(PERTURBATIONS)} perturbations")

    out: dict = {"queries": [], "summary": {}}
    method_modes = {
        "jko_rerank": ("wasserstein", 0.0, 0.0, 1.0),
        "kl_rerank":  ("kl",          0.0, 0.0, 1.0),
        "noprox":     ("noproximal",  0.0, 0.0, 1.0),
    }
    # Track instability per method per perturbation
    instab: dict[tuple[str, str], list[float]] = {}

    # also include one-shot baselines for comparison
    one_shot_methods = ["rerank_topk", "dense_topk"]

    for qid in tqdm(qids, desc="stability"):
        q_text = queries[qid]
        # Reuse precomputed candidates for the original query
        if qid in qid_to_cache_idx:
            ci = qid_to_cache_idx[qid]
            c_q = Candidates(
                cand_idx=cache["cand_idx"][ci],
                bm25_scores=cache["bm25_pool"][ci],
                dense_scores=cache["dense_pool"][ci],
                rerank_scores=cache["rerank"][ci],
            )
        else:
            q_emb = encode_one(model, q_text)
            c_q = build_candidates_for_query(idx, q_text, q_emb)

        # compute base distributions
        dists_base = {}
        cands_base = {}
        for mname, (mode, a, b, g) in method_modes.items():
            p_T, _, cand_q, _ = jko_distribution(c_q, idx, mode, a, b, g)
            dists_base[mname] = p_T
            cands_base[mname] = cand_q
        dists_base["rerank_topk"] = topk_dist(c_q, mode="rerank", k=10)
        dists_base["dense_topk"] = topk_dist(c_q, mode="dense", k=10)
        cands_base["rerank_topk"] = c_q.cand_idx
        cands_base["dense_topk"] = c_q.cand_idx

        for p_name, fn in PERTURBATIONS:
            q_perturbed = fn(q_text, seed=seed)
            if q_perturbed.strip() == q_text.strip():
                continue
            q_emb_p = encode_one(model, q_perturbed)
            c_p = build_candidates_for_query(idx, q_perturbed, q_emb_p)

            for mname, (mode, a, b, g) in method_modes.items():
                p_T_p, _, cand_p, _ = jko_distribution(c_p, idx, mode, a, b, g)
                w = w_distance_on_full_pool(
                    dists_base[mname], cands_base[mname],
                    p_T_p, cand_p,
                    idx.embeddings,
                )
                instab.setdefault((mname, p_name), []).append(w)
            for mname in one_shot_methods:
                if mname == "rerank_topk":
                    p_p = topk_dist(c_p, mode="rerank", k=10)
                else:
                    p_p = topk_dist(c_p, mode="dense", k=10)
                w = w_distance_on_full_pool(
                    dists_base[mname], cands_base[mname],
                    p_p, c_p.cand_idx, idx.embeddings,
                )
                instab.setdefault((mname, p_name), []).append(w)

    # Summarize
    summary: dict = {}
    for (mname, p_name), vals in instab.items():
        arr = np.asarray(vals)
        summary.setdefault(mname, {})[p_name] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "n":    int(arr.size),
            "p25":  float(np.quantile(arr, 0.25)),
            "p50":  float(np.quantile(arr, 0.50)),
            "p75":  float(np.quantile(arr, 0.75)),
        }

    out["summary"] = summary
    out["per_method_mean_over_perturbations"] = {
        m: float(np.mean([v["mean"] for v in pdict.values()])) for m, pdict in summary.items()
    }

    (RESULTS_DIR / "stability.json").write_text(json.dumps(out, indent=2))
    print("\n=== Mean retrieval instability (lower = more stable) ===")
    for m, score in sorted(out["per_method_mean_over_perturbations"].items(), key=lambda x: x[1]):
        print(f"  {m:<14s}  W_C(p,p') = {score:.4f}")


if __name__ == "__main__":
    main(n_queries=60)
