"""Multi-dataset version of stability experiment.

Same protocol as run_stability.py but parameterized by dataset name.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset
from retrieval import (
    Indices, cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax,
)
from methods import Candidates, rerank_scores
from jko import JKOConfig, run_jko, log_sinkhorn_loss

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TOKEN_RE = re.compile(r"\w+")

STOPWORDS = {"the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by", "from"}
HEDGES = [" in some cases.", " under certain conditions.", " according to studies."]


def tokenize(t): return TOKEN_RE.findall(t.lower())


def perturb_drop_stopword(q, seed=0):
    toks = q.split()
    rng = np.random.default_rng(seed)
    idxs = [i for i, t in enumerate(toks) if t.lower().strip(".,?!") in STOPWORDS]
    if not idxs: return q
    drop = int(rng.choice(idxs))
    return " ".join(toks[:drop] + toks[drop+1:])


def perturb_append_hedge(q, seed=0):
    rng = np.random.default_rng(seed)
    q = q.rstrip(".")
    return q + HEDGES[int(rng.integers(0, len(HEDGES)))]


def perturb_lower_punct(q):
    return re.sub(r"[.\?!]", "", q).lower()


PERTURBATIONS = [
    ("drop_stop", perturb_drop_stopword),
    ("hedge",     perturb_append_hedge),
    ("lower_nop", lambda q, seed=0: perturb_lower_punct(q)),
]


def index_dir(name):
    sub = INDEX_ROOT / name
    return sub if (sub / "doc_ids.json").exists() else INDEX_ROOT


def load_index(name):
    base = index_dir(name)
    with open(base / "doc_ids.json") as f: doc_ids = json.load(f)
    with open(base / "doc_texts.json") as f: doc_texts = json.load(f)
    embeddings = np.load(base / "embeddings.npy")
    with open(base / "bm25.pkl", "rb") as f: bm25_data = pickle.load(f)
    return Indices(
        doc_ids=doc_ids, doc_id_to_idx={d: i for i, d in enumerate(doc_ids)},
        doc_texts=doc_texts, embeddings=embeddings,
        bm25=bm25_data["bm25"], bm25_tokenized=bm25_data["tokenized"],
    )


def hybrid_pool(idx, q_text, q_emb, pool_size=200, each_n=500):
    bm25_all = idx.bm25.get_scores(tokenize(q_text)).astype(np.float32)
    dense_all = (idx.embeddings @ q_emb).astype(np.float32)
    en = min(each_n, len(bm25_all) - 1)
    bm25_top = np.argpartition(-bm25_all, en)[:en]; bm25_top = bm25_top[np.argsort(-bm25_all[bm25_top])]
    dense_top = np.argpartition(-dense_all, en)[:en]; dense_top = dense_top[np.argsort(-dense_all[dense_top])]
    fused = {}; K = 60
    for r, di in enumerate(bm25_top.tolist()): fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    for r, di in enumerate(dense_top.tolist()): fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    items = sorted(fused.items(), key=lambda kv: -kv[1])[:pool_size]
    cand = np.array([i for i, _ in items], dtype=np.int64)
    return cand, bm25_all[cand], dense_all[cand]


def build_candidates(idx, q_text, q_emb, pool_size=200):
    cand, bm25_pool, dense_pool = hybrid_pool(idx, q_text, q_emb, pool_size=pool_size)
    texts = [idx.doc_texts[idx.doc_ids[int(i)]] for i in cand]
    rr = rerank_scores(q_text, texts, batch_size=64)
    return Candidates(cand_idx=cand, bm25_scores=bm25_pool, dense_scores=dense_pool, rerank_scores=rr)


def jko_distribution(c, idx, mode, alpha=0, gamma=1.0):
    Z = idx.embeddings[c.cand_idx]
    C = cost_matrix_cosine(Z); K = redundancy_kernel(Z)
    r = alpha * normalize_minmax(c.dense_scores) + gamma * normalize_minmax(c.rerank_scores)
    energy = -r
    p0 = softmax_np(-energy, tau=0.1)
    cfg = JKOConfig(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1, T=3, inner_steps=25, mode=mode)
    p_T, _ = run_jko(p0, energy, C, K, cfg)
    return p_T, c.cand_idx


def topk_dist(c, mode="rerank", k=10):
    p = np.zeros(len(c.cand_idx), dtype=np.float32)
    s = c.rerank_scores if mode == "rerank" else c.dense_scores
    p[np.argsort(-s)[:k]] = 1.0 / k
    return p


def w_distance_on_full_pool(pa, cand_a, pb, cand_b, embeddings, eps=0.1):
    union = sorted(set(cand_a.tolist()) | set(cand_b.tolist()))
    idx_of = {d: i for i, d in enumerate(union)}
    n = len(union)
    pa_u = np.zeros(n, dtype=np.float32); pb_u = np.zeros(n, dtype=np.float32)
    for i, d in enumerate(cand_a): pa_u[idx_of[int(d)]] += pa[i]
    for i, d in enumerate(cand_b): pb_u[idx_of[int(d)]] += pb[i]
    pa_u = (pa_u + 1e-8); pa_u /= pa_u.sum()
    pb_u = (pb_u + 1e-8); pb_u /= pb_u.sum()
    Z = embeddings[np.array(union, dtype=np.int64)]
    C = (1.0 - np.clip(Z @ Z.T, -1.0, 1.0)) ** 2
    pa_t = torch.tensor(pa_u, dtype=torch.float32)
    pb_t = torch.tensor(pb_u, dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)
    with torch.no_grad():
        w = log_sinkhorn_loss(torch.log(pa_t), torch.log(pb_t), Ct, eps=eps, n_iter=80)
    return float(w)


def encode_one(model, text):
    return model.encode([text], normalize_embeddings=True)[0].astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-queries", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    print(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset)
    idx = load_index(args.dataset)
    queries = ds.queries
    qrels_test = ds.qrels.get("test", {})

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load precomputed candidates if available
    sub_cache = index_dir(args.dataset) / "candidates_test.npz"
    cache_qids = []
    cache = None
    if sub_cache.exists():
        cache = np.load(sub_cache, allow_pickle=True)
        cache_qids = [str(x) for x in cache["q_ids"]]
    qid_to_ci = {q: i for i, q in enumerate(cache_qids)}

    all_qids = list(qrels_test.keys())
    rng = np.random.default_rng(args.seed)
    qids = list(rng.choice(all_qids, size=min(args.n_queries, len(all_qids)), replace=False))
    print(f"Running stability over {len(qids)} queries x {len(PERTURBATIONS)} perturbations")

    method_modes = {
        "jko_rerank": ("wasserstein", 0.0, 1.0),
        "kl_rerank":  ("kl",          0.0, 1.0),
        "noprox":     ("noproximal",  0.0, 1.0),
    }
    one_shot_methods = ["rerank_topk", "dense_topk"]
    instab = {}

    for qid in tqdm(qids, desc=f"{args.dataset} stability"):
        q_text = queries[qid]
        if qid in qid_to_ci and cache is not None:
            ci = qid_to_ci[qid]
            c_q = Candidates(
                cand_idx=cache["cand_idx"][ci], bm25_scores=cache["bm25_pool"][ci],
                dense_scores=cache["dense_pool"][ci], rerank_scores=cache["rerank"][ci],
            )
        else:
            q_emb = encode_one(model, q_text)
            c_q = build_candidates(idx, q_text, q_emb)

        dists_base = {}; cands_base = {}
        for mname, (mode, a, g) in method_modes.items():
            p_T, cand_q = jko_distribution(c_q, idx, mode, a, g)
            dists_base[mname] = p_T; cands_base[mname] = cand_q
        dists_base["rerank_topk"] = topk_dist(c_q, mode="rerank", k=10)
        dists_base["dense_topk"] = topk_dist(c_q, mode="dense", k=10)
        cands_base["rerank_topk"] = c_q.cand_idx
        cands_base["dense_topk"] = c_q.cand_idx

        for p_name, fn in PERTURBATIONS:
            q_perturbed = fn(q_text, seed=args.seed)
            if q_perturbed.strip() == q_text.strip(): continue
            q_emb_p = encode_one(model, q_perturbed)
            c_p = build_candidates(idx, q_perturbed, q_emb_p)

            for mname, (mode, a, g) in method_modes.items():
                p_T_p, cand_p = jko_distribution(c_p, idx, mode, a, g)
                w = w_distance_on_full_pool(
                    dists_base[mname], cands_base[mname], p_T_p, cand_p, idx.embeddings)
                instab.setdefault((mname, p_name), []).append(w)
            for mname in one_shot_methods:
                p_p = topk_dist(c_p, mode="rerank" if mname == "rerank_topk" else "dense", k=10)
                w = w_distance_on_full_pool(
                    dists_base[mname], cands_base[mname], p_p, c_p.cand_idx, idx.embeddings)
                instab.setdefault((mname, p_name), []).append(w)

    summary = {}
    for (mname, p_name), vals in instab.items():
        arr = np.asarray(vals)
        summary.setdefault(mname, {})[p_name] = {
            "mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size),
            "p25": float(np.quantile(arr, 0.25)), "p50": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
        }

    out = {
        "dataset": args.dataset,
        "n_queries": len(qids),
        "summary": summary,
        "per_method_mean_over_perturbations": {
            m: float(np.mean([v["mean"] for v in pd.values()])) for m, pd in summary.items()
        },
    }
    name = f"stability_{args.dataset}{args.out_suffix}.json"
    (RESULTS_DIR / name).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {RESULTS_DIR / name}")
    print(f"\n=== {args.dataset} retrieval instability (lower = more stable) ===")
    for m, score in sorted(out["per_method_mean_over_perturbations"].items(), key=lambda x: x[1]):
        print(f"  {m:<14s}  W_C(p,p') = {score:.4f}")


if __name__ == "__main__":
    main()
