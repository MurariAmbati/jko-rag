"""Benchmark MR-JKO vs vanilla JKO on (a) synthetic data at varying M and
(b) the SciFact test set at M=200.

We measure:
  - Wall time per query
  - Retrieval quality (nDCG@10, recall@10) where ground truth exists
  - Mass concentration on the relevant cluster (synthetic only)

Output: results/mr_jko_bench.json with timing + quality numbers.
"""
from __future__ import annotations

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
from retrieval import Indices, cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax
from jko import JKOConfig, run_jko
from hierarchical_jko import mr_jko
from evaluation import ndcg_at_k, recall_at_k, semantic_diversity, bootstrap_ci

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def index_dir(name):
    sub = INDEX_ROOT / name
    return sub if (sub / "doc_ids.json").exists() else INDEX_ROOT


def load_index(name) -> Indices:
    base = index_dir(name)
    with open(base / "doc_ids.json") as f: doc_ids = json.load(f)
    with open(base / "doc_texts.json") as f: doc_texts = json.load(f)
    embeddings = np.load(base / "embeddings.npy")
    with open(base / "bm25.pkl", "rb") as f: bm25_data = pickle.load(f)
    return Indices(doc_ids=doc_ids, doc_id_to_idx={d: i for i, d in enumerate(doc_ids)},
                   doc_texts=doc_texts, embeddings=embeddings,
                   bm25=bm25_data["bm25"], bm25_tokenized=bm25_data["tokenized"])


def load_cache(name, split):
    npz = np.load(index_dir(name) / f"candidates_{split}.npz", allow_pickle=True)
    return {
        "cand_idx": npz["cand_idx"], "bm25_pool": npz["bm25_pool"],
        "dense_pool": npz["dense_pool"], "rerank": npz["rerank"],
        "q_ids": [str(x) for x in npz["q_ids"]],
    }


# ----------------------------------------------------------------------
# (a) Synthetic scaling benchmark
# ----------------------------------------------------------------------

def synthetic_bench(M_values=(100, 200, 500, 1000), d=384, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for M in M_values:
        # build M chunks in 20 clusters
        K = 20
        centers = rng.normal(size=(K, d))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        Z = []
        rel = []
        per_cluster = M // K
        for ci, c in enumerate(centers):
            for _ in range(per_cluster):
                z = c + 0.05 * rng.normal(size=d)
                z /= np.linalg.norm(z)
                Z.append(z)
                # cluster 0 is the "gold" cluster
                rel.append(0.9 + 0.05 * rng.normal() if ci == 0 else 0.3 + 0.1 * rng.normal())
        Z = np.stack(Z).astype(np.float32)
        rel = np.array(rel, dtype=np.float32).clip(0, 1)

        # Vanilla JKO
        C = cost_matrix_cosine(Z).astype(np.float32)
        Kr = redundancy_kernel(Z).astype(np.float32)
        p0 = softmax_np(rel, tau=0.2)
        cfg = JKOConfig(h=2.0, lam=0.1, rho=0.05, sinkhorn_eps=0.2, T=3, inner_steps=20)
        t0 = time.time()
        p_vanilla, _ = run_jko(p0, -rel, C, Kr, cfg)
        t_vanilla = time.time() - t0

        # MR-JKO
        t0 = time.time()
        p_mr, pool = mr_jko(Z, rel, G=max(10, M // 25), G_keep=4)
        t_mr = time.time() - t0

        # Mass on cluster 0 (indices 0..per_cluster-1)
        mass_van = float(p_vanilla[:per_cluster].sum())
        mass_mr  = float(p_mr[:per_cluster].sum())

        # Top-10 overlap with cluster 0
        gold = set(range(per_cluster))
        top10_van = set(np.argsort(-p_vanilla)[:10].tolist())
        top10_mr  = set(np.argsort(-p_mr)[:10].tolist())
        rec_van = len(top10_van & gold) / len(gold) if gold else 0
        rec_mr  = len(top10_mr  & gold) / len(gold) if gold else 0

        out.append({
            "M": M, "n_clusters": K, "per_cluster": per_cluster,
            "vanilla_sec": t_vanilla, "mr_sec": t_mr,
            "speedup_x": t_vanilla / max(t_mr, 1e-6),
            "vanilla_mass_on_gold_cluster": mass_van,
            "mr_mass_on_gold_cluster": mass_mr,
            "vanilla_top10_in_gold_frac": float(rec_van),
            "mr_top10_in_gold_frac": float(rec_mr),
            "mr_refined_pool_size": int(len(pool)),
        })
        print(f"  M={M:>5d}: vanilla {t_vanilla*1000:.0f}ms (mass={mass_van:.2f}, rec={rec_van:.2f})  "
              f"MR {t_mr*1000:.0f}ms (mass={mass_mr:.2f}, rec={rec_mr:.2f})  "
              f"speedup={t_vanilla/max(t_mr,1e-6):.2f}x")
    return out


# ----------------------------------------------------------------------
# (b) SciFact test benchmark
# ----------------------------------------------------------------------

def scifact_bench():
    ds = load_dataset("scifact")
    qrels = ds.qrels["test"]
    idx = load_index("scifact")
    cache = load_cache("scifact", "test")
    q_ids = cache["q_ids"]

    cfg_fine = {"h": 2.0, "lam": 0.1, "rho": 0.05, "sinkhorn_eps": 0.2,
                "T": 3, "inner_steps": 20, "tau0": 1.0}
    cfg_coarse = {"h": 1.0, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
                   "T": 2, "inner_steps": 15, "tau0": 0.2}

    # Methods: vanilla, plain MR (kmeans), SAM with beta in {0.5, 1.0, 2.0, 4.0}
    method_cfgs = [
        ("vanilla", None, None),
        ("mr_kmeans", "kmeans", None),
        ("sam_b05", "sam", 0.5),
        ("sam_b10", "sam", 1.0),
        ("sam_b20", "sam", 2.0),
        ("sam_b40", "sam", 4.0),
    ]
    per_query = {m: defaultdict(list) for m, _, _ in method_cfgs}
    times = {m: 0.0 for m, _, _ in method_cfgs}
    n_eval = 0
    for qi, qid in enumerate(tqdm(q_ids, desc="mr_bench/scifact")):
        if qid not in qrels: continue
        c = Candidates(cand_idx=cache["cand_idx"][qi], bm25_scores=cache["bm25_pool"][qi],
                       dense_scores=cache["dense_pool"][qi], rerank_scores=cache["rerank"][qi])
        Z = idx.embeddings[c.cand_idx]
        rel = normalize_minmax(0.4 * normalize_minmax(c.dense_scores)
                               + 0.6 * normalize_minmax(c.rerank_scores))
        C = cost_matrix_cosine(Z).astype(np.float32)
        Kr = redundancy_kernel(Z).astype(np.float32)
        energy = -rel
        p0 = softmax_np(-energy, tau=cfg_fine["tau0"])
        jcfg = JKOConfig(h=cfg_fine["h"], lam=cfg_fine["lam"], rho=cfg_fine["rho"],
                          sinkhorn_eps=cfg_fine["sinkhorn_eps"], T=cfg_fine["T"],
                          inner_steps=cfg_fine["inner_steps"], mode="wasserstein")

        for name, clustering, sam_beta in method_cfgs:
            t0 = time.time()
            if name == "vanilla":
                p, _ = run_jko(p0, energy, C, Kr, jcfg)
            else:
                p, _ = mr_jko(Z, rel, G=20, G_keep=4,
                              coarse_cfg=cfg_coarse, fine_cfg=cfg_fine,
                              clustering=clustering, sam_beta=sam_beta or 1.0)
            times[name] += time.time() - t0
            top = c.cand_idx[np.argsort(-p)[:20]].tolist()
            dids = [idx.doc_ids[i] for i in top]
            rels = qrels[qid]
            per_query[name]["ndcg@10"].append(ndcg_at_k(dids, rels, 10))
            per_query[name]["recall@10"].append(recall_at_k(dids, rels, 10))
            per_query[name]["diversity@10"].append(semantic_diversity(top[:10], idx.embeddings))
        n_eval += 1

    summary = {}
    for name, pq in per_query.items():
        summary[name] = {m: dict(zip(("mean","ci_lo","ci_hi"), bootstrap_ci(pq[m])))
                         for m in pq}
        summary[name]["sec_per_q"] = times[name] / max(n_eval, 1)
        summary[name]["total_sec"] = times[name]
        summary[name]["speedup_x"] = times["vanilla"] / max(times[name], 1e-6)
    return {
        "n_queries": n_eval,
        "summary": summary,
        "per_query": {m: dict(d) for m, d in per_query.items()},
    }


def main():
    print("=== MR-JKO BENCHMARK ===")
    print("\n(a) Synthetic scaling:")
    syn = synthetic_bench()

    print("\n(b) SciFact test (M=200, including SAM-JKO with varying beta):")
    sci = scifact_bench()
    print(f"  {'method':<14s}  nDCG@10  R@10   ms/q  speedup")
    for m, info in sci["summary"].items():
        print(f"  {m:<14s}  {info['ndcg@10']['mean']:.3f}    {info['recall@10']['mean']:.3f}  "
              f"{info['sec_per_q']*1000:.0f}    {info['speedup_x']:.2f}x")

    out = {"synthetic": syn, "scifact_test": sci}
    (RESULTS_DIR / "mr_jko_bench.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {RESULTS_DIR / 'mr_jko_bench.json'}")


if __name__ == "__main__":
    main()
