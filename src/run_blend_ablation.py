"""Add the missing W-vs-KL ablation with blended energy.

The original Stage 1 only ran KL with rerank-only energy. With that energy
both W and KL collapse to the reranker's argmax (254/300 identical). The
proper decisive test is: same BLENDED energy, only the proximal differs.

Adds these methods to the comparison and saves to results/blend_ablation.json:
  - kl_blend            (alpha=0.4, gamma=0.6, KL-prox)
  - kl_blend_dense      (alpha=0.7, gamma=0.3, KL-prox)
  - noprox_blend        (alpha=0.4, gamma=0.6, no prox)
  - noprox_blend_dense  (alpha=0.7, gamma=0.3, no prox)
  - jko_blend           (rerun for direct comparison)
  - jko_blend_dense     (rerun for direct comparison)
"""
from __future__ import annotations

import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset
from methods import Candidates
from retrieval import cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax
from jko import JKOConfig, run_jko
from evaluation import ndcg_at_k, recall_at_k, bootstrap_ci, paired_bootstrap_diff

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_idx_and_cache(name="scifact"):
    base = INDEX_ROOT
    with open(base / "doc_ids.json") as f:
        doc_ids = json.load(f)
    embeddings = np.load(base / "embeddings.npy")
    npz = np.load(base / "candidates_test.npz", allow_pickle=True)
    return doc_ids, embeddings, {
        "cand_idx":   npz["cand_idx"],
        "bm25_pool":  npz["bm25_pool"],
        "dense_pool": npz["dense_pool"],
        "rerank":     npz["rerank"],
        "q_ids":      [str(x) for x in npz["q_ids"]],
    }


VARIANTS = [
    ("noprox_blend",          "noproximal",  0.4, 0.6),
    ("kl_blend",              "kl",          0.4, 0.6),
    ("jko_blend",             "wasserstein", 0.4, 0.6),
    ("noprox_blend_dense",    "noproximal",  0.7, 0.3),
    ("kl_blend_dense",        "kl",          0.7, 0.3),
    ("jko_blend_dense",       "wasserstein", 0.7, 0.3),
]


def main():
    print("Loading SciFact test...")
    _, queries, qrels_test, _ = load_dataset("scifact").qrels, load_dataset("scifact").queries, load_dataset("scifact").qrels.get("test", {}), None
    ds = load_dataset("scifact")
    queries, qrels_test = ds.queries, ds.qrels["test"]
    doc_ids, embeddings, cache = load_idx_and_cache()
    q_ids = cache["q_ids"]

    common = dict(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1,
                  T=3, inner_steps=25, tau0=0.1)

    per_query = {name: defaultdict(list) for name, *_ in VARIANTS}
    t0 = time.time()
    for qi, qid in enumerate(tqdm(q_ids, desc="blend ablation")):
        if qid not in qrels_test:
            continue
        c = Candidates(
            cand_idx=cache["cand_idx"][qi],
            bm25_scores=cache["bm25_pool"][qi],
            dense_scores=cache["dense_pool"][qi],
            rerank_scores=cache["rerank"][qi],
        )
        Z = embeddings[c.cand_idx]
        C = cost_matrix_cosine(Z); K = redundancy_kernel(Z)

        for name, mode, alpha, gamma in VARIANTS:
            energy = -(alpha * normalize_minmax(c.dense_scores)
                       + gamma * normalize_minmax(c.rerank_scores))
            p0 = softmax_np(-energy, tau=common["tau0"])
            cfg = JKOConfig(
                h=common["h"], lam=common["lam"], rho=common["rho"],
                sinkhorn_eps=common["sinkhorn_eps"], T=common["T"],
                inner_steps=common["inner_steps"], mode=mode,
            )
            p_T, _ = run_jko(p0, energy, C, K, cfg)
            order = np.argsort(-p_T)[:20]
            dids = [doc_ids[i] for i in c.cand_idx[order]]
            per_query[name]["ndcg@10"].append(ndcg_at_k(dids, qrels_test[qid], 10))
            per_query[name]["recall@10"].append(recall_at_k(dids, qrels_test[qid], 10))
            per_query[name]["recall@20"].append(recall_at_k(dids, qrels_test[qid], 20))
    print(f"\nDone in {time.time()-t0:.1f}s")

    # summary
    summary = {}
    for m, d in per_query.items():
        for metric, scores in d.items():
            mean, lo, hi = bootstrap_ci(scores)
            summary.setdefault(m, {})[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}

    # paired diffs: jko vs kl with SAME energy
    paired = {}
    for energy_pair in [("jko_blend", "kl_blend"), ("jko_blend_dense", "kl_blend_dense"),
                        ("jko_blend", "noprox_blend"), ("kl_blend", "noprox_blend")]:
        a_name, b_name = energy_pair
        key = f"{a_name}_vs_{b_name}"
        paired[key] = {}
        for metric in ("ndcg@10", "recall@10", "recall@20"):
            diff, lo, hi = paired_bootstrap_diff(per_query[a_name][metric], per_query[b_name][metric])
            paired[key][metric] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}

    out = {"summary": summary, "paired": paired, "per_query": {m: dict(d) for m, d in per_query.items()}}
    (RESULTS_DIR / "blend_ablation.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved {RESULTS_DIR / 'blend_ablation.json'}\n")

    print("=" * 78)
    print("BLEND ABLATION (alpha=0.4, gamma=0.6) on SciFact test (n=300)")
    print("=" * 78)
    print(f"{'method':<22s} {'nDCG@10':>22s} {'Recall@10':>22s} {'Recall@20':>22s}")
    for m in [v[0] for v in VARIANTS]:
        s = summary[m]
        def cell(d): return f"{d['mean']:.3f}[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"
        print(f"{m:<22s} {cell(s['ndcg@10']):>22s} {cell(s['recall@10']):>22s} {cell(s['recall@20']):>22s}")

    print("\n=== Paired diffs (W - KL on same energy) ===")
    for key, mvals in paired.items():
        print(f"  {key}:")
        for metric, d in mvals.items():
            sig = " *" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else "  "
            print(f"    {metric:<10s} diff = {d['diff']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]{sig}")


if __name__ == "__main__":
    main()
