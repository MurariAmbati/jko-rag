"""D1a -- stability evaluation for new contribution methods.

Mirrors run_stability_multi.py but compares:
  - jko_rerank      (vanilla Wasserstein baseline)
  - kl_rerank       (KL proximal baseline)
  - nm_jko_rerank   (C1: vanilla cost replaced by learned metric W_C)
  - bw_jko_a50      (C2: Bregman alpha=0.5 mix of W^2 and KL)
  - bw_jko_a25      (C2: more KL)
  - bw_jko_a75      (C2: more Wasserstein)

We use the standard 3-perturbation protocol (drop-stop, hedge, lower-no-punct)
and report W_C(p_T(q), p_T(q')) over the union of the two candidate pools.
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
from learned_metric import cost_matrix_learned, load_learned_metric

# Borrow helpers
sys.path.insert(0, str(Path(__file__).parent))
from run_stability_multi import (
    PERTURBATIONS, hybrid_pool, build_candidates, encode_one,
    w_distance_on_full_pool, load_index,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def jko_distribution_explicit(c, idx, mode, alpha_e, gamma_e, cfg_dict, C_override=None,
                                alpha_prox=0.5):
    """Run JKO with explicit hyperparameters and an optional cost-matrix override."""
    Z = idx.embeddings[c.cand_idx]
    C = C_override if C_override is not None else cost_matrix_cosine(Z)
    K = redundancy_kernel(Z)
    r = alpha_e * normalize_minmax(c.dense_scores) + gamma_e * normalize_minmax(c.rerank_scores)
    energy = -r
    p0 = softmax_np(-energy, tau=cfg_dict["tau0"])
    cfg = JKOConfig(
        h=cfg_dict["h"], lam=cfg_dict["lam"], rho=cfg_dict["rho"],
        sinkhorn_eps=cfg_dict["sinkhorn_eps"], T=cfg_dict["T"],
        inner_steps=cfg_dict["inner_steps"], mode=mode, alpha_prox=alpha_prox,
    )
    p_T, _ = run_jko(p0, energy, C, K, cfg)
    return p_T, c.cand_idx


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

    cfg = {"h": 2.0, "lam": 0.1, "rho": 0.05, "sinkhorn_eps": 0.2,
           "T": 2, "inner_steps": 15, "tau0": 1.0}    # fast config

    # Learned metric (if available)
    W = None
    try:
        W, info = load_learned_metric(args.dataset)
        print(f"  Loaded learned metric W: {W.shape}")
    except FileNotFoundError:
        print(f"  No learned metric for {args.dataset}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    from pathlib import Path as _P
    sub_cache = (_P(idx.doc_ids[0]).parent if False else None)
    cache = None
    cache_qids = []
    sub_cache_path = (Path(__file__).resolve().parents[1] / "indices" / args.dataset / "candidates_test.npz")
    if not sub_cache_path.exists():
        sub_cache_path = Path(__file__).resolve().parents[1] / "indices" / "candidates_test.npz"
    if sub_cache_path.exists():
        cache = np.load(sub_cache_path, allow_pickle=True)
        cache_qids = [str(x) for x in cache["q_ids"]]
    qid_to_ci = {q: i for i, q in enumerate(cache_qids)}

    all_qids = list(qrels_test.keys())
    rng = np.random.default_rng(args.seed)
    qids = list(rng.choice(all_qids, size=min(args.n_queries, len(all_qids)), replace=False))
    print(f"Stability over {len(qids)} queries x {len(PERTURBATIONS)} perturbations")

    # Methods: name -> (mode, use_learned_metric, alpha_prox)
    method_specs = [
        ("jko_rerank",   "wasserstein", False, 1.0),   # vanilla
        ("kl_rerank",    "kl",          False, 0.0),   # KL baseline
        ("bw_jko_a25",   "bregman",     False, 0.25),
        ("bw_jko_a50",   "bregman",     False, 0.50),
        ("bw_jko_a75",   "bregman",     False, 0.75),
    ]
    if W is not None:
        method_specs.append(("nm_jko",       "wasserstein", True,  1.0))
        method_specs.append(("nm_bw_a50",    "bregman",     True,  0.50))

    instab = {}  # (method, perturb_name) -> list of W_C values
    for qid in tqdm(qids, desc=f"{args.dataset} stability_new"):
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
        Z_q = idx.embeddings[c_q.cand_idx]
        C_q_W = cost_matrix_learned(Z_q, W) if W is not None else None

        dists_base = {}; cands_base = {}
        for name, mode, use_W, alpha_prox in method_specs:
            C_override = C_q_W if use_W else None
            p_T, cand = jko_distribution_explicit(c_q, idx, mode, 0.0, 1.0, cfg,
                                                   C_override=C_override, alpha_prox=alpha_prox)
            dists_base[name] = p_T; cands_base[name] = cand

        for p_name, fn in PERTURBATIONS:
            q_perturbed = fn(q_text, seed=args.seed)
            if q_perturbed.strip() == q_text.strip(): continue
            q_emb_p = encode_one(model, q_perturbed)
            c_p = build_candidates(idx, q_perturbed, q_emb_p)
            Z_p = idx.embeddings[c_p.cand_idx]
            C_p_W = cost_matrix_learned(Z_p, W) if W is not None else None

            for name, mode, use_W, alpha_prox in method_specs:
                C_override_p = C_p_W if use_W else None
                p_T_p, cand_p = jko_distribution_explicit(c_p, idx, mode, 0.0, 1.0, cfg,
                                                           C_override=C_override_p, alpha_prox=alpha_prox)
                w = w_distance_on_full_pool(
                    dists_base[name], cands_base[name], p_T_p, cand_p, idx.embeddings)
                instab.setdefault((name, p_name), []).append(w)

    summary = {}
    means_perturb_avg = {}
    for (mname, p_name), vals in instab.items():
        arr = np.asarray(vals)
        summary.setdefault(mname, {})[p_name] = {
            "mean": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr)),
        }
    for mname in summary:
        m = float(np.mean([summary[mname][pn]["mean"] for pn in summary[mname]]))
        means_perturb_avg[mname] = m

    out = {
        "dataset": args.dataset, "n_queries": len(qids), "config": cfg,
        "used_learned_metric": W is not None,
        "summary": summary,
        "per_method_mean_over_perturbations": means_perturb_avg,
    }
    name = f"stability_new_{args.dataset}{args.out_suffix}.json"
    (RESULTS_DIR / name).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {RESULTS_DIR / name}")

    print(f"\n=== Stability of new methods ({args.dataset}) ===")
    print(f"{'method':<14s}  Mean W_C  drop_stop   hedge      lower_nop")
    ordered = sorted(means_perturb_avg.items(), key=lambda x: x[1])
    for m, mean in ordered:
        row = summary[m]
        print(f"  {m:<14s}  {mean:.4f}    "
              f"{row.get('drop_stop', {}).get('mean', float('nan')):.4f}      "
              f"{row.get('hedge', {}).get('mean', float('nan')):.4f}     "
              f"{row.get('lower_nop', {}).get('mean', float('nan')):.4f}")


if __name__ == "__main__":
    main()
