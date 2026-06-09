"""D1b -- DUAL-RANK selective coverage curves.

Given JKO retrieval distributions p_T(q) for a set of queries q, the Sinkhorn
dual potential f (returned by run_jko_with_duals) assigns a real number f_i
to each candidate. We define a *query-level confidence*:

    conf(q) = f_{top-1(q)}  -  median_i f_i

i.e. how much the top-1 chunk "sticks out" in transport potential from the
median chunk. High conf -> the OT problem strongly favours the chosen chunk.

For each method (vanilla JKO, NM-JKO, BW-JKO, KL-JKO -- though KL has no
duals so it gets a placeholder), we compute conf(q) on the existing contrib
results and sort queries by conf.

We then produce a SELECTIVE PRECISION-COVERAGE CURVE:
    - At coverage c in {1.0, 0.9, 0.8, ..., 0.1}, keep the top c-fraction of
      queries by conf(q).
    - Report mean nDCG@10 and Recall@10 on the kept set.

If DUAL-RANK is informative, the precision should INCREASE as we shrink
coverage (we abstain on low-confidence queries). The slope of that curve is a
useful new evaluation axis for retrievers.

Output: results/dual_selective_<dataset>.json with the curves.
"""
from __future__ import annotations

import argparse
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
from retrieval import (Indices, cost_matrix_cosine, redundancy_kernel,
                       softmax_np, normalize_minmax)
from jko import JKOConfig, run_jko_with_duals
from learned_metric import cost_matrix_learned, load_learned_metric
from evaluation import ndcg_at_k, recall_at_k, bootstrap_ci

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


def compute_conf_from_duals(f: np.ndarray, top_local_idx: int) -> float:
    """Confidence = f at chosen top1 minus median f.  Larger = more confident."""
    med = float(np.median(f))
    return float(f[top_local_idx] - med)


def softmax_baseline_conf(scores: np.ndarray) -> float:
    """Baseline confidence: softmax max minus uniform expectation."""
    s = (scores - scores.max())
    p = np.exp(s) / np.exp(s).sum()
    return float(p.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--config-file", default=None)
    p.add_argument("--use-learned-metric", action="store_true")
    args = p.parse_args()

    print(f"=== DUAL-RANK selective coverage: {args.dataset}/{args.split} ===")
    ds = load_dataset(args.dataset)
    qrels = ds.qrels[args.split]
    idx = load_index(args.dataset)
    cache = load_cache(args.dataset, args.split)
    q_ids = cache["q_ids"]

    base_cfg = {"h": 0.5, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
                "T": 3, "inner_steps": 25, "tau0": 0.1}
    if args.config_file:
        loaded = json.loads(Path(args.config_file).read_text())
        if "best" in loaded: base_cfg.update(loaded["best"]["cfg"])
        else: base_cfg.update(loaded)
    jko_keys = {"h", "lam", "rho", "sinkhorn_eps", "T", "inner_steps", "tau0"}
    jcfg = {k: v for k, v in base_cfg.items() if k in jko_keys}

    W = None
    if args.use_learned_metric:
        try:
            W, info = load_learned_metric(args.dataset)
            print(f"  Loaded learned metric W: {W.shape}")
        except FileNotFoundError:
            print(f"  No learned metric for {args.dataset}; using cosine cost")
            W = None

    cfg = JKOConfig(
        h=jcfg["h"], lam=jcfg["lam"], rho=jcfg["rho"],
        sinkhorn_eps=jcfg["sinkhorn_eps"], T=jcfg["T"],
        inner_steps=jcfg["inner_steps"], mode="wasserstein",
    )

    per_query_info = []   # list of {qid, conf_dual, conf_softmax, ndcg@10, recall@10}
    t0 = time.time()
    for qi, qid in enumerate(tqdm(q_ids, desc=f"dual_sel/{args.dataset}")):
        if qid not in qrels: continue
        c = Candidates(cand_idx=cache["cand_idx"][qi], bm25_scores=cache["bm25_pool"][qi],
                       dense_scores=cache["dense_pool"][qi], rerank_scores=cache["rerank"][qi])
        Z = idx.embeddings[c.cand_idx]
        C = cost_matrix_learned(Z, W) if W is not None else cost_matrix_cosine(Z)
        Kr = redundancy_kernel(Z)
        energy = -(0.4 * normalize_minmax(c.dense_scores)
                  + 0.6 * normalize_minmax(c.rerank_scores))
        p0 = softmax_np(-energy, tau=jcfg["tau0"])
        p_T, f, g = run_jko_with_duals(p0, energy, C, Kr, cfg)

        top_local = int(np.argmax(p_T))
        top_global = int(c.cand_idx[top_local])
        topk_local = np.argsort(-p_T)[:20]
        topk_global = c.cand_idx[topk_local].tolist()
        dids = [idx.doc_ids[i] for i in topk_global]
        rels = qrels[qid]

        # Confidences
        conf_dual = compute_conf_from_duals(f, top_local)
        # baseline 1: softmax max of rerank scores
        conf_softmax = softmax_baseline_conf(c.rerank_scores)
        # baseline 2: top-1 minus second-highest in p_T (margin)
        sp = np.sort(p_T)[::-1]
        conf_margin = float(sp[0] - sp[1])

        per_query_info.append({
            "qid": qid,
            "conf_dual": conf_dual, "conf_softmax": conf_softmax,
            "conf_margin": conf_margin,
            "ndcg@10": ndcg_at_k(dids, rels, 10),
            "recall@10": recall_at_k(dids, rels, 10),
            "top1_is_relevant": 1 if idx.doc_ids[top_global] in rels else 0,
        })
    elapsed = time.time() - t0

    # Compute selective curves for each confidence signal
    coverages = [1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
    curves = {}
    for conf_key in ("conf_dual", "conf_softmax", "conf_margin"):
        sorted_q = sorted(per_query_info, key=lambda x: -x[conf_key])  # high conf first
        curve = []
        for cov in coverages:
            n_keep = max(1, int(round(cov * len(sorted_q))))
            kept = sorted_q[:n_keep]
            n = len(kept)
            ndcg_vals = [r["ndcg@10"] for r in kept]
            rec_vals = [r["recall@10"] for r in kept]
            p1_rel = [r["top1_is_relevant"] for r in kept]
            curve.append({
                "coverage": cov, "n_kept": n,
                "ndcg@10_mean": float(np.mean(ndcg_vals)),
                "recall@10_mean": float(np.mean(rec_vals)),
                "top1_acc": float(np.mean(p1_rel)),
            })
        curves[conf_key] = curve

    out = {
        "dataset": args.dataset, "split": args.split,
        "n_queries": len(per_query_info),
        "elapsed_sec": elapsed,
        "used_learned_metric": W is not None,
        "selective_curves": curves,
        "per_query": per_query_info,
    }
    out_name = f"dual_selective_{args.dataset}.json"
    (RESULTS_DIR / out_name).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {RESULTS_DIR / out_name}")

    print(f"\n=== Selective coverage curves: {args.dataset} ===")
    print(f"{'cov':<6s} | {'dual_nDCG':<10s} {'dual_top1':<10s} | "
          f"{'soft_nDCG':<10s} {'soft_top1':<10s} | {'marg_nDCG':<10s} {'marg_top1':<10s}")
    for i, cov in enumerate(coverages):
        d = curves["conf_dual"][i]; s = curves["conf_softmax"][i]; m = curves["conf_margin"][i]
        print(f"{cov:<6.2f} | {d['ndcg@10_mean']:<10.3f} {d['top1_acc']:<10.3f} | "
              f"{s['ndcg@10_mean']:<10.3f} {s['top1_acc']:<10.3f} | "
              f"{m['ndcg@10_mean']:<10.3f} {m['top1_acc']:<10.3f}")


if __name__ == "__main__":
    main()
