"""Run Stage 1 retrieval experiment on a given BEIR dataset using a saved config.

Usage:
  python run_experiment_multi.py --dataset scifact --tuned-config results/best_hparams_scifact.json
  python run_experiment_multi.py --dataset nfcorpus --tuned-config results/best_hparams_scifact.json
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
from methods import Candidates, method_bm25, method_dense, method_hybrid_rrf, method_rerank, method_mmr
from retrieval import (
    Indices, cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax,
)
from jko import JKOConfig, run_jko
from evaluation import (
    ndcg_at_k, recall_at_k, precision_at_k, reciprocal_rank,
    semantic_diversity, bootstrap_ci, paired_bootstrap_diff,
)

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TOKEN_RE = re.compile(r"\w+")


def tokenize(t):
    return TOKEN_RE.findall(t.lower())


def load_index(name: str) -> Indices:
    base = INDEX_ROOT / name
    with open(base / "doc_ids.json") as f:
        doc_ids = json.load(f)
    with open(base / "doc_texts.json") as f:
        doc_texts = json.load(f)
    embeddings = np.load(base / "embeddings.npy")
    with open(base / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    return Indices(
        doc_ids=doc_ids,
        doc_id_to_idx={d: i for i, d in enumerate(doc_ids)},
        doc_texts=doc_texts,
        embeddings=embeddings,
        bm25=bm25_data["bm25"],
        bm25_tokenized=bm25_data["tokenized"],
    )


def load_cache(name: str, split: str = "test"):
    npz = np.load(INDEX_ROOT / name / f"candidates_{split}.npz", allow_pickle=True)
    return {
        "cand_idx":   npz["cand_idx"],
        "bm25_pool":  npz["bm25_pool"],
        "dense_pool": npz["dense_pool"],
        "rerank":     npz["rerank"],
        "q_ids":      [str(x) for x in npz["q_ids"]],
    }


def make_candidates_from_cache(cache, i, do_rerank=True):
    return Candidates(
        cand_idx=cache["cand_idx"][i],
        bm25_scores=cache["bm25_pool"][i],
        dense_scores=cache["dense_pool"][i],
        rerank_scores=cache["rerank"][i] if do_rerank else None,
    )


def make_energy(c: Candidates, alpha: float, gamma: float) -> np.ndarray:
    r = alpha * normalize_minmax(c.dense_scores) + gamma * normalize_minmax(c.rerank_scores)
    return -r


def jko_method(c, idx, mode, h, lam, rho, sinkhorn_eps, T, inner_steps,
               alpha, gamma, tau0, k):
    Z = idx.embeddings[c.cand_idx]
    C = cost_matrix_cosine(Z); K = redundancy_kernel(Z)
    energy = make_energy(c, alpha, gamma)
    p0 = softmax_np(-energy, tau=tau0)
    cfg = JKOConfig(h=h, lam=lam, rho=rho, sinkhorn_eps=sinkhorn_eps,
                    T=T, inner_steps=inner_steps, mode=mode)
    p_T, _ = run_jko(p0, energy, C, K, cfg)
    order = np.argsort(-p_T)[:k]
    return c.cand_idx[order].tolist(), p_T


def evaluate(topk_idx, idx, qrels, embeddings):
    dids = [idx.doc_ids[i] for i in topk_idx]
    return {
        "ndcg@10":      ndcg_at_k(dids, qrels, 10),
        "recall@5":     recall_at_k(dids, qrels, 5),
        "recall@10":    recall_at_k(dids, qrels, 10),
        "recall@20":    recall_at_k(dids, qrels, 20),
        "precision@5":  precision_at_k(dids, qrels, 5),
        "mrr@10":       reciprocal_rank(dids, qrels, 10),
        "diversity@10": semantic_diversity(topk_idx[:10], embeddings),
    }


METHOD_NAMES = [
    "bm25", "dense", "hybrid_rrf", "rerank", "mmr",
    "noprox", "kl_prox", "jko_prox",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--config-file", default=None, help="JSON of tuned hyperparams")
    p.add_argument("--out-suffix", default="")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = p.parse_args()

    print(f"Dataset: {args.dataset}, split: {args.split}")
    ds = load_dataset(args.dataset)
    qrels = ds.qrels[args.split]
    idx = load_index(args.dataset)
    cache = load_cache(args.dataset, args.split)
    q_ids = cache["q_ids"]

    # Default config (untuned)
    cfg = {
        "h": 0.5, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
        "T": 3, "inner_steps": 25, "tau0": 0.1,
        "alpha": 0.4, "gamma": 0.6,
    }
    if args.config_file:
        loaded = json.loads(Path(args.config_file).read_text())
        if "best" in loaded:
            cfg.update(loaded["best"]["cfg"])
        else:
            cfg.update(loaded)
        print(f"Loaded tuned config: {cfg}")

    q_emb_all = np.load(INDEX_ROOT / args.dataset / f"q_embeddings_{args.split}.npy")
    with open(INDEX_ROOT / args.dataset / f"q_ids_{args.split}.json") as f:
        emb_qids = json.load(f)
    q_to_i_emb = {q: i for i, q in enumerate(emb_qids)}

    per_seed: dict[int, dict[str, dict[str, list[float]]]] = {}
    retrieved_per_seed: dict[int, dict[str, dict[str, list[str]]]] = {}

    for seed in args.seeds:
        import torch
        torch.manual_seed(seed); np.random.seed(seed)
        per_query = {m: defaultdict(list) for m in METHOD_NAMES}
        retrieved = {m: {} for m in METHOD_NAMES}

        for qi, qid in enumerate(tqdm(q_ids, desc=f"{args.dataset}/{args.split}/seed{seed}")):
            if qid not in qrels:
                continue
            q_emb = q_emb_all[q_to_i_emb[qid]]
            c = make_candidates_from_cache(cache, qi)
            k_max = 20

            tops = {
                "bm25":       method_bm25(idx, ds.queries[qid], q_emb, k=k_max),
                "dense":      method_dense(idx, ds.queries[qid], q_emb, k=k_max),
                "hybrid_rrf": method_hybrid_rrf(c, k=k_max),
                "rerank":     method_rerank(c, k=k_max),
                "mmr":        method_mmr(c, idx, k=k_max, lambda_mmr=0.5),
            }
            common = dict(h=cfg["h"], lam=cfg["lam"], rho=cfg["rho"],
                          sinkhorn_eps=cfg["sinkhorn_eps"],
                          T=cfg["T"], inner_steps=cfg["inner_steps"],
                          alpha=cfg["alpha"], gamma=cfg["gamma"],
                          tau0=cfg["tau0"], k=k_max)
            ids_no, _ = jko_method(c, idx, "noproximal",  **common)
            ids_kl, _ = jko_method(c, idx, "kl",          **common)
            ids_jk, _ = jko_method(c, idx, "wasserstein", **common)
            tops["noprox"] = ids_no
            tops["kl_prox"] = ids_kl
            tops["jko_prox"] = ids_jk

            for m, doc_indices in tops.items():
                ms = evaluate(doc_indices, idx, qrels[qid], idx.embeddings)
                for metric, v in ms.items():
                    per_query[m][metric].append(v)
                retrieved[m][qid] = [idx.doc_ids[i] for i in doc_indices[:20]]

        per_seed[seed] = per_query
        retrieved_per_seed[seed] = retrieved

    # Average across seeds (per-query then bootstrap)
    summary = {}
    pq_avg = {m: {} for m in METHOD_NAMES}
    for m in METHOD_NAMES:
        metrics = list(per_seed[args.seeds[0]][m].keys())
        for metric in metrics:
            arrs = [np.asarray(per_seed[s][m][metric]) for s in args.seeds]
            avg = np.mean(np.stack(arrs), axis=0)
            pq_avg[m][metric] = avg.tolist()
            mean, lo, hi = bootstrap_ci(avg.tolist())
            summary.setdefault(m, {})[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                                                  "n": len(avg), "n_seeds": len(args.seeds)}

    paired = {}
    for baseline in ("rerank", "kl_prox", "hybrid_rrf"):
        if baseline not in pq_avg:
            continue
        paired[baseline] = {}
        for metric in ("ndcg@10", "recall@10", "recall@20"):
            row = {}
            for m in METHOD_NAMES:
                if m == baseline:
                    continue
                if metric not in pq_avg[m] or metric not in pq_avg[baseline]:
                    continue
                diff, lo, hi = paired_bootstrap_diff(pq_avg[m][metric], pq_avg[baseline][metric])
                row[m] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}
            paired[baseline][metric] = row

    out = {
        "dataset": args.dataset,
        "split": args.split,
        "config": cfg,
        "n_queries": sum(1 for q in q_ids if q in qrels),
        "seeds": args.seeds,
        "summary": summary,
        "paired": paired,
        "per_query_avg": pq_avg,
    }
    out_name = f"stage1_{args.dataset}{args.out_suffix}.json"
    (RESULTS_DIR / out_name).write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved {RESULTS_DIR / out_name}")

    print(f"\n=== {args.dataset} headline (mean over {len(args.seeds)} seed(s)) ===")
    print(f"{'method':<14s}  {'nDCG@10':>20s} {'Recall@10':>20s} {'Recall@20':>20s}")
    for m in METHOD_NAMES:
        s = summary[m]
        def cell(d): return f"{d['mean']:.3f}[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"
        print(f"{m:<14s}  {cell(s['ndcg@10']):>20s} {cell(s['recall@10']):>20s} {cell(s['recall@20']):>20s}")


if __name__ == "__main__":
    main()
