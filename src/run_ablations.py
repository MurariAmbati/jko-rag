"""Full ablation matrix on SciFact dev (or train held-out).

Ablation list from the proposal:
- W_full       : Wasserstein-prox, semantic cost matrix
- KL_prox      : KL-proximal instead of W
- no_prox      : no proximal term (energy + entropy + redundancy only)
- random_C     : Wasserstein-prox but cost matrix is random uniform[0, 1]
- identity_C   : Wasserstein-prox but cost matrix C_ii=0, C_ij=1 (no semantics)
- no_entropy   : lambda=0
- no_redund    : rho=0
- one_step     : T=1
- many_step    : T=5
- rerank_only  : energy = -rerank score, no JKO at all (just argmax)

For each ablation, sweep nothing else. Just compare on the same queries.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset
from methods import Candidates
from retrieval import (
    cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax,
)
from jko import JKOConfig, run_jko
from evaluation import (
    ndcg_at_k, recall_at_k, paired_bootstrap_diff, bootstrap_ci,
)

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_index(name: str):
    sub = INDEX_ROOT / name
    base = sub if (sub / "doc_ids.json").exists() else INDEX_ROOT
    with open(base / "doc_ids.json") as f:
        doc_ids = json.load(f)
    embeddings = np.load(base / "embeddings.npy")
    return doc_ids, embeddings


def load_cache(name: str, split: str = "test"):
    sub_p = INDEX_ROOT / name / f"candidates_{split}.npz"
    flat_p = INDEX_ROOT / f"candidates_{split}.npz"
    p = sub_p if sub_p.exists() else flat_p
    npz = np.load(p, allow_pickle=True)
    return {
        "cand_idx":   npz["cand_idx"],
        "bm25_pool":  npz["bm25_pool"],
        "dense_pool": npz["dense_pool"],
        "rerank":     npz["rerank"],
        "q_ids":      [str(x) for x in npz["q_ids"]],
    }


def make_energy(c, alpha, gamma):
    return -(alpha * normalize_minmax(c.dense_scores) + gamma * normalize_minmax(c.rerank_scores))


ABLATIONS = {
    # name: (mode, modifier on C, modifier on lam, modifier on rho, T)
    "W_full":       ("wasserstein", "semantic",       None, None, 3),
    "KL_prox":      ("kl",          "semantic",       None, None, 3),
    "no_prox":      ("noproximal",  "semantic",       None, None, 3),
    "random_C":     ("wasserstein", "random",         None, None, 3),
    "identity_C":   ("wasserstein", "identity",       None, None, 3),
    "no_entropy":   ("wasserstein", "semantic",       0.0,  None, 3),
    "no_redund":    ("wasserstein", "semantic",       None, 0.0,  3),
    "one_step":     ("wasserstein", "semantic",       None, None, 1),
    "many_step":    ("wasserstein", "semantic",       None, None, 5),
}


def build_C(Z: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    if kind == "semantic":
        return cost_matrix_cosine(Z)
    if kind == "random":
        rng = np.random.default_rng(seed)
        C = rng.uniform(0.0, 4.0, size=(Z.shape[0], Z.shape[0])).astype(np.float32)
        # symmetrize and zero diagonal
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        return C
    if kind == "identity":
        n = Z.shape[0]
        C = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(C, 0.0)
        return C
    raise ValueError(kind)


def eval_ablation(
    ablation: dict, cands, doc_ids, embeddings, qrels, base_cfg, k=10,
):
    ndcg_scores, rec_scores = [], []
    qrels_qids = set(qrels.keys())
    for qi, qid, c in cands:
        if qid not in qrels_qids:
            continue
        Z = embeddings[c.cand_idx]
        C = build_C(Z, ablation["C_kind"])
        K = redundancy_kernel(Z)
        energy = make_energy(c, base_cfg["alpha"], base_cfg["gamma"])
        p0 = softmax_np(-energy, tau=base_cfg["tau0"])
        cfg = JKOConfig(
            h=base_cfg["h"],
            lam=ablation.get("lam", base_cfg["lam"]),
            rho=ablation.get("rho", base_cfg["rho"]),
            sinkhorn_eps=base_cfg["sinkhorn_eps"],
            T=ablation["T"],
            inner_steps=base_cfg["inner_steps"],
            mode=ablation["mode"],
        )
        p_T, _ = run_jko(p0, energy, C, K, cfg)
        order = np.argsort(-p_T)[:k]
        dids = [doc_ids[i] for i in c.cand_idx[order]]
        ndcg_scores.append(ndcg_at_k(dids, qrels[qid], k))
        rec_scores.append(recall_at_k(dids, qrels[qid], k))
    return {"ndcg@10": ndcg_scores, "recall@10": rec_scores}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--config-file", default=None)
    args = p.parse_args()

    print(f"Running ablations on {args.dataset}/{args.split}")
    ds = load_dataset(args.dataset)
    qrels = ds.qrels[args.split]
    doc_ids, embeddings = load_index(args.dataset)
    cache = load_cache(args.dataset, args.split)
    q_ids = cache["q_ids"]
    cands = [
        (i, q, Candidates(
            cand_idx=cache["cand_idx"][i], bm25_scores=cache["bm25_pool"][i],
            dense_scores=cache["dense_pool"][i], rerank_scores=cache["rerank"][i]))
        for i, q in enumerate(q_ids)
    ]

    base_cfg = {
        "h": 0.5, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
        "T": 3, "inner_steps": 25, "tau0": 0.1,
        "alpha": 0.4, "gamma": 0.6,
    }
    if args.config_file:
        loaded = json.loads(Path(args.config_file).read_text())
        if "best" in loaded:
            base_cfg.update(loaded["best"]["cfg"])
        else:
            base_cfg.update(loaded)
    print(f"Base config: {base_cfg}\n")

    per_ablation = {}
    for ab_name, (mode, C_kind, lam_override, rho_override, T) in tqdm(
        ABLATIONS.items(), desc="ablations"
    ):
        ab = {"mode": mode, "C_kind": C_kind, "T": T}
        if lam_override is not None: ab["lam"] = lam_override
        if rho_override is not None: ab["rho"] = rho_override
        out = eval_ablation(ab, cands, doc_ids, embeddings, qrels, base_cfg)
        per_ablation[ab_name] = out

    # summary
    summary = {}
    for ab, d in per_ablation.items():
        for metric, scores in d.items():
            mean, lo, hi = bootstrap_ci(scores)
            summary.setdefault(ab, {})[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}

    # paired diffs vs W_full
    paired = {}
    for metric in ("ndcg@10", "recall@10"):
        paired[metric] = {}
        for ab in per_ablation:
            if ab == "W_full":
                continue
            diff, lo, hi = paired_bootstrap_diff(
                per_ablation["W_full"][metric], per_ablation[ab][metric])
            paired[metric][ab] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}

    out_path = RESULTS_DIR / f"ablations_{args.dataset}_{args.split}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset, "split": args.split,
        "base_cfg": base_cfg,
        "summary": summary, "paired_vs_W_full": paired,
        "per_ablation": per_ablation,
    }, indent=2, default=float))
    print(f"Saved {out_path}\n")

    print("=" * 78)
    print(f"ABLATION TABLE on {args.dataset}/{args.split}")
    print("=" * 78)
    print(f"{'ablation':<15s} {'nDCG@10':>26s} {'Recall@10':>26s}")
    for ab in ABLATIONS:
        s = summary[ab]
        n = lambda d: f"{d['mean']:.3f}[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"
        print(f"  {ab:<13s} {n(s['ndcg@10']):>26s} {n(s['recall@10']):>26s}")

    print(f"\nPaired diff (W_full - <ablation>) — positive means W_full is better:")
    for ab in ABLATIONS:
        if ab == "W_full":
            continue
        d = paired["ndcg@10"][ab]
        sig = " *" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else "  "
        print(f"  W_full - {ab:<13s}  nDCG@10: {d['diff']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]{sig}")


if __name__ == "__main__":
    main()
