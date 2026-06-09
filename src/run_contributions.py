"""Run all four method contributions side-by-side on a dataset's test split.

Methods compared:

  - rerank             : top-k by reranker score (baseline 1)
  - mmr                : MMR with lambda=0.5 (baseline 2)
  - jko_blend          : VANILLA JKO with cosine cost (Stage 1 numbers, reproduces prior)
  - kl_blend           : KL-proximal JKO (decisive ablation)
  - nm_jko             : C1  Neural-metric JKO (learned low-rank cost)
  - bw_jko_a25         : C2  Bregman-JKO with alpha=0.25  (mostly KL)
  - bw_jko_a50         : C2  Bregman-JKO with alpha=0.50
  - bw_jko_a75         : C2  Bregman-JKO with alpha=0.75  (mostly W2)
  - jko_dual_select    : C3  JKO + DUAL-RANK abstain (selective retrieval)

Output: results/contrib_<dataset>.json with per-method per-query metrics +
bootstrap CIs + paired diffs vs vanilla jko_blend.
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
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset
from methods import Candidates
from retrieval import (
    Indices, cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax,
)
from jko import JKOConfig, run_jko, run_jko_with_duals
from evaluation import (
    ndcg_at_k, recall_at_k, semantic_diversity, bootstrap_ci, paired_bootstrap_diff,
)
from learned_metric import cost_matrix_learned, load_learned_metric

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TOKEN_RE = re.compile(r"\w+")


def tokenize(t): return TOKEN_RE.findall(t.lower())


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
# Method runners
# ----------------------------------------------------------------------

def rerank_topk(c, k):
    return c.cand_idx[np.argsort(-c.rerank_scores)[:k]].tolist()


def mmr_pool(c, idx, k, lam=0.5):
    Z = idx.embeddings[c.cand_idx]
    rel = normalize_minmax(c.rerank_scores); sim = Z @ Z.T
    selected = []; remaining = set(range(len(c.cand_idx)))
    while len(selected) < min(k, len(c.cand_idx)):
        best_i, best_score = -1, -1e18
        for i in remaining:
            if not selected: score = rel[i]
            else:
                div = max(sim[i, j] for j in selected)
                score = lam * rel[i] - (1 - lam) * div
            if score > best_score: best_score, best_i = score, i
        selected.append(best_i); remaining.discard(best_i)
    return c.cand_idx[selected].tolist()


def jko_topk_with_cost(c, idx, mode, alpha, gamma, k, jko_cfg, C_override=None, alpha_prox=None):
    """JKO with explicit cost matrix (cosine or learned). mode/alpha_prox configure proximal."""
    Z = idx.embeddings[c.cand_idx]
    C = C_override if C_override is not None else cost_matrix_cosine(Z)
    K = redundancy_kernel(Z)
    energy = -(alpha * normalize_minmax(c.dense_scores) + gamma * normalize_minmax(c.rerank_scores))
    p0 = softmax_np(-energy, tau=jko_cfg["tau0"])
    cfg = JKOConfig(
        h=jko_cfg["h"], lam=jko_cfg["lam"], rho=jko_cfg["rho"],
        sinkhorn_eps=jko_cfg["sinkhorn_eps"], T=jko_cfg["T"],
        inner_steps=jko_cfg["inner_steps"], mode=mode,
        alpha_prox=alpha_prox if alpha_prox is not None else 0.5,
    )
    p_T, _ = run_jko(p0, energy, C, K, cfg)
    return c.cand_idx[np.argsort(-p_T)[:k]].tolist(), p_T


def jko_dual_rank(c, idx, alpha, gamma, k, jko_cfg, C_override=None, abstain_threshold=None):
    """C3: JKO + dual-variable confidence. Returns (topk_indices, p_T, dual_f).

    dual_f is the dual potential from a final Sinkhorn read-out: high f_i means
    the OT problem assigns chunk i high "transport potential" -- i.e. it is
    a hard-to-displace participant in the distribution.
    """
    Z = idx.embeddings[c.cand_idx]
    C = C_override if C_override is not None else cost_matrix_cosine(Z)
    K = redundancy_kernel(Z)
    energy = -(alpha * normalize_minmax(c.dense_scores) + gamma * normalize_minmax(c.rerank_scores))
    p0 = softmax_np(-energy, tau=jko_cfg["tau0"])
    cfg = JKOConfig(
        h=jko_cfg["h"], lam=jko_cfg["lam"], rho=jko_cfg["rho"],
        sinkhorn_eps=jko_cfg["sinkhorn_eps"], T=jko_cfg["T"],
        inner_steps=jko_cfg["inner_steps"], mode="wasserstein",
    )
    p_T, f, g = run_jko_with_duals(p0, energy, C, K, cfg)
    return c.cand_idx[np.argsort(-p_T)[:k]].tolist(), p_T, f


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(topk_idx, idx, qrels, embeddings):
    dids = [idx.doc_ids[i] for i in topk_idx]
    return {
        "ndcg@10":      ndcg_at_k(dids, qrels, 10),
        "recall@10":    recall_at_k(dids, qrels, 10),
        "recall@20":    recall_at_k(dids, qrels, 20),
        "diversity@10": semantic_diversity(topk_idx[:10], embeddings),
    }


METHOD_NAMES = [
    "rerank", "mmr",
    "jko_blend", "kl_blend",
    "nm_jko",
    "bw_jko_a25", "bw_jko_a50", "bw_jko_a75",
    "jko_dual",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--config-file", default=None)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--no-nm", action="store_true", help="skip NM-JKO if no metric trained")
    args = p.parse_args()

    print(f"=== Contributions eval: {args.dataset}/{args.split} ===")
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
    jko_cfg = {k: v for k, v in base_cfg.items() if k in jko_keys}
    print(f"JKO config: {jko_cfg}")

    # Load learned metric if present
    W = None
    if not args.no_nm:
        try:
            W, info = load_learned_metric(args.dataset)
            print(f"  Loaded learned metric W: shape {W.shape}, trained on {info['n_train']} queries")
        except FileNotFoundError as e:
            print(f"  WARN: no learned metric for {args.dataset} -- skipping nm_jko")
            W = None

    per_query = {m: defaultdict(list) for m in METHOD_NAMES}
    dual_records = []   # (qid, f_distribution, top-1 was relevant?)
    t0 = time.time()
    k = 20
    for qi, qid in enumerate(tqdm(q_ids, desc=f"contrib/{args.dataset}")):
        if qid not in qrels: continue
        c = Candidates(cand_idx=cache["cand_idx"][qi], bm25_scores=cache["bm25_pool"][qi],
                       dense_scores=cache["dense_pool"][qi], rerank_scores=cache["rerank"][qi])
        Z = idx.embeddings[c.cand_idx]
        C_cos = cost_matrix_cosine(Z)
        C_W = cost_matrix_learned(Z, W) if W is not None else None

        # === Vanilla JKO ===
        tops_jko, _ = jko_topk_with_cost(c, idx, "wasserstein", 0.4, 0.6, k, jko_cfg, C_cos)
        tops_kl, _ = jko_topk_with_cost(c, idx, "kl", 0.4, 0.6, k, jko_cfg, C_cos)

        # === C1 NM-JKO ===
        if C_W is not None:
            tops_nm, _ = jko_topk_with_cost(c, idx, "wasserstein", 0.4, 0.6, k, jko_cfg, C_W)
        else:
            tops_nm = tops_jko  # fallback

        # === C2 Bregman-JKO ===
        tops_b25, _ = jko_topk_with_cost(c, idx, "bregman", 0.4, 0.6, k, jko_cfg, C_cos, alpha_prox=0.25)
        tops_b50, _ = jko_topk_with_cost(c, idx, "bregman", 0.4, 0.6, k, jko_cfg, C_cos, alpha_prox=0.50)
        tops_b75, _ = jko_topk_with_cost(c, idx, "bregman", 0.4, 0.6, k, jko_cfg, C_cos, alpha_prox=0.75)

        # === C3 DUAL-RANK ===
        tops_dual, _, dual_f = jko_dual_rank(c, idx, 0.4, 0.6, k, jko_cfg, C_cos)
        # store dual confidence of top-1
        top1_local = int(np.argmax(np.array([1.0 if i in tops_dual[:1] else 0.0 for i in range(len(c.cand_idx))])))
        top1_did = tops_dual[0]
        is_rel = idx.doc_ids[top1_did] in qrels[qid]
        # f at the chosen top-1 chunk:
        top1_local_for_f = int(np.where(c.cand_idx == top1_did)[0][0])
        dual_records.append({
            "qid": qid, "top1_f": float(dual_f[top1_local_for_f]),
            "f_mean": float(np.mean(dual_f)), "f_std": float(np.std(dual_f)),
            "is_rel": bool(is_rel),
        })

        all_tops = {
            "rerank": rerank_topk(c, k), "mmr": mmr_pool(c, idx, k, lam=0.5),
            "jko_blend": tops_jko, "kl_blend": tops_kl,
            "nm_jko": tops_nm,
            "bw_jko_a25": tops_b25, "bw_jko_a50": tops_b50, "bw_jko_a75": tops_b75,
            "jko_dual": tops_dual,
        }
        for m, doc_indices in all_tops.items():
            ms = evaluate(doc_indices, idx, qrels[qid], idx.embeddings)
            for metric, v in ms.items():
                per_query[m][metric].append(v)
    elapsed = time.time() - t0

    summary = {}
    for m, d in per_query.items():
        for metric, scores in d.items():
            mean, lo, hi = bootstrap_ci(scores)
            summary.setdefault(m, {})[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                                                  "n": len(scores)}

    paired = {}
    for pair in [("nm_jko", "jko_blend"), ("nm_jko", "kl_blend"),
                 ("bw_jko_a50", "jko_blend"), ("bw_jko_a50", "kl_blend"),
                 ("bw_jko_a25", "kl_blend"), ("bw_jko_a75", "jko_blend"),
                 ("jko_dual", "jko_blend"),
                 ("jko_blend", "rerank")]:
        a, b = pair
        paired[f"{a}_vs_{b}"] = {}
        for metric in ("ndcg@10", "recall@10", "diversity@10"):
            diff, lo, hi = paired_bootstrap_diff(per_query[a][metric], per_query[b][metric])
            paired[f"{a}_vs_{b}"][metric] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}

    # ECE for DUAL-RANK on top-1
    # Bin by f, measure mean is_rel within bin
    if dual_records:
        fs = np.array([r["top1_f"] for r in dual_records])
        rels = np.array([r["is_rel"] for r in dual_records], dtype=np.float32)
        # rank-normalise f to [0,1]
        from scipy.stats import rankdata
        f_norm = (rankdata(fs) - 1) / max(len(fs) - 1, 1)
        n_bins = 10
        ece = 0.0
        bin_calib = []
        for b_lo in np.linspace(0, 1, n_bins, endpoint=False):
            b_hi = b_lo + 1 / n_bins
            mask = (f_norm >= b_lo) & (f_norm < b_hi if b_hi < 1 else f_norm <= b_hi)
            if mask.sum() > 0:
                mean_conf = f_norm[mask].mean()
                mean_acc = rels[mask].mean()
                bin_calib.append({"bin_lo": float(b_lo), "bin_hi": float(b_hi),
                                  "n": int(mask.sum()),
                                  "mean_conf": float(mean_conf), "mean_acc": float(mean_acc)})
                ece += abs(mean_conf - mean_acc) * mask.sum() / len(fs)
        ece = float(ece)
    else:
        ece = float("nan"); bin_calib = []

    out = {
        "dataset": args.dataset, "split": args.split, "config": jko_cfg,
        "n_queries": sum(1 for q in q_ids if q in qrels),
        "elapsed_sec": elapsed, "summary": summary, "paired": paired,
        "dual_ece_top1": ece, "dual_calibration_bins": bin_calib,
        "n_train_metric": (int(info["n_train"]) if W is not None else 0),
        "metric_rank_r": (int(W.shape[0]) if W is not None else 0),
        "per_query": {m: dict(d) for m, d in per_query.items()},
    }
    out_name = f"contrib_{args.dataset}{args.out_suffix}.json"
    (RESULTS_DIR / out_name).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {RESULTS_DIR / out_name}")

    print(f"\n=== Headline ({args.dataset}/{args.split}) ===")
    print(f"{'method':<14s}  nDCG@10                R@10")
    for m in METHOD_NAMES:
        s = summary[m]; n = s["ndcg@10"]; r = s["recall@10"]
        print(f"{m:<14s}  {n['mean']:.3f}[{n['ci_lo']:.3f},{n['ci_hi']:.3f}]  {r['mean']:.3f}")
    print("\nKey paired diffs (ndcg@10):")
    for kk, mvals in paired.items():
        d = mvals["ndcg@10"]
        sig = "*" if (d['ci_lo'] > 0 or d['ci_hi'] < 0) else " "
        print(f"  {sig} {kk:<32s} {d['diff']:+.4f} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]")
    print(f"\nDUAL-RANK top-1 ECE: {ece:.4f}")


if __name__ == "__main__":
    main()
