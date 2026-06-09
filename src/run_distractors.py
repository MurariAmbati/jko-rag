"""Distractor-injection robustness experiment.

For each query, augment its candidate pool with N "distractors" — chunks that
are semantically similar to the gold doc(s) but are NOT relevant (per qrels).
We pull these from elsewhere in the corpus: for each gold doc d_gold of query q,
find the K nearest neighbours of d_gold in dense space that are NOT marked
relevant for ANY query in qrels (true distractors, not just neighbours of gold
that might also be relevant).

We then measure how each method's nDCG@10 / Recall@10 degrades as the number
of injected distractors grows from 0 to ~50. The proposal claim is that the
Wasserstein proximal resists collapse onto misleading chunks because moving
mass to a semantically close-but-wrong chunk costs little in W² (cost is
low) — but the entropy + redundancy + gold-relevance energy should still keep
the right mass distribution.

A more pessimistic interpretation: W's "preservation" of semantic clusters
makes it MORE susceptible to distractors that lie inside the gold cluster.
We test which prediction is right.
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
from retrieval import (
    Indices, cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax,
)
from jko import JKOConfig, run_jko
from evaluation import ndcg_at_k, recall_at_k, paired_bootstrap_diff, bootstrap_ci

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


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


def load_cache(name, split="test"):
    sub_p = index_dir(name) / f"candidates_{split}.npz"
    flat_p = INDEX_ROOT / f"candidates_{split}.npz"
    p = sub_p if sub_p.exists() else flat_p
    npz = np.load(p, allow_pickle=True)
    return {
        "cand_idx": npz["cand_idx"], "bm25_pool": npz["bm25_pool"],
        "dense_pool": npz["dense_pool"], "rerank": npz["rerank"],
        "q_ids": [str(x) for x in npz["q_ids"]],
    }


def find_distractors_for_query(
    idx: Indices, qrels_all: dict[str, dict[str, int]], gold_dids: list[str], k: int = 30,
) -> list[int]:
    """For each gold doc, find K dense nearest neighbours that are not relevant
    for ANY query. Returns up to k unique distractor *corpus indices*."""
    # Build set of all known-relevant doc ids
    known_relevant: set[str] = set()
    for q, dids in qrels_all.items():
        for d, r in dids.items():
            if r > 0: known_relevant.add(d)

    distractors: list[int] = []
    seen: set[int] = set()
    for gold in gold_dids:
        if gold not in idx.doc_id_to_idx: continue
        gi = idx.doc_id_to_idx[gold]
        z = idx.embeddings[gi]
        sims = idx.embeddings @ z
        order = np.argsort(-sims)
        cnt = 0
        for nb in order:
            nb = int(nb)
            if nb == gi or nb in seen: continue
            did = idx.doc_ids[nb]
            if did in known_relevant: continue  # this would be relevant somewhere
            distractors.append(nb)
            seen.add(nb)
            cnt += 1
            if cnt >= k: break
    return distractors[:k]


def inject_distractors(c: Candidates, distractors: list[int], idx: Indices, q_text: str):
    """Return new Candidates with distractors injected.
    Distractor BM25 and dense scores are computed fresh; reranker is approximated
    by their max dense score among existing candidates' rerank scale (we re-rerank below)."""
    import re as _re
    bm25_all = idx.bm25.get_scores(_re.findall(r"\w+", q_text.lower())).astype(np.float32)
    # Dense via dot product with query - we don't have q_emb here; use existing pool's q_emb-equivalent
    # We'll just use cosine to mean of existing relevant candidates as an approximation. Simpler:
    # leave dense at low value (this is a distractor; if it has high dense, it's not really a distractor).
    # Best: compute dense via query embedding. We pass it in below.
    dense_all = None  # filled by caller
    return bm25_all, dense_all


def stress_test_one(
    c0: Candidates, distractor_indices: list[int], idx: Indices, q_emb: np.ndarray,
    q_text: str, qrels_q: dict[str, int], n_inject: int,
    do_rerank=False,
) -> tuple[Candidates, list[str]]:
    """Build an augmented Candidates by appending `n_inject` distractors to the pool."""
    if n_inject == 0: return c0, []
    extra_idx = np.asarray(distractor_indices[:n_inject], dtype=np.int64)
    # Compute scores for extras
    bm25_extra = idx.bm25.get_scores(re.findall(r"\w+", q_text.lower())).astype(np.float32)[extra_idx]
    dense_extra = (idx.embeddings[extra_idx] @ q_emb).astype(np.float32)
    # Approximate reranker for distractors as midpoint of pool reranker range — keeps them
    # "competitive" so they have a real chance of being chosen, not trivially filtered.
    # This is a *stress* test, so we should give distractors a fighting chance.
    rr_min, rr_max = float(c0.rerank_scores.min()), float(c0.rerank_scores.max())
    rerank_extra = np.full(n_inject, 0.5 * (rr_min + rr_max), dtype=np.float32)

    cand_new = np.concatenate([c0.cand_idx, extra_idx])
    bm25_new = np.concatenate([c0.bm25_scores, bm25_extra])
    dense_new = np.concatenate([c0.dense_scores, dense_extra])
    rerank_new = np.concatenate([c0.rerank_scores, rerank_extra])
    distractor_dids = [idx.doc_ids[int(j)] for j in extra_idx]
    return Candidates(cand_idx=cand_new, bm25_scores=bm25_new,
                      dense_scores=dense_new, rerank_scores=rerank_new), distractor_dids


def jko_topk(c, idx, mode, alpha, gamma, k=10, cfg=None):
    Z = idx.embeddings[c.cand_idx]
    C = cost_matrix_cosine(Z); K = redundancy_kernel(Z)
    energy = -(alpha * normalize_minmax(c.dense_scores)
               + gamma * normalize_minmax(c.rerank_scores))
    p0 = softmax_np(-energy, tau=cfg.get("tau0", 0.1) if cfg else 0.1)
    jcfg = JKOConfig(
        h=cfg.get("h", 0.5) if cfg else 0.5,
        lam=cfg.get("lam", 0.05) if cfg else 0.05,
        rho=cfg.get("rho", 0.05) if cfg else 0.05,
        sinkhorn_eps=cfg.get("sinkhorn_eps", 0.1) if cfg else 0.1,
        T=cfg.get("T", 3) if cfg else 3,
        inner_steps=cfg.get("inner_steps", 25) if cfg else 25,
        mode=mode,
    )
    p_T, _ = run_jko(p0, energy, C, K, jcfg)
    return c.cand_idx[np.argsort(-p_T)[:k]].tolist()


def rerank_topk(c, k=10):
    return c.cand_idx[np.argsort(-c.rerank_scores)[:k]].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--n-queries", type=int, default=100,
                   help="Subsample queries (set high for full)")
    p.add_argument("--inject-counts", nargs="+", type=int, default=[0, 5, 10, 20, 40])
    p.add_argument("--config-file", default=None)
    args = p.parse_args()

    ds = load_dataset(args.dataset)
    qrels = ds.qrels[args.split]
    idx = load_index(args.dataset)
    cache = load_cache(args.dataset, args.split)
    q_ids = cache["q_ids"]
    # Read query embeddings
    sub_qe = index_dir(args.dataset) / f"q_embeddings_{args.split}.npy"
    flat_qe = INDEX_ROOT / f"q_embeddings_{args.split}.npy"
    qemb_path = sub_qe if sub_qe.exists() else flat_qe
    sub_qids = index_dir(args.dataset) / f"q_ids_{args.split}.json"
    flat_qids = INDEX_ROOT / f"q_ids_{args.split}.json"
    qids_path = sub_qids if sub_qids.exists() else flat_qids
    q_emb_all = np.load(qemb_path)
    with open(qids_path) as f: emb_qids = json.load(f)
    q_to_i_emb = {q: i for i, q in enumerate(emb_qids)}

    # Load config
    base_cfg = {"h": 0.5, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
                "T": 3, "inner_steps": 25, "tau0": 0.1}
    if args.config_file:
        loaded = json.loads(Path(args.config_file).read_text())
        base_cfg.update(loaded.get("best", {}).get("cfg", loaded))
    print(f"Using config: {base_cfg}")

    # Subsample queries (deterministic)
    rng = np.random.default_rng(0)
    q_with_rel = [q for q in q_ids if q in qrels and any(r > 0 for r in qrels[q].values())]
    if len(q_with_rel) > args.n_queries:
        idx_subset = rng.choice(len(q_with_rel), args.n_queries, replace=False)
        q_with_rel = [q_with_rel[i] for i in sorted(idx_subset)]
    print(f"Stress testing on {len(q_with_rel)} queries with relevant docs")

    results = {n: defaultdict(lambda: defaultdict(list)) for n in args.inject_counts}
    methods = ["rerank", "noprox_blend", "kl_blend", "jko_blend"]

    for qid in tqdm(q_with_rel, desc=f"{args.dataset} distractor stress"):
        qi = q_ids.index(qid)
        q_emb = q_emb_all[q_to_i_emb[qid]]
        c0 = Candidates(
            cand_idx=cache["cand_idx"][qi], bm25_scores=cache["bm25_pool"][qi],
            dense_scores=cache["dense_pool"][qi], rerank_scores=cache["rerank"][qi],
        )
        gold = [d for d, r in qrels[qid].items() if r > 0]
        distractors = find_distractors_for_query(idx, qrels, gold, k=max(args.inject_counts))

        for n_inject in args.inject_counts:
            c, distractor_dids = stress_test_one(
                c0, distractors, idx, q_emb, ds.queries[qid], qrels[qid], n_inject,
            )
            # rerank
            tops_rerank = rerank_topk(c, k=10)
            tops_noprox = jko_topk(c, idx, "noproximal", 0.4, 0.6, cfg=base_cfg)
            tops_kl = jko_topk(c, idx, "kl", 0.4, 0.6, cfg=base_cfg)
            tops_jko = jko_topk(c, idx, "wasserstein", 0.4, 0.6, cfg=base_cfg)
            for m, tops in zip(methods, [tops_rerank, tops_noprox, tops_kl, tops_jko]):
                dids = [idx.doc_ids[i] for i in tops]
                results[n_inject][m]["ndcg@10"].append(ndcg_at_k(dids, qrels[qid], 10))
                results[n_inject][m]["recall@10"].append(recall_at_k(dids, qrels[qid], 10))
                # Distractor leakage: fraction of top-10 that are distractors
                distractor_set = set(distractor_dids[:n_inject])
                leak = sum(1 for d in dids if d in distractor_set) / 10
                results[n_inject][m]["distractor_leakage@10"].append(leak)

    # summarize
    summary = {}
    for n in args.inject_counts:
        summary[n] = {}
        for m in methods:
            for metric in ["ndcg@10", "recall@10", "distractor_leakage@10"]:
                mean, lo, hi = bootstrap_ci(results[n][m][metric])
                summary[n].setdefault(m, {})[metric] = {
                    "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(results[n][m][metric]),
                }

    # paired W vs KL at each level
    paired = {}
    for n in args.inject_counts:
        paired[n] = {}
        for metric in ["ndcg@10", "recall@10", "distractor_leakage@10"]:
            d, lo, hi = paired_bootstrap_diff(
                results[n]["jko_blend"][metric], results[n]["kl_blend"][metric])
            paired[n][metric] = {"diff": d, "ci_lo": lo, "ci_hi": hi}

    out = {
        "dataset": args.dataset, "split": args.split, "config": base_cfg,
        "inject_counts": args.inject_counts, "n_queries": len(q_with_rel),
        "summary": summary, "paired_jko_vs_kl": paired,
    }
    name = f"distractors_{args.dataset}.json"
    (RESULTS_DIR / name).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {RESULTS_DIR / name}")

    print(f"\n=== Distractor injection on {args.dataset} (n={len(q_with_rel)}) ===")
    print("Distractor leakage @10 (fraction of top-10 that are injected distractors)")
    print(f"{'N inject':>10s}  " + "  ".join(f"{m:>18s}" for m in methods))
    for n in args.inject_counts:
        row = [f"N={n:<5d} "]
        for m in methods:
            v = summary[n][m]["distractor_leakage@10"]
            row.append(f"{v['mean']:.4f}[{v['ci_lo']:.4f},{v['ci_hi']:.4f}]")
        print("  ".join(row))
    print("\nnDCG@10 vs # distractors:")
    print(f"{'N inject':>10s}  " + "  ".join(f"{m:>18s}" for m in methods))
    for n in args.inject_counts:
        row = [f"N={n:<5d} "]
        for m in methods:
            v = summary[n][m]["ndcg@10"]
            row.append(f"{v['mean']:.4f}[{v['ci_lo']:.4f},{v['ci_hi']:.4f}]")
        print("  ".join(row))

    print(f"\nPaired (jko - kl) on distractor_leakage@10 (negative = jko leaks LESS):")
    for n in args.inject_counts:
        d = paired[n]["distractor_leakage@10"]
        sig = " *" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else "  "
        print(f"  N={n:<5d}  diff={d['diff']:+.4f}  [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]{sig}")


if __name__ == "__main__":
    main()
