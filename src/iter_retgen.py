"""Iter-RetGen baseline (Shao et al., 2023, "Synergistic retrieval-generation").

Pipeline (1 iteration of refinement):
  1. Retrieve top-k_init evidence with the base method (hybrid + rerank).
  2. Use FLAN-T5-base to generate a short answer/summary from the evidence.
  3. Concatenate (claim + answer) as a refined query.
  4. Re-retrieve with the refined query.
  5. Return the refined top-k.

We use it as a baseline for "iterative distributional retrieval"
in the JKO-RAG comparison. This is the standard query-reformulation baseline
that ICLR / ICML reviewers expect to see.

Implementation notes:
  - We restrict iter-retgen to a single refinement iteration (the original
    paper shows 1-2 iterations give most of the gain on QA tasks).
  - The reranker pass is repeated on the new candidate pool (the cross-encoder
    is run again because reranker scores depend on the query).
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
from methods import Candidates, rerank_scores
from retrieval import Indices, normalize_minmax
from evaluation import ndcg_at_k, recall_at_k, bootstrap_ci, paired_bootstrap_diff

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TOKEN_RE = re.compile(r"\w+")


def tokenize(t): return TOKEN_RE.findall(t.lower())


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--k-init", type=int, default=5)
    p.add_argument("--k-final", type=int, default=10)
    args = p.parse_args()

    print(f"=== Iter-RetGen baseline on {args.dataset}/{args.split} ===")
    ds = load_dataset(args.dataset)
    qrels = ds.qrels[args.split]
    idx = load_index(args.dataset)

    # Encode queries with the dense model (same as everywhere)
    from sentence_transformers import SentenceTransformer
    print("Loading dense model...")
    dense = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # FLAN-T5-base for answer generation
    print("Loading FLAN-T5-base...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    lm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    lm.eval()

    # Subsample queries (deterministic)
    rng = np.random.default_rng(0)
    qids = [q for q in qrels if q in ds.queries and any(r > 0 for r in qrels[q].values())]
    if len(qids) > args.n_queries:
        sel = rng.choice(len(qids), args.n_queries, replace=False)
        qids = [qids[i] for i in sorted(sel)]
    print(f"Evaluating on {len(qids)} queries")

    per_query = {"rerank_baseline": defaultdict(list), "iter_retgen": defaultdict(list)}

    for qid in tqdm(qids, desc="iter-retgen"):
        q_text = ds.queries[qid]
        # === First pass ===
        q_emb = dense.encode([q_text], normalize_embeddings=True)[0].astype(np.float32)
        cand, bm25_pool, dense_pool = hybrid_pool(idx, q_text, q_emb, pool_size=200)
        texts = [idx.doc_texts[idx.doc_ids[int(j)]] for j in cand]
        rr = rerank_scores(q_text, texts, batch_size=64)

        # Original baseline top-k
        order = np.argsort(-rr)[:args.k_final]
        baseline_dids = [idx.doc_ids[int(cand[i])] for i in order]
        per_query["rerank_baseline"]["ndcg@10"].append(ndcg_at_k(baseline_dids, qrels[qid], 10))
        per_query["rerank_baseline"]["recall@10"].append(recall_at_k(baseline_dids, qrels[qid], 10))
        per_query["rerank_baseline"]["recall@20"].append(recall_at_k(baseline_dids, qrels[qid], 20))

        # === Iter-RetGen: generate from top-k_init, re-retrieve with refined query ===
        init_order = np.argsort(-rr)[:args.k_init]
        init_evidence = [idx.doc_texts[idx.doc_ids[int(cand[i])]][:500] for i in init_order]
        ev_block = "\n\n".join(f"Evidence {i+1}: {e}" for i, e in enumerate(init_evidence))
        prompt = (
            f"Given the evidence below, write a one-sentence factual answer "
            f"or summary that addresses the query.\n\n{ev_block}\n\n"
            f"Query: {q_text}\nAnswer:"
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = lm.generate(**inputs, max_new_tokens=40, do_sample=False, num_beams=1)
        answer = tok.decode(out[0], skip_special_tokens=True).strip()

        # Refined query
        refined = (q_text + " " + answer)[:500]
        refined_emb = dense.encode([refined], normalize_embeddings=True)[0].astype(np.float32)
        cand2, _, _ = hybrid_pool(idx, refined, refined_emb, pool_size=200)
        texts2 = [idx.doc_texts[idx.doc_ids[int(j)]] for j in cand2]
        rr2 = rerank_scores(refined, texts2, batch_size=64)
        order2 = np.argsort(-rr2)[:args.k_final]
        refined_dids = [idx.doc_ids[int(cand2[i])] for i in order2]
        per_query["iter_retgen"]["ndcg@10"].append(ndcg_at_k(refined_dids, qrels[qid], 10))
        per_query["iter_retgen"]["recall@10"].append(recall_at_k(refined_dids, qrels[qid], 10))
        per_query["iter_retgen"]["recall@20"].append(recall_at_k(refined_dids, qrels[qid], 20))

    summary = {}
    for m, d in per_query.items():
        for metric, scores in d.items():
            mean, lo, hi = bootstrap_ci(scores)
            summary.setdefault(m, {})[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}

    paired = {}
    for metric in ("ndcg@10", "recall@10", "recall@20"):
        d, lo, hi = paired_bootstrap_diff(per_query["iter_retgen"][metric],
                                           per_query["rerank_baseline"][metric])
        paired[metric] = {"diff": d, "ci_lo": lo, "ci_hi": hi}

    out = {
        "dataset": args.dataset, "split": args.split, "n_queries": len(qids),
        "k_init": args.k_init, "k_final": args.k_final,
        "summary": summary, "paired_iter_vs_baseline": paired,
        "per_query": {m: dict(d) for m, d in per_query.items()},
    }
    (RESULTS_DIR / f"iter_retgen_{args.dataset}.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\nSaved iter_retgen_{args.dataset}.json")

    print(f"\n=== Iter-RetGen vs baseline on {args.dataset}/{args.split} (n={len(qids)}) ===")
    for m in ["rerank_baseline", "iter_retgen"]:
        s = summary[m]
        print(f"  {m:<18s} nDCG@10={s['ndcg@10']['mean']:.4f}[{s['ndcg@10']['ci_lo']:.4f},{s['ndcg@10']['ci_hi']:.4f}] "
              f"R@10={s['recall@10']['mean']:.4f} R@20={s['recall@20']['mean']:.4f}")
    print("\nPaired diff (iter_retgen - baseline):")
    for metric, d in paired.items():
        sig = " *" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else "  "
        print(f"  {metric}: {d['diff']:+.4f}  [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]{sig}")


if __name__ == "__main__":
    main()
