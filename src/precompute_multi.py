"""Precompute candidate pools + reranker scores for all BEIR datasets / splits.

Saved to indices/<name>/candidates_<split>.npz with arrays:
  cand_idx: (Q, M) int64, indices into doc_ids
  bm25_pool: (Q, M) float32
  dense_pool: (Q, M) float32
  rerank:    (Q, M) float32
  q_ids:     (Q,) string array
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset
from methods import rerank_scores

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
POOL_SIZE = 200
TOKEN_RE = re.compile(r"\w+")


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(t.lower())


def load_index(name: str):
    base = INDEX_ROOT / name
    with open(base / "doc_ids.json") as f:
        doc_ids = json.load(f)
    with open(base / "doc_texts.json") as f:
        doc_texts = json.load(f)
    embeddings = np.load(base / "embeddings.npy")
    with open(base / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    return {
        "doc_ids": doc_ids,
        "doc_texts": doc_texts,
        "embeddings": embeddings,
        "bm25": bm25_data["bm25"],
    }


def hybrid_pool(idx, q_text: str, q_emb: np.ndarray, pool_size: int = 200, each_n: int = 500):
    bm25_all = idx["bm25"].get_scores(tokenize(q_text)).astype(np.float32)
    dense_all = (idx["embeddings"] @ q_emb).astype(np.float32)
    en = min(each_n, len(bm25_all) - 1, len(dense_all) - 1)
    bm25_top = np.argpartition(-bm25_all, en)[:en]; bm25_top = bm25_top[np.argsort(-bm25_all[bm25_top])]
    dense_top = np.argpartition(-dense_all, en)[:en]; dense_top = dense_top[np.argsort(-dense_all[dense_top])]
    fused = {}; K = 60
    for r, di in enumerate(bm25_top.tolist()):
        fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    for r, di in enumerate(dense_top.tolist()):
        fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    items = sorted(fused.items(), key=lambda kv: -kv[1])[:pool_size]
    cand = np.array([i for i, _ in items], dtype=np.int64)
    if len(cand) < pool_size:
        # pad with top BM25 if necessary
        extra = [int(i) for i in bm25_top if int(i) not in set(cand.tolist())]
        cand = np.concatenate([cand, np.asarray(extra[: pool_size - len(cand)], dtype=np.int64)])
    return cand, bm25_all[cand], dense_all[cand]


def precompute(name: str, split: str = "test"):
    ds = load_dataset(name)
    if split not in ds.qrels:
        print(f"[{name}] no {split} qrels, skipping")
        return
    idx = load_index(name)
    out_path = INDEX_ROOT / name / f"candidates_{split}.npz"
    if out_path.exists():
        print(f"[{name}/{split}] cache exists at {out_path}; skipping (delete to recompute)")
        return

    q_ids_path = INDEX_ROOT / name / f"q_ids_{split}.json"
    q_emb_path = INDEX_ROOT / name / f"q_embeddings_{split}.npy"
    with open(q_ids_path) as f:
        qids = json.load(f)
    q_emb = np.load(q_emb_path)

    M = POOL_SIZE
    Q = len(qids)
    cand_arr = np.zeros((Q, M), dtype=np.int64)
    bm25_arr = np.zeros((Q, M), dtype=np.float32)
    dense_arr = np.zeros((Q, M), dtype=np.float32)
    rerank_arr = np.zeros((Q, M), dtype=np.float32)
    print(f"[{name}/{split}] candidate pools for {Q} queries...")
    for i, qid in enumerate(tqdm(qids, desc=f"{name}/{split} pools")):
        cand, b, d = hybrid_pool(idx, ds.queries[qid], q_emb[i], pool_size=M)
        cand_arr[i] = cand; bm25_arr[i] = b; dense_arr[i] = d
    print(f"[{name}/{split}] reranker scoring {Q * M:,} pairs...")
    t0 = time.time()
    for i, qid in enumerate(tqdm(qids, desc=f"{name}/{split} rerank")):
        texts = [idx["doc_texts"][idx["doc_ids"][int(j)]] for j in cand_arr[i]]
        rerank_arr[i] = rerank_scores(ds.queries[qid], texts, batch_size=64)
    print(f"[{name}/{split}] reranker done in {time.time() - t0:.1f}s")
    np.savez(
        out_path, cand_idx=cand_arr, bm25_pool=bm25_arr,
        dense_pool=dense_arr, rerank=rerank_arr, q_ids=np.asarray(qids),
    )
    print(f"[{name}/{split}] saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--splits", nargs="+", default=["test"])
    args = p.parse_args()
    for ds in args.datasets:
        for sp in args.splits:
            precompute(ds, sp)


if __name__ == "__main__":
    main()
