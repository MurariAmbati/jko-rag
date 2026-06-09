"""Upgrade retriever + reranker pipeline.

Replaces:
  dense:   all-MiniLM-L6-v2     -> BAAI/bge-small-en-v1.5
  rerank:  ms-marco-MiniLM-L-6  -> BAAI/bge-reranker-base

Both BGE models are 2023-era, trained on much larger and cleaner contrastive
datasets. Same parameter scale class but substantially stronger.

Saves to indices_bge/ — fully separate from the MiniLM indices so we can
compare side-by-side without overwriting prior results.

Usage:
  python upgrade_pipeline.py --datasets scifact nfcorpus fiqa scidocs trec-covid \
                              --steps encode candidates
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset

INDEX_ROOT_OLD = Path(__file__).resolve().parents[1] / "indices"
INDEX_ROOT_NEW = Path(__file__).resolve().parents[1] / "indices_bge"
INDEX_ROOT_NEW.mkdir(exist_ok=True)

DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # 33M params, 384-d
RERANK_MODEL = "BAAI/bge-reranker-base"  # 110M params

TOKEN_RE = re.compile(r"\w+")


def tokenize(t): return TOKEN_RE.findall(t.lower())


def doc_text(d):
    return (d.get("title", "") + ". " + d.get("text", "")).strip()


# -----------------------------------------------------------------------------
# Encoders
# -----------------------------------------------------------------------------
_DENSE = None


def get_dense():
    global _DENSE
    if _DENSE is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading dense {DENSE_MODEL}...")
        _DENSE = SentenceTransformer(DENSE_MODEL)
    return _DENSE


_RERANKER = None


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker {RERANK_MODEL}...")
        _RERANKER = CrossEncoder(RERANK_MODEL, max_length=512)
    return _RERANKER


# -----------------------------------------------------------------------------
# Encode + index per dataset
# -----------------------------------------------------------------------------
def encode_dataset(name):
    out = INDEX_ROOT_NEW / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Encoding {name} ===")
    ds = load_dataset(name)
    doc_ids = sorted(ds.corpus.keys())
    doc_texts = {d: doc_text(ds.corpus[d]) for d in doc_ids}

    # save / reuse doc metadata
    (out / "doc_ids.json").write_text(json.dumps(doc_ids))
    (out / "doc_texts.json").write_text(json.dumps(doc_texts))

    # BM25 (reuse from old indices if same corpus)
    bm25_dst = out / "bm25.pkl"
    if not bm25_dst.exists():
        # Try to copy from old indices (same corpus)
        old_sub = INDEX_ROOT_OLD / name / "bm25.pkl"
        old_flat = INDEX_ROOT_OLD / "bm25.pkl" if name == "scifact" else None
        src = old_sub if old_sub.exists() else (old_flat if (old_flat and old_flat.exists()) else None)
        if src and src.exists():
            shutil.copy2(src, bm25_dst)
            print(f"[{name}] copied BM25 from {src}")
        else:
            from rank_bm25 import BM25Okapi
            print(f"[{name}] building BM25 from scratch...")
            tok = [tokenize(doc_texts[d]) for d in tqdm(doc_ids, desc="bm25 tokenize")]
            bm25 = BM25Okapi(tok)
            with open(bm25_dst, "wb") as f:
                pickle.dump({"bm25": bm25, "tokenized": tok}, f)

    # Corpus embeddings with BGE
    emb_path = out / "embeddings.npy"
    if not emb_path.exists():
        model = get_dense()
        texts = [doc_texts[d] for d in doc_ids]
        # BGE-small uses query/passage instruction prefixes; for passages we use the raw text
        print(f"[{name}] encoding {len(texts):,} docs...")
        emb = model.encode(
            texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        np.save(emb_path, emb)
        print(f"[{name}] embeddings {emb.shape}")
    else:
        print(f"[{name}] embeddings exist")

    # Query embeddings per split (with the bge query instruction)
    for split in ("train", "dev", "test"):
        if split not in ds.qrels:
            continue
        q_emb_path = out / f"q_embeddings_{split}.npy"
        q_ids_path = out / f"q_ids_{split}.json"
        if q_emb_path.exists():
            continue
        qids = sorted(ds.qrels[split].keys())
        qids = [q for q in qids if q in ds.queries]
        # BGE-small-en-v1.5 query prefix
        prefix = "Represent this sentence for searching relevant passages: "
        texts = [prefix + ds.queries[q] for q in qids]
        model = get_dense()
        emb = model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        np.save(q_emb_path, emb)
        q_ids_path.write_text(json.dumps(qids))
        print(f"[{name}] {split} queries: {emb.shape}")


# -----------------------------------------------------------------------------
# Build candidate pools + reranker scores
# -----------------------------------------------------------------------------
def hybrid_pool(idx_emb, idx_bm25, q_text, q_emb, pool_size=200, each_n=500):
    bm25_all = idx_bm25.get_scores(tokenize(q_text)).astype(np.float32)
    dense_all = (idx_emb @ q_emb).astype(np.float32)
    en = min(each_n, len(bm25_all) - 1)
    bm25_top = np.argpartition(-bm25_all, en)[:en]; bm25_top = bm25_top[np.argsort(-bm25_all[bm25_top])]
    dense_top = np.argpartition(-dense_all, en)[:en]; dense_top = dense_top[np.argsort(-dense_all[dense_top])]
    fused = {}; K = 60
    for r, di in enumerate(bm25_top.tolist()): fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    for r, di in enumerate(dense_top.tolist()): fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    items = sorted(fused.items(), key=lambda kv: -kv[1])[:pool_size]
    cand = np.array([i for i, _ in items], dtype=np.int64)
    return cand, bm25_all[cand], dense_all[cand]


def build_candidates(name, split="test"):
    out = INDEX_ROOT_NEW / name
    cache_path = out / f"candidates_{split}.npz"
    if cache_path.exists():
        print(f"[{name}/{split}] cache exists")
        return

    ds = load_dataset(name)
    if split not in ds.qrels:
        print(f"[{name}/{split}] no qrels")
        return

    with open(out / "doc_ids.json") as f: doc_ids = json.load(f)
    with open(out / "doc_texts.json") as f: doc_texts = json.load(f)
    embeddings = np.load(out / "embeddings.npy")
    with open(out / "bm25.pkl", "rb") as f: bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    qids = json.load(open(out / f"q_ids_{split}.json"))
    q_emb = np.load(out / f"q_embeddings_{split}.npy")

    M = 200
    Q = len(qids)
    cand_arr = np.zeros((Q, M), dtype=np.int64)
    bm25_arr = np.zeros((Q, M), dtype=np.float32)
    dense_arr = np.zeros((Q, M), dtype=np.float32)
    rerank_arr = np.zeros((Q, M), dtype=np.float32)

    print(f"\n=== [{name}/{split}] candidate pools + BGE reranker ({Q} queries) ===")
    for i, qid in enumerate(tqdm(qids, desc=f"{name}/{split} pools")):
        cand, b, d = hybrid_pool(embeddings, bm25, ds.queries[qid], q_emb[i], pool_size=M)
        cand_arr[i] = cand; bm25_arr[i] = b; dense_arr[i] = d

    rr = get_reranker()
    print(f"[{name}/{split}] scoring {Q * M:,} pairs with BGE-reranker-base...")
    t0 = time.time()
    for i, qid in enumerate(tqdm(qids, desc=f"{name}/{split} rerank")):
        pairs = [(ds.queries[qid], doc_texts[doc_ids[int(j)]]) for j in cand_arr[i]]
        scores = rr.predict(pairs, batch_size=64, show_progress_bar=False)
        rerank_arr[i] = np.asarray(scores, dtype=np.float32)
    print(f"[{name}/{split}] reranker done in {time.time() - t0:.1f}s")

    np.savez(cache_path, cand_idx=cand_arr, bm25_pool=bm25_arr,
             dense_pool=dense_arr, rerank=rerank_arr, q_ids=np.asarray(qids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--steps", nargs="+", default=["encode", "candidates"],
                   choices=["encode", "candidates"])
    p.add_argument("--splits", nargs="+", default=["test"])
    args = p.parse_args()

    if "encode" in args.steps:
        for ds in args.datasets:
            encode_dataset(ds)
    if "candidates" in args.steps:
        for ds in args.datasets:
            for sp in args.splits:
                build_candidates(ds, sp)


if __name__ == "__main__":
    main()
