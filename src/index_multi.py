"""Build BM25 + dense indices for all BEIR datasets we have on disk.

Saves per-dataset under indices/<name>/:
  - doc_ids.json
  - doc_texts.json
  - embeddings.npy           (N, D), L2-normalized
  - bm25.pkl
  - q_embeddings_<split>.npy
  - q_ids_<split>.json
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def doc_text(d: dict) -> str:
    return (d["title"] + ". " + d["text"]).strip()


def build_one(name: str, model: SentenceTransformer | None = None) -> None:
    out_dir = INDEX_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(name)
    print(f"[{name}] {ds.docs():,} docs, splits={list(ds.qrels.keys())}")

    doc_ids = sorted(ds.corpus.keys())
    doc_texts = {d: doc_text(ds.corpus[d]) for d in doc_ids}
    (out_dir / "doc_ids.json").write_text(json.dumps(doc_ids))
    (out_dir / "doc_texts.json").write_text(json.dumps(doc_texts))

    # BM25
    bm25_path = out_dir / "bm25.pkl"
    if not bm25_path.exists():
        print(f"[{name}] tokenizing for BM25...")
        tok = [tokenize(doc_texts[d]) for d in tqdm(doc_ids, desc=f"{name}/bm25")]
        bm25 = BM25Okapi(tok)
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "tokenized": tok}, f)
    else:
        print(f"[{name}] BM25 already built")

    # Dense
    emb_path = out_dir / "embeddings.npy"
    if not emb_path.exists():
        if model is None:
            model = SentenceTransformer(DENSE_MODEL)
        texts = [doc_texts[d] for d in doc_ids]
        emb = model.encode(
            texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        np.save(emb_path, emb)
        print(f"[{name}] embeddings shape={emb.shape}")
    else:
        print(f"[{name}] embeddings already exist")

    # Queries - encode train + dev + test, restricted to qids with qrels
    for split in ("train", "dev", "test"):
        if split not in ds.qrels:
            continue
        qids = sorted(ds.qrels[split].keys())
        # restrict to qids that actually exist in queries (some may have no entry)
        qids = [q for q in qids if q in ds.queries]
        q_path = out_dir / f"q_embeddings_{split}.npy"
        if q_path.exists():
            print(f"[{name}] {split} queries already encoded")
            continue
        texts = [ds.queries[q] for q in qids]
        if model is None:
            model = SentenceTransformer(DENSE_MODEL)
        emb = model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        np.save(q_path, emb)
        (out_dir / f"q_ids_{split}.json").write_text(json.dumps(qids))
        print(f"[{name}] {split} queries: {emb.shape}")
    return model


def main(names: list[str]):
    model = None
    for n in names:
        model = build_one(n, model=model)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Dataset names: scifact nfcorpus trec-covid fiqa")
    args = p.parse_args()
    main(args.datasets)
