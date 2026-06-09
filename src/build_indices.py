"""Build BM25 and dense retrieval indices for SciFact.

Saves:
- indices/bm25.pkl       : pickled BM25Okapi + tokenized corpus
- indices/doc_ids.json   : ordered list of doc IDs
- indices/doc_texts.json : dict of doc_id -> "title. text" for cross-encoder reranking
- indices/embeddings.npy : (N, D) float32 dense embeddings
- indices/q_embeddings_test.npy + q_ids_test.json : test query embeddings
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
from download_data import load_scifact

INDEX_DIR = Path(__file__).resolve().parents[1] / "indices"
INDEX_DIR.mkdir(exist_ok=True)

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d, fast on CPU


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25(corpus: dict, doc_ids: list[str]):
    print("Tokenizing corpus for BM25...")
    tokenized = []
    for did in tqdm(doc_ids):
        d = corpus[did]
        text = (d["title"] + ". " + d["text"]).strip()
        tokenized.append(simple_tokenize(text))
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    print(f"Saved BM25 ({len(doc_ids)} docs)")


def build_dense(corpus: dict, doc_ids: list[str]):
    print(f"Loading dense model: {DENSE_MODEL}")
    model = SentenceTransformer(DENSE_MODEL)
    texts = [(corpus[did]["title"] + ". " + corpus[did]["text"]).strip() for did in doc_ids]
    print("Encoding corpus...")
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(INDEX_DIR / "embeddings.npy", emb)
    print(f"Saved corpus embeddings shape={emb.shape}")
    return model


def encode_queries(model, queries: dict, qids: list[str], suffix: str):
    print(f"Encoding {len(qids)} queries ({suffix})...")
    texts = [queries[q] for q in qids]
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(INDEX_DIR / f"q_embeddings_{suffix}.npy", emb)
    with open(INDEX_DIR / f"q_ids_{suffix}.json", "w") as f:
        json.dump(qids, f)
    print(f"Saved query embeddings ({suffix}) shape={emb.shape}")


def main():
    corpus, queries, qrels_test, qrels_train = load_scifact()
    doc_ids = sorted(corpus.keys())
    test_qids = sorted(qrels_test.keys(), key=lambda x: int(x) if x.isdigit() else x)
    train_qids = sorted(qrels_train.keys(), key=lambda x: int(x) if x.isdigit() else x)

    with open(INDEX_DIR / "doc_ids.json", "w") as f:
        json.dump(doc_ids, f)
    doc_texts = {did: (corpus[did]["title"] + ". " + corpus[did]["text"]).strip() for did in doc_ids}
    with open(INDEX_DIR / "doc_texts.json", "w") as f:
        json.dump(doc_texts, f)

    build_bm25(corpus, doc_ids)
    model = build_dense(corpus, doc_ids)
    encode_queries(model, queries, test_qids, "test")
    encode_queries(model, queries, train_qids, "train")
    print("Done.")


if __name__ == "__main__":
    main()
