"""Re-encode SciFact corpus with the fine-tuned dense retriever and rerun Stage 1.

Outputs go to indices/scifact_ft/ to keep separate from the pretrained run.
"""
from __future__ import annotations

import json
import pickle
import re
import shutil
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset
from methods import rerank_scores

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "scifact_minilm_ft"
FT_DIR = INDEX_ROOT / "scifact_ft"
TOKEN_RE = re.compile(r"\w+")


def tokenize(t): return TOKEN_RE.findall(t.lower())


def hybrid_pool(idx_emb, idx_bm25, doc_ids, q_text, q_emb, pool_size=200, each_n=500):
    bm25_all = idx_bm25.get_scores(tokenize(q_text)).astype(np.float32)
    dense_all = (idx_emb @ q_emb).astype(np.float32)
    en = min(each_n, len(bm25_all) - 1)
    bm25_top = np.argpartition(-bm25_all, en)[:en]; bm25_top = bm25_top[np.argsort(-bm25_all[bm25_top])]
    dense_top = np.argpartition(-dense_all, en)[:en]; dense_top = dense_top[np.argsort(-dense_all[dense_top])]
    fused = {}; K = 60
    for r, di in enumerate(bm25_top.tolist()): fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    for r, di in enumerate(dense_top.tolist()): fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    items = sorted(fused.items(), key=lambda kv: -kv[1])[:pool_size]
    return np.array([i for i, _ in items], dtype=np.int64), bm25_all, dense_all


def main():
    print(f"Loading fine-tuned model from {MODEL_DIR}")
    model = SentenceTransformer(str(MODEL_DIR))

    print("Loading SciFact corpus...")
    ds = load_dataset("scifact")
    doc_ids = sorted(ds.corpus.keys())
    doc_texts = {d: (ds.corpus[d]["title"] + ". " + ds.corpus[d]["text"]).strip() for d in doc_ids}

    FT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Encode corpus
    emb_path = FT_DIR / "embeddings.npy"
    if not emb_path.exists():
        texts = [doc_texts[d] for d in doc_ids]
        print(f"Encoding {len(texts)} docs with FT model...")
        emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        np.save(emb_path, emb)
        print(f"Saved {emb_path} shape={emb.shape}")
    else:
        emb = np.load(emb_path)
        print(f"Loaded existing FT embeddings shape={emb.shape}")

    # 2. Copy BM25 + doc_ids (unchanged)
    for fname in ("doc_ids.json", "doc_texts.json", "bm25.pkl"):
        src = INDEX_ROOT / fname
        dst = FT_DIR / fname
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)

    with open(FT_DIR / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    # 3. Encode test queries
    print("Encoding test queries with FT model...")
    test_qrels = ds.qrels["test"]
    test_qids = sorted(test_qrels.keys(), key=lambda x: int(x) if x.isdigit() else x)
    test_qids = [q for q in test_qids if q in ds.queries]
    q_emb = model.encode([ds.queries[q] for q in test_qids],
                          batch_size=64, normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float32)
    np.save(FT_DIR / "q_embeddings_test.npy", q_emb)
    (FT_DIR / "q_ids_test.json").write_text(json.dumps(test_qids))
    print(f"Saved {len(test_qids)} test query embeddings")

    # 4. Build new candidate pools + rerank
    cache_path = FT_DIR / "candidates_test.npz"
    if not cache_path.exists():
        M = 200
        cand_arr = np.zeros((len(test_qids), M), dtype=np.int64)
        bm25_arr = np.zeros((len(test_qids), M), dtype=np.float32)
        dense_arr = np.zeros((len(test_qids), M), dtype=np.float32)
        rerank_arr = np.zeros((len(test_qids), M), dtype=np.float32)
        print(f"Building candidate pools and reranker scores for {len(test_qids)} queries...")
        for i, qid in enumerate(tqdm(test_qids, desc="cands+rerank")):
            cand, bm25_all, dense_all = hybrid_pool(
                emb, bm25_data["bm25"], doc_ids, ds.queries[qid], q_emb[i], pool_size=M,
            )
            cand_arr[i] = cand
            bm25_arr[i] = bm25_all[cand]
            dense_arr[i] = dense_all[cand]
            texts = [doc_texts[doc_ids[int(j)]] for j in cand]
            rerank_arr[i] = rerank_scores(ds.queries[qid], texts, batch_size=64)
        np.savez(cache_path, cand_idx=cand_arr, bm25_pool=bm25_arr,
                 dense_pool=dense_arr, rerank=rerank_arr,
                 q_ids=np.asarray(test_qids))
        print(f"Saved {cache_path}")
    else:
        print(f"Cache already exists at {cache_path}")


if __name__ == "__main__":
    main()
