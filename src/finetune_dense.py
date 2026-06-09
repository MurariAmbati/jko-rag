"""Fine-tune MiniLM-L6 dense retriever on SciFact training (claim, gold-doc) pairs.

Uses MultipleNegativesRankingLoss (in-batch negatives) — standard, well-tested
loss for retriever training (Reimers & Gurevych, 2019; Henderson et al. 2017).

Training pairs come from SciFact train qrels (the BEIR-flattened train.tsv).
We exclude the 80 queries used for hyperparameter tuning to keep the tuning
config evaluation honest. Eval uses BEIR's dev/test (300 queries) which never
appears in tuning or training.

Saves the fine-tuned model to models/scifact_minilm_ft/.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "scifact_minilm_ft"


def build_training_pairs(exclude_qids: set[str] | None = None):
    """Return list of (query_text, positive_doc_text) tuples from SciFact train."""
    ds = load_dataset("scifact")
    train_qrels = ds.qrels["train"]
    pairs = []
    for qid, dids in train_qrels.items():
        if exclude_qids and qid in exclude_qids: continue
        if qid not in ds.queries: continue
        q = ds.queries[qid]
        for did, rel in dids.items():
            if rel <= 0: continue
            if did not in ds.corpus: continue
            d = ds.corpus[did]
            t = (d.get("title", "") + ". " + d.get("text", "")).strip()
            # truncate long abstracts for stable training
            if len(t) > 1500: t = t[:1500]
            pairs.append((q, t))
    return pairs


def main():
    excluded = set()
    tune_cache = INDEX_ROOT / "candidates_train_n80.npz"
    if tune_cache.exists():
        c = np.load(tune_cache, allow_pickle=True)
        excluded = {str(x) for x in c["qids"]}
        print(f"Excluding {len(excluded)} tuning queries from training")

    pairs = build_training_pairs(excluded)
    random.seed(0); random.shuffle(pairs)
    print(f"Training pairs: {len(pairs)}")

    # Make sure we have at least some pairs
    if len(pairs) < 100:
        raise SystemExit(f"Too few training pairs ({len(pairs)})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {MODEL_DIR}")

    print("Loading base model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    train_examples = [InputExample(texts=[q, d]) for q, d in pairs]
    train_loader = DataLoader(train_examples, batch_size=32, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"Training: 3 epochs, batch_size=32, ~{len(train_loader)} steps/epoch")
    t0 = time.time()
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=3,
        warmup_steps=int(0.1 * 3 * len(train_loader)),
        output_path=str(MODEL_DIR),
        show_progress_bar=True,
    )
    print(f"Trained in {time.time() - t0:.1f}s. Saved to {MODEL_DIR}")

    # Quick sanity: encode a sample claim and doc
    sample = pairs[0]
    e = model.encode(list(sample), normalize_embeddings=True)
    sim = float(e[0] @ e[1])
    print(f"\nSanity: cos(claim, gold_doc) = {sim:.4f}")
    print(f"  Claim: {sample[0][:80]}...")
    print(f"  Doc:   {sample[1][:80]}...")


if __name__ == "__main__":
    main()
