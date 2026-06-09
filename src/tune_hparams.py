"""Hyperparameter random search for JKO-RAG on SciFact training split.

We tune only on TRAIN queries to keep TEST blinded.

Searches over:
  h, lambda, rho, sinkhorn_eps, T, inner_steps, tau0, energy_blend

The energy blend is parameterized as (alpha_dense, gamma_rerank) with beta=0
(BM25 already in the candidate pool).

Reports best config by nDCG@10 on a held-out validation slice of the train set.

Outputs results/best_hparams.json with both the chosen config and the random
search history.
"""
from __future__ import annotations

import json
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset
from retrieval import (
    cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax,
)
from methods import Candidates, rerank_scores
from jko import JKOConfig, run_jko
from evaluation import ndcg_at_k, recall_at_k

INDEX_ROOT = Path(__file__).resolve().parents[1] / "indices"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def index_dir(name: str) -> Path:
    """Resolve where indices for a dataset live. Falls back to flat root for legacy scifact."""
    sub = INDEX_ROOT / name
    if (sub / "doc_ids.json").exists():
        return sub
    return INDEX_ROOT


def load_index(name: str):
    base = index_dir(name)
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
        "doc_id_to_idx": {d: i for i, d in enumerate(doc_ids)},
    }


TOKEN_RE = re.compile(r"\w+")


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(t.lower())


def hybrid_pool(idx, query, q_emb, pool_size=200, each_n=500):
    bm25_all = idx["bm25"].get_scores(tokenize(query)).astype(np.float32)
    dense_all = (idx["embeddings"] @ q_emb).astype(np.float32)
    bm25_top = np.argpartition(-bm25_all, min(each_n, len(bm25_all) - 1))[:each_n]
    bm25_top = bm25_top[np.argsort(-bm25_all[bm25_top])]
    dense_top = np.argpartition(-dense_all, min(each_n, len(dense_all) - 1))[:each_n]
    dense_top = dense_top[np.argsort(-dense_all[dense_top])]
    fused = {}
    K = 60
    for r, di in enumerate(bm25_top.tolist()):
        fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    for r, di in enumerate(dense_top.tolist()):
        fused[di] = fused.get(di, 0.0) + 1.0 / (K + r + 1)
    items = sorted(fused.items(), key=lambda kv: -kv[1])[:pool_size]
    cand = np.array([i for i, _ in items], dtype=np.int64)
    return cand, bm25_all[cand], dense_all[cand]


def make_energy(c: Candidates, alpha: float, gamma: float) -> np.ndarray:
    r = alpha * normalize_minmax(c.dense_scores) + gamma * normalize_minmax(c.rerank_scores)
    return -r


def sample_config(rng: random.Random) -> dict:
    return {
        "h": rng.choice([0.1, 0.2, 0.5, 1.0, 2.0]),
        "lam": rng.choice([0.005, 0.01, 0.03, 0.05, 0.1]),
        "rho": rng.choice([0.0, 0.01, 0.05, 0.1, 0.2]),
        "sinkhorn_eps": rng.choice([0.05, 0.1, 0.2]),
        "T": rng.choice([1, 2, 3, 5]),
        "inner_steps": rng.choice([15, 25, 40]),
        "tau0": rng.choice([0.05, 0.1, 0.3, 1.0]),
        "alpha": rng.choice([0.0, 0.2, 0.4, 0.7, 1.0]),  # dense weight
        "gamma": rng.choice([0.0, 0.3, 0.6, 1.0]),       # rerank weight
        "mode": "wasserstein",
    }


def eval_config(
    cfg: dict,
    cands_by_qid: dict[str, Candidates],
    embeddings: np.ndarray,
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict[str, float]:
    """Run JKO-RAG with given config on each query, return mean nDCG@10 and Recall@10."""
    if cfg["alpha"] == 0 and cfg["gamma"] == 0:
        return {"ndcg@10": 0.0, "recall@10": 0.0, "n": 0}
    ndcg_scores = []
    rec_scores = []
    jcfg = JKOConfig(
        h=cfg["h"], lam=cfg["lam"], rho=cfg["rho"],
        sinkhorn_eps=cfg["sinkhorn_eps"], T=cfg["T"],
        inner_steps=cfg["inner_steps"], mode=cfg["mode"],
    )
    for qid, c in cands_by_qid.items():
        if qid not in qrels:
            continue
        Z = embeddings[c.cand_idx]
        C = cost_matrix_cosine(Z)
        K = redundancy_kernel(Z)
        energy = make_energy(c, cfg["alpha"], cfg["gamma"])
        p0 = softmax_np(-energy, tau=cfg["tau0"])
        p_T, _ = run_jko(p0, energy, C, K, jcfg)
        order = np.argsort(-p_T)[:k]
        dids = [doc_ids[i] for i in c.cand_idx[order]]
        ndcg_scores.append(ndcg_at_k(dids, qrels[qid], k))
        rec_scores.append(recall_at_k(dids, qrels[qid], k))
    return {
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "recall@10": float(np.mean(rec_scores)) if rec_scores else 0.0,
        "n": len(ndcg_scores),
    }


def build_train_candidates(
    dataset_name: str, split: str = "train", n_queries: int = 100, seed: int = 0,
):
    """Build (or load cached) candidates with reranker scores for a slice of train queries."""
    cache_path = index_dir(dataset_name) / f"candidates_{split}_n{n_queries}.npz"
    ds = load_dataset(dataset_name)
    idx = load_index(dataset_name)
    qids_with_qrels = list(ds.qrels[split].keys())
    qids_with_qrels = [q for q in qids_with_qrels if q in ds.queries]
    rng = random.Random(seed)
    rng.shuffle(qids_with_qrels)
    qids = qids_with_qrels[:n_queries]

    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        cached_qids = [str(x) for x in d["qids"]]
        if cached_qids == qids:
            print(f"[{dataset_name}/{split}] loading cached candidates ({n_queries} queries)")
            cands = {}
            for i, q in enumerate(cached_qids):
                cands[q] = Candidates(
                    cand_idx=d["cand_idx"][i], bm25_scores=d["bm25"][i],
                    dense_scores=d["dense"][i], rerank_scores=d["rerank"][i],
                )
            return cands, qids, idx, ds

    # encode queries fresh
    from sentence_transformers import SentenceTransformer
    print(f"[{dataset_name}/{split}] encoding {len(qids)} queries...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([ds.queries[q] for q in qids], normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
    M = 200
    cand_arr = np.zeros((len(qids), M), dtype=np.int64)
    bm25_arr = np.zeros((len(qids), M), dtype=np.float32)
    dense_arr = np.zeros((len(qids), M), dtype=np.float32)
    rerank_arr = np.zeros((len(qids), M), dtype=np.float32)
    print(f"[{dataset_name}/{split}] building candidate pools + reranker scores...")
    for i, q in enumerate(tqdm(qids, desc="cands")):
        cand, b, d = hybrid_pool(idx, ds.queries[q], q_emb[i], pool_size=M)
        cand_arr[i] = cand; bm25_arr[i] = b; dense_arr[i] = d
        texts = [idx["doc_texts"][idx["doc_ids"][int(j)]] for j in cand]
        rerank_arr[i] = rerank_scores(ds.queries[q], texts, batch_size=64)
    np.savez(
        cache_path,
        cand_idx=cand_arr, bm25=bm25_arr, dense=dense_arr, rerank=rerank_arr,
        qids=np.asarray(qids),
    )
    cands = {q: Candidates(cand_idx=cand_arr[i], bm25_scores=bm25_arr[i],
                            dense_scores=dense_arr[i], rerank_scores=rerank_arr[i])
             for i, q in enumerate(qids)}
    return cands, qids, idx, ds


def main(dataset_name: str = "scifact", n_train: int = 80, n_iter: int = 30):
    print(f"Tuning on {dataset_name} train (n={n_train}) for {n_iter} random configs")
    cands, qids, idx, ds = build_train_candidates(dataset_name, split="train",
                                                   n_queries=n_train, seed=0)
    qrels = ds.qrels["train"]
    embeddings = idx["embeddings"]
    doc_ids = idx["doc_ids"]

    rng = random.Random(42)
    history = []
    best = None
    t0 = time.time()
    for i in range(n_iter):
        cfg = sample_config(rng)
        st = time.time()
        res = eval_config(cfg, cands, embeddings, doc_ids, qrels)
        elapsed = time.time() - st
        entry = {"cfg": cfg, **res, "elapsed_s": elapsed}
        history.append(entry)
        marker = ""
        if best is None or res["ndcg@10"] > best["ndcg@10"]:
            best = entry; marker = " <-- new best"
        print(f"  [{i+1}/{n_iter}] ndcg@10={res['ndcg@10']:.4f} recall@10={res['recall@10']:.4f}"
              f" ({elapsed:.1f}s){marker}  cfg={cfg}")
    print(f"\nSearch took {time.time() - t0:.1f}s")
    print("\nBest config:")
    print(json.dumps(best, indent=2))

    out_path = RESULTS_DIR / f"best_hparams_{dataset_name}.json"
    out_path.write_text(json.dumps({"best": best, "history": history}, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-iter", type=int, default=30)
    args = p.parse_args()
    main(args.dataset, args.n_train, args.n_iter)
