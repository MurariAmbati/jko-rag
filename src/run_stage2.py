"""Stage 2 — answer generation with FLAN-T5-small.

For each labeled SciFact train claim (excluding the 80 used in hyperparameter
tuning), we:
  1. Retrieve top-k evidence with each method (rerank, kl_blend, jko_blend).
  2. Build a prompt: "Claim: ... Evidence: ... Is the claim SUPPORTED,
     CONTRADICTED, or is there NOT ENOUGH INFO?"
  3. Generate a single-token answer with FLAN-T5-small.
  4. Compare to the gold label.

Reports 3-way label accuracy + per-class F1.
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
from methods import Candidates, rerank_scores
from retrieval import Indices, cost_matrix_cosine, redundancy_kernel, softmax_np, normalize_minmax
from jko import JKOConfig, run_jko
from evaluation import bootstrap_ci, paired_bootstrap_diff

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
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


def load_scifact_train_labels():
    """Returns {claim_id (str): {'claim': str, 'gold_label': str, 'cited_dids': list[str]}}.

    Label is one of 'SUPPORT' / 'CONTRADICT' / 'NEI'.
    """
    p = DATA_DIR / "scifact_orig" / "data" / "claims_train.jsonl"
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cid = str(d["id"])
            # Aggregate label across evidence
            labels = set()
            if d.get("evidence"):
                for did, evs in d["evidence"].items():
                    for e in evs:
                        labels.add(e["label"])
            if not labels:
                gold = "NEI"
            elif "CONTRADICT" in labels and "SUPPORT" not in labels:
                gold = "CONTRADICT"
            elif "SUPPORT" in labels and "CONTRADICT" not in labels:
                gold = "SUPPORT"
            else:
                gold = list(labels)[0]
            out[cid] = {
                "claim": d["claim"],
                "gold_label": gold,
                "cited_dids": [str(x) for x in d.get("cited_doc_ids", [])],
            }
    return out


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


def jko_topk(c, idx, mode, alpha, gamma, k, cfg):
    Z = idx.embeddings[c.cand_idx]
    C = cost_matrix_cosine(Z); K = redundancy_kernel(Z)
    energy = -(alpha * normalize_minmax(c.dense_scores) + gamma * normalize_minmax(c.rerank_scores))
    p0 = softmax_np(-energy, tau=cfg["tau0"])
    jcfg = JKOConfig(
        h=cfg["h"], lam=cfg["lam"], rho=cfg["rho"],
        sinkhorn_eps=cfg["sinkhorn_eps"], T=cfg["T"],
        inner_steps=cfg["inner_steps"], mode=mode,
    )
    p_T, _ = run_jko(p0, energy, C, K, jcfg)
    return c.cand_idx[np.argsort(-p_T)[:k]].tolist()


def rerank_topk(c, k): return c.cand_idx[np.argsort(-c.rerank_scores)[:k]].tolist()


def parse_label(text: str) -> str:
    t = text.strip().upper()
    # YES/NO/MAYBE verbalization
    if t.startswith("YES") or "TRUE" in t.split()[0:1]:
        return "SUPPORT"
    if t.startswith("NO") and not t.startswith("NOT"):
        return "CONTRADICT"
    if "FALSE" in t.split()[0:1]:
        return "CONTRADICT"
    if t.startswith("MAYBE") or t.startswith("UNCLEAR"):
        return "NEI"
    # also accept verbose answers
    if "SUPPORT" in t: return "SUPPORT"
    if "CONTRADICT" in t or "REFUT" in t or "AGAINST" in t: return "CONTRADICT"
    if "NEI" in t or "NOT ENOUGH" in t or "INSUFFICIENT" in t: return "NEI"
    return "NEI"


def build_prompt(claim: str, evidence_texts: list[str]) -> str:
    ev = "\n\n".join(f"Evidence {i+1}: {e[:600]}" for i, e in enumerate(evidence_texts))
    return (
        f"Answer the question based on the evidence below.\n\n"
        f"{ev}\n\n"
        f"Question: Given the evidence above, is the following claim true, false, or "
        f"is there not enough information to tell?\n"
        f"Claim: {claim}\n"
        f"Answer with one word — YES if true, NO if false, MAYBE if not enough info."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--config-file", default=None)
    p.add_argument("--k", type=int, default=3, help="top-k evidence to use")
    args = p.parse_args()

    print("Loading SciFact labels and indices...")
    label_map = load_scifact_train_labels()  # 809 claims
    ds = load_dataset("scifact")
    idx = load_index("scifact")
    queries = ds.queries  # BEIR queries; ids match SciFact

    # Identify the 80 tuning queries to exclude
    tuning_cache = index_dir("scifact") / "candidates_train_n80.npz"
    excluded = set()
    if tuning_cache.exists():
        cache = np.load(tuning_cache, allow_pickle=True)
        excluded = {str(x) for x in cache["qids"]}
        print(f"Excluding {len(excluded)} tuning queries")

    # Sample claims with labels (any of SUPPORT/CONTRADICT/NEI)
    eligible = [cid for cid in label_map if cid not in excluded and cid in queries]
    print(f"Eligible non-tuning claims: {len(eligible)}")
    rng = np.random.default_rng(42)
    if len(eligible) > args.n_queries:
        sel = rng.choice(len(eligible), args.n_queries, replace=False)
        chosen = [eligible[i] for i in sorted(sel)]
    else:
        chosen = eligible

    # Show label distribution
    from collections import Counter
    label_dist = Counter(label_map[c]["gold_label"] for c in chosen)
    print(f"Selected {len(chosen)} claims, labels: {dict(label_dist)}")

    # Load FLAN-T5
    print(f"Loading FLAN-T5-base...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    mname = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(mname)
    lm = AutoModelForSeq2SeqLM.from_pretrained(mname)
    lm.eval()

    # Load tuned config
    cfg = {"h": 0.5, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
           "T": 3, "inner_steps": 25, "tau0": 0.1}
    if args.config_file:
        c = json.loads(Path(args.config_file).read_text())
        cfg.update(c.get("best", {}).get("cfg", c))
    print(f"Using config: {cfg}")

    # Encode test claims for dense retrieval
    print("Encoding claims...")
    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    claim_texts = [queries[cid] for cid in chosen]
    q_embs = dense_model.encode(claim_texts, batch_size=64, normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)

    # Methods to evaluate
    methods = ["rerank", "kl_blend", "jko_blend"]
    predictions = {m: {} for m in methods}
    retrieved_evidence = {m: {} for m in methods}

    print("Building candidates + generating predictions...")
    for i, cid in enumerate(tqdm(chosen, desc="claims")):
        c_text = queries[cid]
        q_emb = q_embs[i]
        cand, b, d_s = hybrid_pool(idx, c_text, q_emb, pool_size=200)
        texts_for_rerank = [idx.doc_texts[idx.doc_ids[int(j)]] for j in cand]
        rr = rerank_scores(c_text, texts_for_rerank, batch_size=64)
        c = Candidates(cand_idx=cand, bm25_scores=b, dense_scores=d_s, rerank_scores=rr)

        # Retrieve top-k for each method
        method_tops = {
            "rerank":    rerank_topk(c, args.k),
            "kl_blend":  jko_topk(c, idx, "kl",          0.4, 0.6, args.k, cfg),
            "jko_blend": jko_topk(c, idx, "wasserstein", 0.4, 0.6, args.k, cfg),
        }
        for m, tops in method_tops.items():
            ev_texts = [idx.doc_texts[idx.doc_ids[j]] for j in tops]
            retrieved_evidence[m][cid] = [idx.doc_ids[j] for j in tops]
            prompt = build_prompt(c_text, ev_texts)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                out = lm.generate(**inputs, max_new_tokens=8, do_sample=False)
            text = tok.decode(out[0], skip_special_tokens=True)
            predictions[m][cid] = {"raw": text, "label": parse_label(text)}

    # Evaluate
    results = {}
    for m in methods:
        correct = []
        per_class_correct = defaultdict(list)
        confusion = defaultdict(lambda: defaultdict(int))
        for cid in chosen:
            gold = label_map[cid]["gold_label"]
            pred = predictions[m][cid]["label"]
            correct.append(1 if pred == gold else 0)
            per_class_correct[gold].append(1 if pred == gold else 0)
            confusion[gold][pred] += 1
        acc_mean, acc_lo, acc_hi = bootstrap_ci(correct)
        results[m] = {
            "accuracy": {"mean": acc_mean, "ci_lo": acc_lo, "ci_hi": acc_hi, "n": len(correct)},
            "per_class": {c: {"acc": float(np.mean(v)), "n": len(v)} for c, v in per_class_correct.items()},
            "confusion": {g: dict(p) for g, p in confusion.items()},
            "per_query_correct": correct,
        }

    # Paired diffs
    paired = {}
    for a, b in [("jko_blend", "kl_blend"), ("jko_blend", "rerank"), ("kl_blend", "rerank")]:
        diff, lo, hi = paired_bootstrap_diff(results[a]["per_query_correct"], results[b]["per_query_correct"])
        paired[f"{a}_vs_{b}"] = {"diff": diff, "ci_lo": lo, "ci_hi": hi}

    out = {
        "n_queries": len(chosen),
        "k_evidence": args.k,
        "config": cfg,
        "label_distribution": dict(label_dist),
        "results": results,
        "paired": paired,
        "predictions": {m: {cid: predictions[m][cid] for cid in chosen} for m in methods},
        "retrieved_evidence": retrieved_evidence,
    }
    (RESULTS_DIR / "stage2_scifact.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {RESULTS_DIR / 'stage2_scifact.json'}")

    print(f"\n=== Stage 2 label accuracy (n={len(chosen)}, FLAN-T5-small, top-{args.k} evidence) ===")
    for m in methods:
        a = results[m]["accuracy"]
        pc = results[m]["per_class"]
        per_class_str = ", ".join(f"{c}={v['acc']:.3f}(n={v['n']})" for c, v in pc.items())
        print(f"  {m:<12s}  acc={a['mean']:.4f} [{a['ci_lo']:.4f}, {a['ci_hi']:.4f}]  per-class: {per_class_str}")

    print("\nPaired bootstrap on accuracy:")
    for k, p in paired.items():
        sig = " *" if (p["ci_lo"] > 0 or p["ci_hi"] < 0) else "  "
        print(f"  {k:<28s}  diff={p['diff']:+.4f}  [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]{sig}")


if __name__ == "__main__":
    main()
