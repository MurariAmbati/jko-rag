"""Show concrete query examples to illustrate qualitative differences between methods.

Picks queries where Wasserstein-prox notably beats KL-prox, and prints:
- the query
- the gold relevant doc(s)
- top-5 retrieved by each method
- which positions contain gold docs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from download_data import load_scifact

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main(n_show: int = 5):
    s1 = json.loads((RESULTS_DIR / "stage1.json").read_text())
    retrieved = json.loads((RESULTS_DIR / "retrieved.json").read_text())
    _, queries, qrels_test, _ = load_scifact()

    # find q_ids in order
    # per_query metric arrays match enumeration order through METHOD_NAMES;
    # retrieved.json maps qid -> doc ids per method
    qids = list(retrieved["rerank"].keys())
    a = np.asarray(s1["per_query"]["jko_rerank"]["ndcg@10"])
    b = np.asarray(s1["per_query"]["kl_rerank"]["ndcg@10"])
    diff = a - b
    pos_idx = np.argsort(-diff)[:n_show]
    neg_idx = np.argsort(diff)[:3]

    def show_one(i, label):
        qid = qids[i]
        print(f"\n[{label}] qid={qid}: {queries[qid]}")
        rel = {d for d, r in qrels_test.get(qid, {}).items() if r > 0}
        print(f"  Relevant: {sorted(rel)}")
        for m in ["rerank", "mmr", "kl_rerank", "jko_rerank", "jko_blend"]:
            top5 = retrieved[m][qid][:5]
            marks = ["GOLD" if d in rel else "" for d in top5]
            line = ", ".join(f"{d}{'*' if d in rel else ''}" for d in top5)
            print(f"  {m:<14s}: {line}")

    print("=" * 72)
    print("EXAMPLES: queries where Wasserstein > KL on nDCG@10")
    print("=" * 72)
    for i in pos_idx:
        show_one(int(i), f"W beats KL by {diff[i]:+.3f}")

    print("\n" + "=" * 72)
    print("EXAMPLES: queries where Wasserstein < KL on nDCG@10")
    print("=" * 72)
    for i in neg_idx:
        show_one(int(i), f"W trails KL by {diff[i]:+.3f}")


if __name__ == "__main__":
    main()
