"""Aggregate per-dataset Stage 1 results into a single cross-dataset table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_results():
    out = {}
    for f in RESULTS_DIR.glob("stage1_*.json"):
        name = f.stem.replace("stage1_", "")
        out[name] = json.loads(f.read_text())
    # also legacy single-dataset
    legacy = RESULTS_DIR / "stage1.json"
    if legacy.exists() and "scifact" not in out:
        d = json.loads(legacy.read_text())
        out["scifact_default"] = d
    return out


def cell(d):
    return f"{d['mean']:.3f}[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"


def make_table(metric: str = "ndcg@10"):
    res = load_results()
    if not res:
        print("No results found")
        return
    # canonical method order
    methods = ["bm25", "dense", "hybrid_rrf", "rerank", "mmr",
               "noprox", "kl_prox", "jko_prox",
               "noprox_rerank", "kl_rerank", "jko_rerank",
               "jko_blend", "jko_blend_dense"]
    datasets = sorted(res.keys())
    print(f"\n{metric.upper()} across datasets (mean [95% CI])")
    print(f"{'method':<18s}  " + "  ".join(f"{d:<22s}" for d in datasets))
    for m in methods:
        row = [m.ljust(18)]
        present = False
        for d in datasets:
            s = res[d].get("summary", {}).get(m)
            if s and metric in s:
                row.append(cell(s[metric]).ljust(22))
                present = True
            else:
                row.append("—".ljust(22))
        if present:
            print("  ".join(row))


def best_per_dataset(metric: str = "ndcg@10"):
    res = load_results()
    print(f"\nBest method per dataset by {metric}")
    for d, r in res.items():
        s = r.get("summary", {})
        best = max(s.items(), key=lambda kv: kv[1].get(metric, {}).get("mean", -1))
        print(f"  {d:<22s} -> {best[0]} {cell(best[1][metric])}")


if __name__ == "__main__":
    for m in ("ndcg@10", "recall@10", "recall@20", "diversity@10"):
        make_table(m)
    best_per_dataset("ndcg@10")
