"""Plot Stage 1 and Stage 3 results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

METHOD_ORDER = [
    "bm25", "dense", "hybrid_rrf", "rerank", "mmr",
    "noprox_rerank", "kl_rerank", "jko_rerank",
    "jko_blend", "jko_blend_dense",
]


def plot_bar_with_ci(ax, methods, summary, metric, color_map):
    means, los, his = [], [], []
    for m in methods:
        s = summary[m][metric]
        means.append(s["mean"])
        los.append(s["mean"] - s["ci_lo"])
        his.append(s["ci_hi"] - s["mean"])
    x = np.arange(len(methods))
    colors = [color_map[m] for m in methods]
    ax.bar(x, means, yerr=[los, his], color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax.set_title(metric)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def main():
    s1 = json.loads((RESULTS_DIR / "stage1.json").read_text())
    methods = [m for m in METHOD_ORDER if m in s1["summary"]]
    color_map = {}
    for m in methods:
        if m in ("bm25", "dense", "hybrid_rrf"):
            color_map[m] = "#888"
        elif m == "rerank":
            color_map[m] = "#2a9d8f"
        elif m == "mmr":
            color_map[m] = "#e9c46a"
        elif m == "noprox_rerank":
            color_map[m] = "#f4a261"
        elif m == "kl_rerank":
            color_map[m] = "#e76f51"
        elif m.startswith("jko"):
            color_map[m] = "#264653"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_bar_with_ci(axes[0, 0], methods, s1["summary"], "ndcg@10", color_map)
    plot_bar_with_ci(axes[0, 1], methods, s1["summary"], "recall@10", color_map)
    plot_bar_with_ci(axes[1, 0], methods, s1["summary"], "recall@20", color_map)
    plot_bar_with_ci(axes[1, 1], methods, s1["summary"], "diversity@10", color_map)
    fig.suptitle("JKO-RAG on SciFact (n=300 queries, 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stage1_bars.png", dpi=140)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'stage1_bars.png'}")

    # Paired diff JKO - KL scatter (per query nDCG)
    if "jko_rerank" in s1["per_query"] and "kl_rerank" in s1["per_query"]:
        a = np.asarray(s1["per_query"]["jko_rerank"]["ndcg@10"])
        b = np.asarray(s1["per_query"]["kl_rerank"]["ndcg@10"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(b, a, alpha=0.5, s=18)
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[0].set_xlabel("KL-prox nDCG@10")
        ax[0].set_ylabel("Wasserstein-prox nDCG@10")
        ax[0].set_title("Per-query nDCG@10: W vs KL")
        ax[0].grid(linestyle=":", alpha=0.5)
        diff = a - b
        ax[1].hist(diff, bins=40, color="#264653")
        ax[1].axvline(0, color="k", lw=0.8)
        ax[1].axvline(diff.mean(), color="r", lw=1, label=f"mean={diff.mean():+.4f}")
        ax[1].set_xlabel("nDCG@10 difference (W - KL)")
        ax[1].set_ylabel("queries")
        ax[1].set_title("Distribution of (W - KL) per-query")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "w_vs_kl.png", dpi=140)
        plt.close(fig)
        print(f"Saved {RESULTS_DIR / 'w_vs_kl.png'}")

    # Stability plot
    stab_path = RESULTS_DIR / "stability.json"
    if stab_path.exists():
        stab = json.loads(stab_path.read_text())
        method_names = list(stab["per_method_mean_over_perturbations"].keys())
        method_names.sort(key=lambda m: stab["per_method_mean_over_perturbations"][m])
        perts = ["drop_stop", "hedge", "lower_nop"]
        means = np.array([
            [stab["summary"][m].get(p, {}).get("mean", float("nan")) for p in perts]
            for m in method_names
        ])
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(method_names))
        width = 0.27
        for i, p in enumerate(perts):
            ax.bar(x + (i - 1) * width, means[:, i], width, label=p)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=20, ha="right")
        ax.set_ylabel("W_C(p, p') — lower = more stable")
        ax.set_title("Retrieval distribution shift under query perturbation")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "stability_bars.png", dpi=140)
        plt.close(fig)
        print(f"Saved {RESULTS_DIR / 'stability_bars.png'}")


if __name__ == "__main__":
    main()
