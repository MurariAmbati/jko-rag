"""Generate paper figures from result JSON files.

Outputs to paper/figures/:
  fig1_stability_bars.pdf     -- stability comparison (jko vs kl vs rerank)
  fig2_selective_curves.pdf   -- DUAL-RANK selective coverage on 4 datasets
  fig3_sam_jko.pdf            -- SAM-JKO quality-speedup tradeoff
  fig4_alpha_sweep.pdf        -- BW-JKO stability vs alpha
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLORS = {
    "jko":    "#2166ac",
    "kl":     "#d6604d",
    "rerank": "#999999",
    "noprox": "#4dac26",
    "dual":   "#762a83",
    "softmax":"#e08214",
    "margin": "#40004b",
    "sam":    "#1b7837",
}

# ─────────────────────────────────────────────
# FIG 1 — Stability bars (jko vs kl vs rerank)
# ─────────────────────────────────────────────

def fig1_stability():
    datasets = ["scifact", "nfcorpus", "trec-covid", "fiqa"]
    labels   = ["SciFact", "NFCorpus", "TREC-COVID", "FiQA"]
    data = {}
    for ds in datasets:
        p = RESULTS / f"stability_{ds}.json"
        if not p.exists():
            continue
        with open(p) as f:
            obj = json.load(f)
        # obj is list of dicts or dict — handle both shapes
        if isinstance(obj, list):
            d = {r["method"]: r["mean_wc"] for r in obj}
        else:
            d = obj.get("per_method", obj)
        data[ds] = d

    if not data:
        print("  [fig1] no stability JSON found; skipping")
        return

    methods    = ["jko_rerank", "kl_rerank", "rerank_topk"]
    mlabels    = ["JKO-RAG", "KL-Prox", "Cross-Enc Top-k"]
    mcolors    = [COLORS["jko"], COLORS["kl"], COLORS["rerank"]]
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for i, (m, ml, mc) in enumerate(zip(methods, mlabels, mcolors)):
        vals = []
        for ds in datasets:
            v = data.get(ds, {}).get(m, None)
            vals.append(v if v is not None else 0)
        bars = ax.bar(x + i * width, vals, width, label=ml, color=mc, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean $W_C(p_T(q),\,p_T(q'))$  ↓")
    ax.set_title("Stability under Query Perturbation (lower = more stable)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout(pad=0.5)
    out = FIGS / "fig1_stability_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# FIG 2 — DUAL-RANK selective coverage curves
# ─────────────────────────────────────────────

def fig2_selective():
    datasets = ["scifact", "nfcorpus", "fiqa", "scidocs"]
    dlabels  = ["SciFact", "NFCorpus", "FiQA", "SCIDOCS"]
    signals  = ["conf_dual", "conf_softmax", "conf_margin"]
    slabels  = ["Dual pot. (ours)", "Softmax-max", "Prob. margin"]
    scolors  = [COLORS["dual"], COLORS["softmax"], COLORS["margin"]]
    slines   = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), sharey=False)
    for ax, ds, dl in zip(axes, datasets, dlabels):
        p = RESULTS / f"dual_selective_{ds}.json"
        if not p.exists():
            ax.set_visible(False)
            continue
        with open(p) as f:
            obj = json.load(f)
        curves = obj.get("selective_curves", {})
        for sig, sl, sc, ss in zip(signals, slabels, scolors, slines):
            curve = curves.get(sig, [])
            if not curve:
                continue
            covs  = [c["coverage"] for c in curve]
            ndcgs = [c["ndcg@10_mean"] for c in curve]
            ax.plot(covs, ndcgs, color=sc, linestyle=ss, label=sl, marker="o",
                    markersize=2.5, markeredgewidth=0)
        ax.set_xlim(1.0, 0.10)  # coverage decreases left to right
        ax.invert_xaxis()
        ax.set_xlabel("Coverage fraction")
        ax.set_title(dl)
        if ax is axes[0]:
            ax.set_ylabel("nDCG@10")
    # shared legend below
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("DUAL-RANK Selective Coverage: nDCG@10 vs Abstention Threshold",
                 y=1.02, fontsize=9)
    fig.tight_layout(pad=0.4)
    out = FIGS / "fig2_selective_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# FIG 3 — SAM-JKO quality vs speedup
# ─────────────────────────────────────────────

def fig3_sam():
    p = RESULTS / "mr_jko_bench.json"
    if not p.exists():
        print("  [fig3] mr_jko_bench.json not found; skipping")
        return
    with open(p) as f:
        obj = json.load(f)
    # JSON has top-level key "scifact_test" containing a "summary" dict
    scifact_test = obj.get("scifact_test", {})
    summary = scifact_test.get("summary", {})
    if not summary:
        bench = obj.get("scifact_bench", obj.get("benchmark", []))
    else:
        # convert summary dict to list of records
        bench = []
        for mname, mdata in summary.items():
            bench.append({
                "method": mname,
                "ndcg@10": mdata["ndcg@10"]["mean"],
                "speedup": mdata.get("speedup_x", 1.0),
            })
    if not bench:
        print("  [fig3] no scifact_bench key found; skipping")
        return

    names   = [r["method"] for r in bench]
    ndcgs   = [r.get("ndcg@10", r.get("ndcg", 0)) for r in bench]
    speedup = [r.get("speedup", 1.0) for r in bench]

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    cmap = plt.cm.get_cmap("RdYlGn", len(names))
    for i, (n, nd, sp) in enumerate(zip(names, ndcgs, speedup)):
        label = n.replace("_", " ").replace("sam b", "SAM β=").replace("mr kmeans", "MR (plain)").replace("vanilla", "Vanilla JKO")
        color = COLORS["sam"] if "sam" in n else (COLORS["jko"] if "vanilla" in n else COLORS["kl"])
        ax.scatter(sp, nd, s=60, color=color, zorder=3)
        ax.annotate(label, (sp, nd), textcoords="offset points", xytext=(4, 2), fontsize=7)

    ax.set_xlabel("Speedup vs vanilla JKO  →")
    ax.set_ylabel("nDCG@10  →")
    ax.set_title("SAM-JKO: Quality–Speed Pareto (SciFact)")
    ax.axvline(1.0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    fig.tight_layout(pad=0.5)
    out = FIGS / "fig3_sam_jko.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# FIG 4 — BW-JKO alpha sweep stability
# ─────────────────────────────────────────────

def fig4_alpha_sweep():
    datasets = ["scifact", "nfcorpus", "fiqa"]
    dlabels  = ["SciFact", "NFCorpus", "FiQA"]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]  # 0.0 = KL, 1.0 = W²
    alpha_labels = ["KL\n(α=0)", "α=0.25", "α=0.50", "α=0.75", "W²\n(α=1)"]
    method_map = {
        0.0:  "kl_rerank",
        0.25: "bw_jko_a25",
        0.50: "bw_jko_a50",
        0.75: "bw_jko_a75",
        1.0:  "jko_rerank",
    }

    ds_colors = ["#2166ac", "#d6604d", "#4dac26"]
    for ds, dl, dc in zip(datasets, dlabels, ds_colors):
        p = RESULTS / f"stability_new_{ds}.json"
        if not p.exists():
            continue
        with open(p) as f:
            obj = json.load(f)
        summary = obj.get("per_method_mean_over_perturbations", {})
        vals = []
        for a in alphas:
            key = method_map[a]
            vals.append(summary.get(key, np.nan))
        ax.plot(alphas, vals, color=dc, marker="o", markersize=4, label=dl)

    ax.set_xticks(alphas)
    ax.set_xticklabels(alpha_labels, fontsize=7.5)
    ax.set_xlabel("Interpolation α  (0 = KL, 1 = Wasserstein²)")
    ax.set_ylabel(r"Mean $W_C$  ↓  (more stable)")
    ax.set_title("BW-JKO Stability vs. Interpolation Weight")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.5)
    out = FIGS / "fig4_alpha_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures ...")
    fig1_stability()
    fig2_selective()
    fig3_sam()
    fig4_alpha_sweep()
    print("Done.")
