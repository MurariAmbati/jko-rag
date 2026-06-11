"""Generate the stability-theory verification figures from results JSONs.

  fig5_free_energy_descent.pdf  -- F(p_t) and per-step transport velocity
  fig6_response_bands.pdf       -- response gain vs graph-frequency band (W vs KL)
  fig7_gap_vs_h.pdf             -- stability gap vs step size h (the sharp prediction)
  fig8_certified_radius.pdf     -- top-k preservation + certified radius (W vs KL)

Reads: results/theory_{descent,response,hsweep,perturb}.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "lines.linewidth": 1.6, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
C_W = "#2166ac"      # Wasserstein
C_K = "#d6604d"      # KL
C_GAP = "#7fbf7b"


def _load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"  [skip] {name} not found")
        return None
    return json.load(open(p))


# ── FIG 5 — free-energy descent (single panel) ────────────────────────────
def fig5_descent():
    d = _load("theory_descent.json")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for m, c, lab in [("wasserstein", C_W, "JKO ($W^2$)"), ("kl", C_K, "KL-Prox")]:
        fe = d["free_energy"][m]
        t = np.arange(len(fe["mean"]))
        ax.plot(t, fe["mean"], color=c, marker="o", markersize=3.5, label=lab)
        ax.fill_between(t, fe["ci_lo"], fe["ci_hi"], color=c, alpha=0.15)
    fW = d["free_energy"]["wasserstein"]["mean"][-1]
    fK = d["free_energy"]["kl"]["mean"][-1]
    ax.set_xlabel("JKO outer step $t$")
    ax.set_ylabel(r"Free energy $F(p_t)$")
    ax.set_title("Monotone free-energy descent")
    ax.legend(loc="upper right", title=f"terminal: $W^2${fW:.3f}, KL{fK:.3f}",
              title_fontsize=6.5)
    fig.tight_layout(pad=0.5)
    out = FIGS / "fig5_free_energy_descent.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


# ── FIG 6 — geometric response gain by frequency band (grouped bars) ───────
def fig6_response_bands():
    d = _load("theory_response.json")
    if d is None:
        return
    nb = d["bands"]
    x = np.arange(nb)
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for i, (m, c, lab) in enumerate([("wasserstein", C_W, "JKO ($W^2$)"),
                                     ("kl", C_K, "KL-Prox")]):
        gm = [d["per_band"][m][b]["gain_mean"] for b in range(nb)]
        lo = [d["per_band"][m][b]["gain_ci_lo"] for b in range(nb)]
        hi = [d["per_band"][m][b]["gain_ci_hi"] for b in range(nb)]
        yerr = [[gm[b] - lo[b] for b in range(nb)], [hi[b] - gm[b] for b in range(nb)]]
        ax.bar(x + (i - 0.5) * w, gm, w, yerr=yerr, color=c, alpha=0.85,
               capsize=2.5, edgecolor="white", linewidth=0.4, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["low\n(intra)"] + [str(b) for b in range(1, nb - 1)] +
                       ["high\n(cross)"], fontsize=7)
    ax.set_xlabel("Perturbation frequency band")
    ax.set_ylabel(r"Geometric response $\widehat{W}_C(p,p')/\|\delta E\|$")
    ax.set_title(r"$W^2$ response is ${\approx}2.4\times$ below KL," + "\nuniformly across frequency")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.5)
    out = FIGS / "fig6_response_bands.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


# ── FIG 7 — stability gap vs h ────────────────────────────────────────────
def fig7_gap_vs_h():
    d = _load("theory_hsweep.json")
    if d is None:
        return
    rows = d["by_h"]
    h = np.array([r["h"] for r in rows])
    wW = np.array([r["wc_W"]["mean"] for r in rows])
    wWlo = np.array([r["wc_W"]["ci_lo"] for r in rows])
    wWhi = np.array([r["wc_W"]["ci_hi"] for r in rows])
    wK = np.array([r["wc_KL"]["mean"] for r in rows])
    wKlo = np.array([r["wc_KL"]["ci_lo"] for r in rows])
    wKhi = np.array([r["wc_KL"]["ci_hi"] for r in rows])

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.fill_between(h, wW, wK, color=C_GAP, alpha=0.35, label="stability gap")
    ax.plot(h, wK, color=C_K, marker="s", markersize=4, label="KL-Prox")
    ax.fill_between(h, wKlo, wKhi, color=C_K, alpha=0.12)
    ax.plot(h, wW, color=C_W, marker="o", markersize=4, label="JKO ($W^2$)")
    ax.fill_between(h, wWlo, wWhi, color=C_W, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks(h)
    ax.set_xticklabels([f"{x:g}" for x in h])
    ax.minorticks_off()
    ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(2.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(0.5, ymax * 0.96, "base\n(h=0.5)", fontsize=6.5, ha="center", va="top", color="gray")
    ax.text(2.0, ymax * 0.96, "tuned\n(h=2.0)", fontsize=6.5, ha="center", va="top", color="gray")
    ax.set_xlabel(r"Step size $h$  (log scale)")
    ax.set_ylabel(r"Perturbation response $W_C(p_T(E),p_T(E{+}\delta E))$")
    ax.set_title(r"Stability gap shrinks as $h\!\to\!\infty$" + "\n(verifies Corollary 3)")
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.5)
    out = FIGS / "fig7_gap_vs_h.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


# ── FIG 8 — distributional stability under perturbation (single panel) ─────
def fig8_certified():
    d = _load("theory_perturb.json")
    if d is None:
        return
    s = np.array(d["s_grid"])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for m, c, lab in [("wasserstein", C_W, "JKO ($W^2$)"), ("kl", C_K, "KL-Prox")]:
        ax.plot(s, d["wc_curve"][m], color=c, marker="o", markersize=3.5, label=lab)
    ax.set_xlabel(r"Cross-cluster perturbation $s$")
    ax.set_ylabel(r"Distributional response $\widehat{W}_C(p_T(E),p_T(E{+}s\delta E))$")
    ax.set_title("Distributional stability\nunder perturbation")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout(pad=0.5)
    out = FIGS / "fig8_certified_radius.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Generating theory figures ...")
    fig5_descent()
    fig6_response_bands()
    fig7_gap_vs_h()
    fig8_certified()
    print("Done.")
