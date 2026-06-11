"""E3 -- Stability gap vs step size h (verifies Corollary 3, the sharp prediction).

Theory predicts the W-vs-KL stability gap is monotonically decreasing in h: it
is maximal for small h (strong proximal, the base config) and vanishes as
h -> infinity (weak proximal, the tuned config). We measure, at each h, the
energy-perturbation stability W_C(p_T(E), p_T(E + s*dE)) for both proximals,
averaged over queries and random cross-spectrum perturbation directions.

Output: results/theory_hsweep.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from theory_common import (
    load_pools, build_inputs, wc, make_cfg, flow,
    laplacian_spectrum, random_band_perturbation,
    select_queries, RESULTS_DIR,
)
from evaluation import bootstrap_ci, paired_bootstrap_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dirs", type=int, default=2)
    ap.add_argument("--s", type=float, default=0.3)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hs = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    pools = load_pools(args.dataset)
    qrows = select_queries(pools, n=args.n, seed=args.seed)
    print(f"[hsweep] {args.dataset}: {len(qrows)} q, h in {hs}, s={args.s}", flush=True)

    # Pre-build per-query inputs and a fixed full-spectrum perturbation set so the
    # SAME perturbations are reused across h (clean paired comparison across h).
    rng = np.random.default_rng(args.seed)
    qinfo = []
    for ci in qrows:
        qi = build_inputs(pools, ci)
        evals, U = laplacian_spectrum(qi.C, eps=0.1)
        dEs = [random_band_perturbation(U, 0, len(qi.C), rng) for _ in range(args.dirs)]
        qinfo.append((qi, dEs))

    out = {"dataset": args.dataset, "n_queries": len(qrows), "s": args.s,
           "T": args.T, "dirs": args.dirs, "h_grid": hs, "by_h": []}
    for h in hs:
        wW_per, wK_per = [], []          # per (query,dir) W_C, paired
        for qi, dEs in qinfo:
            for dE in dEs:
                cfgW = make_cfg(h=h, T=args.T, mode="wasserstein")
                cfgK = make_cfg(h=h, T=args.T, mode="kl")
                pbW = flow(qi.energy, qi.C, qi.K, qi.p0, cfgW)
                ppW = flow(qi.energy + args.s * dE, qi.C, qi.K, qi.p0, cfgW)
                pbK = flow(qi.energy, qi.C, qi.K, qi.p0, cfgK)
                ppK = flow(qi.energy + args.s * dE, qi.C, qi.K, qi.p0, cfgK)
                wW_per.append(wc(pbW, ppW, qi.C))
                wK_per.append(wc(pbK, ppK, qi.C))
        mW, loW, hiW = bootstrap_ci(wW_per)
        mK, loK, hiK = bootstrap_ci(wK_per)
        dgap, glo, ghi = paired_bootstrap_diff(wK_per, wW_per)   # KL - W
        out["by_h"].append({
            "h": h,
            "wc_W": {"mean": mW, "ci_lo": loW, "ci_hi": hiW},
            "wc_KL": {"mean": mK, "ci_lo": loK, "ci_hi": hiK},
            "gap_KL_minus_W": {"mean": dgap, "ci_lo": glo, "ci_hi": ghi},
            "rel_gap": float(dgap / max(mK, 1e-9)),
        })
        print(f"  h={h:<5.2f} W={mW:.4f} KL={mK:.4f} gap={dgap:+.4f} "
              f"[{glo:+.4f},{ghi:+.4f}]", flush=True)

    (RESULTS_DIR / "theory_hsweep.json").write_text(json.dumps(out, indent=2))
    gaps = [r["gap_KL_minus_W"]["mean"] for r in out["by_h"]]
    mono = all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1))
    print(f"[hsweep] saved. gap monotone-decreasing in h: {mono} | gaps={['%.4f'%g for g in gaps]}",
          flush=True)


if __name__ == "__main__":
    main()
