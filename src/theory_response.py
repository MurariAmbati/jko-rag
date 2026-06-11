"""E2 -- Response anisotropy across graph-frequency bands (verifies Prop. 2).

For each query we build the Gibbs-affinity graph Laplacian of the OT geometry
and split its eigenbasis into frequency bands (low = smooth/intra-cluster,
high = oscillatory/cross-cluster). For each band we apply unit-norm, mean-zero
energy perturbations and measure the response gain ||dp|| / s of the JKO map
under both proximals.

Prediction (Corollary of Thm 1 + Prop 2): the Wasserstein gain DECREASES with
frequency (cross-cluster perturbations are damped), while KL stays comparatively
flat -- so W's cross-cluster gain is strictly below KL's.

Output: results/theory_response.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from theory_common import (
    load_pools, build_inputs, make_cfg, flow, wc,
    laplacian_spectrum, band_slices, random_band_perturbation,
    select_queries, RESULTS_DIR,
)
from evaluation import bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--dirs", type=int, default=3, help="random perturbations per band")
    ap.add_argument("--s", type=float, default=0.25, help="perturbation magnitude")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pools = load_pools(args.dataset)
    qrows = select_queries(pools, n=args.n, seed=args.seed)
    print(f"[response] {args.dataset}: {len(qrows)} q, {args.bands} bands, "
          f"{args.dirs} dirs/band, s={args.s}, h={args.h}", flush=True)

    modes = ("wasserstein", "kl")
    # per-query, per-band mean gain, measured BOTH in W_C (geometric, primary) and
    # Euclidean (secondary). W_C is the paper's stability metric and the
    # apples-to-apples comparison across proximals.
    wcg = {m: [[] for _ in range(args.bands)] for m in modes}   # W_C gain
    eug = {m: [[] for _ in range(args.bands)] for m in modes}   # Euclidean gain
    rng = np.random.default_rng(args.seed)

    for n, ci in enumerate(qrows):
        qi = build_inputs(pools, ci)
        evals, U = laplacian_spectrum(qi.C, eps=0.1)
        bands = band_slices(len(qi.C), args.bands)
        for m in modes:
            cfg = make_cfg(h=args.h, T=args.T, mode=m)
            p_base = flow(qi.energy, qi.C, qi.K, qi.p0, cfg)
            for b, (lo, hi) in enumerate(bands):
                wg_, eg_ = [], []
                for _ in range(args.dirs):
                    dE = random_band_perturbation(U, lo, hi, rng)
                    p_pert = flow(qi.energy + args.s * dE, qi.C, qi.K, qi.p0, cfg)
                    wg_.append(wc(p_base, p_pert, qi.C) / args.s)
                    eg_.append(float(np.linalg.norm(p_pert - p_base) / args.s))
                wcg[m][b].append(float(np.mean(wg_)))
                eug[m][b].append(float(np.mean(eg_)))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(qrows)}", flush=True)

    out = {"dataset": args.dataset, "n_queries": len(qrows), "bands": args.bands,
           "dirs_per_band": args.dirs, "s": args.s, "h": args.h, "T": args.T,
           "metric": "W_C response gain (primary); Euclidean (secondary)",
           "per_band": {}, "per_band_euclid": {}}
    for m in modes:
        out["per_band"][m] = []
        out["per_band_euclid"][m] = []
        for b in range(args.bands):
            mean, lo, hi = bootstrap_ci(wcg[m][b])
            out["per_band"][m].append({"band": b, "gain_mean": mean,
                                        "gain_ci_lo": lo, "gain_ci_hi": hi})
            em, el, eh = bootstrap_ci(eug[m][b])
            out["per_band_euclid"][m].append({"band": b, "gain_mean": em,
                                              "gain_ci_lo": el, "gain_ci_hi": eh})
    wg = [out["per_band"]["wasserstein"][b]["gain_mean"] for b in range(args.bands)]
    kg = [out["per_band"]["kl"][b]["gain_mean"] for b in range(args.bands)]
    out["headline"] = {
        "metric": "W_C",
        "W_cross_gain": wg[-1], "KL_cross_gain": kg[-1],
        "W_intra_gain": wg[0], "KL_intra_gain": kg[0],
        "W_high_over_low": wg[-1] / max(wg[0], 1e-9),
        "KL_high_over_low": kg[-1] / max(kg[0], 1e-9),
        "cross_gain_ratio_W_over_KL": wg[-1] / max(kg[-1], 1e-9),
    }
    (RESULTS_DIR / "theory_response.json").write_text(json.dumps(out, indent=2))
    print(f"[response] saved. W_C cross-cluster gain W={wg[-1]:.4f} KL={kg[-1]:.4f} "
          f"(W/KL={out['headline']['cross_gain_ratio_W_over_KL']:.2f}); "
          f"W high/low={out['headline']['W_high_over_low']:.2f} "
          f"KL high/low={out['headline']['KL_high_over_low']:.2f}", flush=True)


if __name__ == "__main__":
    main()
