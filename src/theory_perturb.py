"""E4 -- Certified stability radius and the Lipschitz response curve.

A new evaluation axis for retrieval: how large an energy perturbation can the
top-k withstand before it changes? At the base config (h=0.5) we sweep the
perturbation magnitude s and, for both proximals, record:

  (a) W_C(p_T(E), p_T(E + s*dE))          -- the distributional Lipschitz curve;
  (b) top-10 set preservation |S0 ∩ S_s|/10 -- the certified-radius curve;

then report the per-query certified radius = the largest s keeping the top-10
set fully intact, averaged over random perturbation directions.

Output: results/theory_perturb.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from theory_common import (
    load_pools, build_inputs, wc, make_cfg, flow,
    laplacian_spectrum, band_slices, random_band_perturbation,
    select_queries, RESULTS_DIR,
)
from evaluation import bootstrap_ci


def topk_set(p, k=10):
    return set(np.argsort(-p)[:k].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dirs", type=int, default=3)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    pools = load_pools(args.dataset)
    qrows = select_queries(pools, n=args.n, seed=args.seed)
    print(f"[perturb] {args.dataset}: {len(qrows)} q, s_grid={s_grid}, h={args.h}", flush=True)
    rng = np.random.default_rng(args.seed)

    modes = ("wasserstein", "kl")
    wc_curve = {m: {s: [] for s in s_grid} for m in modes}
    pres_curve = {m: {s: [] for s in s_grid} for m in modes}
    cert_radius = {m: [] for m in modes}     # per (query,dir)

    for n, ci in enumerate(qrows):
        qi = build_inputs(pools, ci)
        evals, U = laplacian_spectrum(qi.C, eps=0.1)
        # Certified radius is measured against CROSS-CLUSTER (high-frequency)
        # perturbations -- the regime the theory (Prop. 2) says W^2 damps and
        # that paraphrase / distractor injection actually induce. Full-spectrum
        # noise is dominated by intra-cluster components where no method has a
        # geometric advantage and the sharper KL map looks artificially stable.
        lo_hi = band_slices(len(qi.C), 4)[-1]   # top frequency band
        dEs = [random_band_perturbation(U, lo_hi[0], lo_hi[1], rng) for _ in range(args.dirs)]
        for m in modes:
            cfg = make_cfg(h=args.h, T=args.T, mode=m)
            p_base = flow(qi.energy, qi.C, qi.K, qi.p0, cfg)
            S0 = topk_set(p_base, args.k)
            for dE in dEs:
                radius = 0.0
                broke = False
                for s in s_grid:
                    if s == 0.0:
                        wc_curve[m][s].append(0.0)
                        pres_curve[m][s].append(1.0)
                        continue
                    p_pert = flow(qi.energy + s * dE, qi.C, qi.K, qi.p0, cfg)
                    wc_curve[m][s].append(wc(p_base, p_pert, qi.C))
                    inter = len(S0 & topk_set(p_pert, args.k)) / args.k
                    pres_curve[m][s].append(inter)
                    if (not broke) and inter == 1.0:
                        radius = s
                    elif not broke:
                        broke = True
                cert_radius[m].append(radius)
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(qrows)}", flush=True)

    out = {"dataset": args.dataset, "n_queries": len(qrows), "h": args.h, "T": args.T,
           "k": args.k, "dirs": args.dirs, "s_grid": s_grid,
           "perturbation": "cross-cluster (top graph-frequency band)",
           "wc_curve": {}, "preservation_curve": {}, "certified_radius": {}}
    for m in modes:
        out["wc_curve"][m] = [float(np.mean(wc_curve[m][s])) for s in s_grid]
        out["preservation_curve"][m] = [float(np.mean(pres_curve[m][s])) for s in s_grid]
        mean, lo, hi = bootstrap_ci(cert_radius[m])
        out["certified_radius"][m] = {
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "median": float(np.median(cert_radius[m])),
        }
    rW = out["certified_radius"]["wasserstein"]
    rK = out["certified_radius"]["kl"]
    out["headline"] = {
        "radius_W_mean": rW["mean"], "radius_KL_mean": rK["mean"],
        "radius_ratio_W_over_KL": float(rW["mean"] / max(rK["mean"], 1e-9)),
    }
    (RESULTS_DIR / "theory_perturb.json").write_text(json.dumps(out, indent=2))
    print(f"[perturb] saved. certified radius W={rW['mean']:.3f} KL={rK['mean']:.3f} "
          f"(W/KL={out['headline']['radius_ratio_W_over_KL']:.2f})", flush=True)


if __name__ == "__main__":
    main()
