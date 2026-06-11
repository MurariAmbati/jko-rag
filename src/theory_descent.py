"""E1 -- Free-energy descent curves (validates the gradient-flow foundation).

For each query we run the JKO flow under both proximals and record the free
energy F(p_t) at every outer step, plus the per-step transport velocity
W_C(p_{t-1}, p_t). Confirms (i) F decreases monotonically (the scheme really is
a free-energy gradient flow), and (ii) W and KL reach similar terminal F (hence
similar nDCG) along geometrically different paths.

Output: results/theory_descent.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from theory_common import (
    load_pools, build_inputs, free_energy, wc, make_cfg, flow_trace,
    select_queries, RESULTS_DIR, BASE_CFG,
)
from evaluation import bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pools = load_pools(args.dataset)
    qrows = select_queries(pools, n=args.n, seed=args.seed)
    lam, rho = BASE_CFG["lam"], BASE_CFG["rho"]
    print(f"[descent] {args.dataset}: {len(qrows)} queries, h={args.h}, T={args.T}",
          flush=True)

    modes = ("wasserstein", "kl")
    F_curves = {m: [] for m in modes}        # list over queries of [F_0..F_T]
    V_curves = {m: [] for m in modes}        # per-step W_C(p_{t-1}, p_t)
    for n, ci in enumerate(qrows):
        qi = build_inputs(pools, ci)
        for m in modes:
            cfg = make_cfg(h=args.h, T=args.T, mode=m)
            traj = flow_trace(qi.energy, qi.C, qi.K, qi.p0, cfg)
            F_curves[m].append([free_energy(p, qi.energy, qi.K, lam, rho) for p in traj])
            V_curves[m].append([wc(traj[t], traj[t + 1], qi.C) for t in range(len(traj) - 1)])
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(qrows)}", flush=True)

    out = {"dataset": args.dataset, "n_queries": len(qrows), "h": args.h, "T": args.T,
           "lam": lam, "rho": rho, "free_energy": {}, "velocity": {}}
    for m in modes:
        Fc = np.array(F_curves[m])           # (Q, T+1)
        Vc = np.array(V_curves[m])           # (Q, T)
        out["free_energy"][m] = {
            "mean": Fc.mean(axis=0).tolist(),
            "ci_lo": [bootstrap_ci(Fc[:, t].tolist())[1] for t in range(Fc.shape[1])],
            "ci_hi": [bootstrap_ci(Fc[:, t].tolist())[2] for t in range(Fc.shape[1])],
            "monotone_frac": float(np.mean([
                all(Fc[q, t + 1] <= Fc[q, t] + 1e-6 for t in range(Fc.shape[1] - 1))
                for q in range(Fc.shape[0])])),
            "total_drop_mean": float((Fc[:, 0] - Fc[:, -1]).mean()),
        }
        out["velocity"][m] = {"mean": Vc.mean(axis=0).tolist()}

    (RESULTS_DIR / "theory_descent.json").write_text(json.dumps(out, indent=2))
    print(f"[descent] saved. terminal F: "
          f"W={out['free_energy']['wasserstein']['mean'][-1]:.4f} "
          f"KL={out['free_energy']['kl']['mean'][-1]:.4f} | "
          f"monotone W={out['free_energy']['wasserstein']['monotone_frac']:.2f} "
          f"KL={out['free_energy']['kl']['monotone_frac']:.2f}", flush=True)


if __name__ == "__main__":
    main()
