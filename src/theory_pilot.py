"""Fast pilot: sanity-check all four theoretical predictions on a few queries.

Run BEFORE the full experiments. If a prediction fails here, we rethink the
theory rather than burn an hour of compute. Prints diagnostic numbers only.

Predictions under test:
  P1 (descent)    : F(p_t) decreases monotonically for both W and KL.
  P2 (anisotropy) : W's response to high-freq (cross-cluster) energy
                    perturbations is damped vs low-freq; KL is flatter.
                    Headline: W cross-cluster gain < KL cross-cluster gain.
  P3 (h-sweep)    : stability gap (KL W_C - W W_C) > 0 and DECREASES as h grows.
  P4 (cert radius): W preserves its top-10 under larger energy perturbations.
"""
from __future__ import annotations

import numpy as np

from theory_common import (
    load_pools, build_inputs, free_energy, wc, make_cfg, flow, flow_trace,
    laplacian_spectrum, band_slices, random_band_perturbation, select_queries,
)


def topk_set(p: np.ndarray, k: int = 10) -> set:
    return set(np.argsort(-p)[:k].tolist())


def main():
    pools = load_pools("scifact")
    qrows = select_queries(pools, n=4, seed=0)
    print(f"Pilot on {len(qrows)} scifact queries: rows {qrows}\n", flush=True)
    lam, rho = 0.05, 0.05

    # ---- P1: free-energy descent ------------------------------------------
    print("=== P1: free-energy descent F(p_t) ===")
    for mode in ("wasserstein", "kl"):
        cfg = make_cfg(h=0.5, T=4, mode=mode)
        curves = []
        for ci in qrows:
            qi = build_inputs(pools, ci)
            traj = flow_trace(qi.energy, qi.C, qi.K, qi.p0, cfg)
            F = [free_energy(p, qi.energy, qi.K, lam, rho) for p in traj]
            curves.append(F)
        Fm = np.mean(curves, axis=0)
        mono = all(Fm[t + 1] <= Fm[t] + 1e-6 for t in range(len(Fm) - 1))
        print(f"  {mode:<12s} F: " + " -> ".join(f"{x:.4f}" for x in Fm)
              + f"   monotone={mono}")

    # ---- P2: response anisotropy across frequency bands -------------------
    print("\n=== P2: response gain by frequency band (||dp||/s) ===", flush=True)
    n_bands = 3
    s = 0.25
    rng = np.random.default_rng(1)
    band_gain = {m: np.zeros(n_bands) for m in ("wasserstein", "kl")}
    band_cnt = np.zeros(n_bands)
    for ci in qrows:
        qi = build_inputs(pools, ci)
        evals, U = laplacian_spectrum(qi.C, eps=0.1)
        bands = band_slices(len(qi.C), n_bands)
        for mode in ("wasserstein", "kl"):
            cfg = make_cfg(h=0.5, T=2, mode=mode)
            p_base = flow(qi.energy, qi.C, qi.K, qi.p0, cfg)
            for b, (lo, hi) in enumerate(bands):
                gains = []
                for _ in range(1):
                    dE = random_band_perturbation(U, lo, hi, rng)
                    p_pert = flow(qi.energy + s * dE, qi.C, qi.K, qi.p0, cfg)
                    gains.append(np.linalg.norm(p_pert - p_base) / s)
                band_gain[mode][b] += np.mean(gains)
                if mode == "wasserstein":
                    band_cnt[b] += 1
    for mode in ("wasserstein", "kl"):
        g = band_gain[mode] / np.maximum(band_cnt, 1)
        print(f"  {mode:<12s} low->high freq: " + "  ".join(f"{x:.3f}" for x in g))
    gW = band_gain["wasserstein"] / np.maximum(band_cnt, 1)
    gK = band_gain["kl"] / np.maximum(band_cnt, 1)
    print(f"  cross-cluster (highest band) gain:  W={gW[-1]:.3f}  KL={gK[-1]:.3f}"
          f"  -> W damps {'YES' if gW[-1] < gK[-1] else 'NO'}")
    print(f"  W high/low ratio={gW[-1]/max(gW[0],1e-9):.2f}  "
          f"KL high/low ratio={gK[-1]/max(gK[0],1e-9):.2f}")

    # ---- P3: stability gap vs h -------------------------------------------
    print("\n=== P3: stability gap (KL W_C - W W_C) vs h ===", flush=True)
    s_h = 0.3
    rng = np.random.default_rng(2)
    hs = [0.25, 0.5, 2.0]
    for h in hs:
        wc_by = {"wasserstein": [], "kl": []}
        for ci in qrows:
            qi = build_inputs(pools, ci)
            dE = random_band_perturbation(*(lambda ev_U: (ev_U[1], 0, len(qi.C)))(
                laplacian_spectrum(qi.C, eps=0.1)), rng)
            for mode in ("wasserstein", "kl"):
                cfg = make_cfg(h=h, T=3, mode=mode)
                p_base = flow(qi.energy, qi.C, qi.K, qi.p0, cfg)
                p_pert = flow(qi.energy + s_h * dE, qi.C, qi.K, qi.p0, cfg)
                wc_by[mode].append(wc(p_base, p_pert, qi.C))
        wW = float(np.mean(wc_by["wasserstein"]))
        wK = float(np.mean(wc_by["kl"]))
        print(f"  h={h:<5.2f}  W_C: W={wW:.4f}  KL={wK:.4f}  gap={wK - wW:+.4f}")

    # ---- P4: certified radius (top-10 preservation) -----------------------
    print("\n=== P4: top-10 preservation vs perturbation magnitude ===", flush=True)
    rng = np.random.default_rng(3)
    s_grid = [0.1, 0.25, 0.5, 1.0]
    pres = {m: np.zeros(len(s_grid)) for m in ("wasserstein", "kl")}
    cnt = 0
    for ci in qrows:
        qi = build_inputs(pools, ci)
        evals, U = laplacian_spectrum(qi.C, eps=0.1)
        dE = random_band_perturbation(U, 0, len(qi.C), rng)
        for mode in ("wasserstein", "kl"):
            cfg = make_cfg(h=0.5, T=3, mode=mode)
            p_base = flow(qi.energy, qi.C, qi.K, qi.p0, cfg)
            S0 = topk_set(p_base, 10)
            for j, sm in enumerate(s_grid):
                p_pert = flow(qi.energy + sm * dE, qi.C, qi.K, qi.p0, cfg)
                pres[mode][j] += len(S0 & topk_set(p_pert, 10)) / 10.0
        cnt += 1
    for mode in ("wasserstein", "kl"):
        pr = pres[mode] / cnt
        print(f"  {mode:<12s} s={s_grid}: " + "  ".join(f"{x:.2f}" for x in pr))

    print("\nPilot done.")


if __name__ == "__main__":
    main()
