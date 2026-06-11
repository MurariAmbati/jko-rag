"""Print the exact numbers needed to fill the paper's \\todofill placeholders."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def g(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None


print("=" * 60)
d = g("theory_descent.json")
if d:
    fw = d["free_energy"]["wasserstein"]
    fk = d["free_energy"]["kl"]
    print(f"DESCENT: terminal F  W={fw['mean'][-1]:.3f}  KL={fk['mean'][-1]:.3f}")
    print(f"         monotone   W={fw['monotone_frac']:.2f}  KL={fk['monotone_frac']:.2f}")

print("=" * 60)
d = g("theory_response.json")
if d:
    h = d["headline"]
    print(f"RESPONSE (W_C metric):")
    print(f"  Wcross (W cross-cluster gain)  = {h['W_cross_gain']:.4f}")
    print(f"  Wintra (W intra-cluster gain)  = {h['W_intra_gain']:.4f}")
    print(f"  Whilo  (W high/low ratio)      = {h['W_high_over_low']:.2f}")
    print(f"  Khilo  (KL high/low ratio)     = {h['KL_high_over_low']:.2f}")
    print(f"  KL cross-cluster gain          = {h['KL_cross_gain']:.4f}")

print("=" * 60)
d = g("theory_hsweep.json")
if d:
    rows = d["by_h"]
    lo = rows[0]; hi = rows[-1]
    print(f"HSWEEP:")
    print(f"  gaplo  (gap at h={lo['h']})  = {lo['gap_KL_minus_W']['mean']:+.4f} "
          f"[{lo['gap_KL_minus_W']['ci_lo']:+.4f},{lo['gap_KL_minus_W']['ci_hi']:+.4f}]")
    print(f"  gaphi  (gap at h={hi['h']})  = {hi['gap_KL_minus_W']['mean']:+.4f} "
          f"[{hi['gap_KL_minus_W']['ci_lo']:+.4f},{hi['gap_KL_minus_W']['ci_hi']:+.4f}]")
    gaps = [r['gap_KL_minus_W']['mean'] for r in rows]
    mono = all(gaps[i] >= gaps[i+1] - 1e-9 for i in range(len(gaps)-1))
    print(f"  monotone-decreasing: {mono} | all gaps: {['%+.4f'%x for x in gaps]}")

print("=" * 60)
d = g("theory_perturb.json")
if d:
    rW = d["certified_radius"]["wasserstein"]
    rK = d["certified_radius"]["kl"]
    print(f"PERTURB (robustness decomposition):")
    print(f"  wcW (W_C response @max s, W)   = {d['wc_curve']['wasserstein'][-1]:.4f}")
    print(f"  wcK (W_C response @max s, KL)  = {d['wc_curve']['kl'][-1]:.4f}")
    print(f"  radW (top-10 set radius, W)    = {rW['mean']:.3f} [{rW['ci_lo']:.3f},{rW['ci_hi']:.3f}]")
    print(f"  radK (top-10 set radius, KL)   = {rK['mean']:.3f} [{rK['ci_lo']:.3f},{rK['ci_hi']:.3f}]")
print("=" * 60)
