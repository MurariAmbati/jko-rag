"""Benchmark a single JKO step on M=200 pool to estimate total runtime."""
from __future__ import annotations

import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from retrieval import cost_matrix_cosine, redundancy_kernel, softmax_np
from jko import JKOConfig, run_jko

rng = np.random.default_rng(0)
M = 200
D = 384
Z = rng.normal(size=(M, D)).astype(np.float32)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
C = cost_matrix_cosine(Z)
K = redundancy_kernel(Z)
energy = rng.normal(size=M).astype(np.float32)
p0 = softmax_np(-energy, tau=0.1)

for mode in ["wasserstein", "kl", "noproximal"]:
    for T, inner, sk in [(3, 40, 60), (2, 25, 40), (2, 20, 30)]:
        cfg = JKOConfig(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1,
                        T=T, inner_steps=inner, sinkhorn_iter=sk, mode=mode)
        # warm
        _ = run_jko(p0, energy, C, K, cfg)
        t0 = time.time()
        for _ in range(3):
            _ = run_jko(p0, energy, C, K, cfg)
        dt = (time.time() - t0) / 3
        print(f"  mode={mode:<12s} T={T} inner={inner} sk={sk}  {dt*1000:.1f} ms/run")
