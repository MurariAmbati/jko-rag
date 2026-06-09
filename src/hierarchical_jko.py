"""C4 -- Multi-resolution Hierarchical JKO (MR-JKO).

Standard JKO operates on a fixed candidate pool of size M (e.g. M=200) and
its dominant cost is the Sinkhorn O(M^2) per iteration. To scale JKO to
LARGE candidate pools (M=1000+), we propose a multi-resolution scheme:

  Step 1 (coarse). Cluster the M candidates into G groups via spherical
    k-means on their embeddings.  Compute group centroids mu_1..mu_G in the
    same embedding space.  Run T_coarse JKO steps on the simplex of size G,
    using:
      energy^coarse_g = - average relevance of candidates in group g
      cost^coarse_gg' = (1 - cos(mu_g, mu_g'))^2
    This yields a distribution p^coarse over groups (size G << M).

  Step 2 (selection). Take the top G_keep groups by p^coarse and keep their
    member candidates as a refined pool of size <= M_keep.

  Step 3 (fine). Run T_fine JKO steps on this smaller pool with the regular
    fine-grained cost matrix.  This produces the final distribution p over
    a much-reduced support, but the OT operation is now on M_keep << M
    candidates.

Why this is novel.
------------------
Multigrid / multi-resolution methods are classical in PDE solvers and in
some metric-learning / structured-prediction methods, but to our knowledge
have not been used for OT-based retrieval refinement. The contribution is
both theoretical (the coarse JKO well-approximates the fine JKO in the
limit of perfect clustering) and practical (sub-quadratic scaling enables
larger candidate pools, which can substantially raise pool recall).

Computational complexity.
-------------------------
Vanilla JKO: O(T * inner_steps * M^2)  -- Sinkhorn dominates.
MR-JKO:      O(T_c * inner * G^2)  +  O(T_f * inner * M_keep^2)
For M=1000, G=50, M_keep=200: roughly 1/16 of vanilla's Sinkhorn cost.
"""
from __future__ import annotations

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from jko import JKOConfig, run_jko
from retrieval import cost_matrix_cosine, redundancy_kernel, normalize_minmax, softmax_np


def spherical_kmeans(Z: np.ndarray, G: int, n_iter: int = 30, seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Simple spherical k-means on unit-norm embeddings.

    Returns (assignments, centroids) where assignments has shape (M,) with
    values in [0, G), and centroids has shape (G, d) -- L2-normalised.
    """
    M, d = Z.shape
    rng = np.random.default_rng(seed)
    # init by random subset of points
    init_ids = rng.choice(M, size=G, replace=False)
    centroids = Z[init_ids].copy()
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-8)
    assignments = np.zeros(M, dtype=np.int64)
    for _ in range(n_iter):
        # E-step: assign each point to closest centroid by cosine
        sims = Z @ centroids.T            # (M, G)
        new_assign = np.argmax(sims, axis=1)
        if (new_assign == assignments).all():
            break
        assignments = new_assign
        # M-step: recompute centroids
        for g in range(G):
            members = Z[assignments == g]
            if len(members) > 0:
                mu = members.mean(axis=0)
                n = np.linalg.norm(mu)
                centroids[g] = mu / max(n, 1e-8)
            # else: keep previous centroid (orphan cluster)
    return assignments, centroids


def relevance_aware_kmeans(
    Z: np.ndarray, relevance: np.ndarray, G: int, beta: float = 1.0,
    n_iter: int = 30, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """SAM-JKO clustering: cluster on augmented features (z_i, beta * rel_i).

    Higher beta means relevance dominates clustering. beta=0 reduces to plain
    spherical k-means. beta=1 (default) gives roughly equal weight to embedding
    geometry and relevance.

    The augmented features are NOT re-normalised so the metric is Euclidean (not
    cosine). Relevance scores must be in [0, 1].
    """
    M, d = Z.shape
    # Augment each point with beta * rel as an extra coordinate.
    aug = np.concatenate([Z, (beta * relevance[:, None]).astype(np.float32)], axis=1)
    rng = np.random.default_rng(seed)
    init_ids = rng.choice(M, size=G, replace=False)
    centroids = aug[init_ids].copy()
    assignments = np.zeros(M, dtype=np.int64)
    for _ in range(n_iter):
        # Euclidean distance to each centroid
        dists = ((aug[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        new_assign = np.argmin(dists, axis=1)
        if (new_assign == assignments).all():
            break
        assignments = new_assign
        for g in range(G):
            members = aug[assignments == g]
            if len(members) > 0:
                centroids[g] = members.mean(axis=0)
    # Return only the embedding part of centroids, L2-normalised
    cent_z = centroids[:, :d]
    cent_z = cent_z / np.linalg.norm(cent_z, axis=1, keepdims=True).clip(1e-8)
    return assignments, cent_z


def mr_jko(
    Z: np.ndarray,            # (M, d) candidate embeddings (assumed L2 normalised)
    relevance: np.ndarray,    # (M,) per-chunk relevance in [0, 1]
    G: int = 50,
    G_keep: int = 8,
    coarse_cfg: dict | None = None,
    fine_cfg: dict | None = None,
    return_internal: bool = False,
    clustering: str = "kmeans",   # "kmeans" | "sam" (Score-Aware MR)
    sam_beta: float = 1.0,         # weight of relevance in SAM clustering
) -> tuple[np.ndarray, np.ndarray]:
    """Run multi-resolution JKO.

    Returns (final_distribution_p_over_M, refined_pool_indices_into_M).
    final_p has support only on the refined pool (other entries are 0).
    """
    if coarse_cfg is None:
        coarse_cfg = {"h": 1.0, "lam": 0.05, "rho": 0.05, "sinkhorn_eps": 0.1,
                       "T": 2, "inner_steps": 15, "tau0": 0.2}
    if fine_cfg is None:
        fine_cfg = {"h": 2.0, "lam": 0.1, "rho": 0.05, "sinkhorn_eps": 0.2,
                     "T": 3, "inner_steps": 25, "tau0": 1.0}

    M = Z.shape[0]
    if G >= M:
        G = max(2, M // 4)

    # --- Coarse step ---
    if clustering == "sam":
        assignments, centroids = relevance_aware_kmeans(Z, relevance, G, beta=sam_beta)
    else:
        assignments, centroids = spherical_kmeans(Z, G)
    # Group sizes (avoid 0)
    group_sizes = np.array([(assignments == g).sum() for g in range(G)], dtype=np.float32)
    group_sizes = np.clip(group_sizes, 1, None)
    # Group-level relevance: weighted mean of member relevance scores
    group_rel = np.zeros(G, dtype=np.float32)
    for g in range(G):
        members = relevance[assignments == g]
        if len(members) > 0:
            group_rel[g] = members.mean()
    rel_norm = normalize_minmax(group_rel)
    energy_coarse = -rel_norm
    C_coarse = cost_matrix_cosine(centroids).astype(np.float32)
    K_coarse = redundancy_kernel(centroids).astype(np.float32)
    p0_coarse = softmax_np(-energy_coarse, tau=coarse_cfg["tau0"])

    coarse_jko = JKOConfig(
        h=coarse_cfg["h"], lam=coarse_cfg["lam"], rho=coarse_cfg["rho"],
        sinkhorn_eps=coarse_cfg["sinkhorn_eps"], T=coarse_cfg["T"],
        inner_steps=coarse_cfg["inner_steps"], mode="wasserstein",
    )
    p_coarse, _ = run_jko(p0_coarse, energy_coarse, C_coarse, K_coarse, coarse_jko)

    # --- Selection ---
    G_keep = min(G_keep, G)
    top_groups = np.argsort(-p_coarse)[:G_keep]
    keep_mask = np.isin(assignments, top_groups)
    pool_idx = np.where(keep_mask)[0]  # indices into the original M
    if len(pool_idx) < 2:
        # Edge case: fall back to original pool
        pool_idx = np.arange(M)

    # --- Fine step on refined pool ---
    Z_fine = Z[pool_idx]
    rel_fine = relevance[pool_idx]
    rel_fine_norm = normalize_minmax(rel_fine)
    energy_fine = -rel_fine_norm
    C_fine = cost_matrix_cosine(Z_fine).astype(np.float32)
    K_fine = redundancy_kernel(Z_fine).astype(np.float32)
    p0_fine = softmax_np(-energy_fine, tau=fine_cfg["tau0"])

    fine_jko = JKOConfig(
        h=fine_cfg["h"], lam=fine_cfg["lam"], rho=fine_cfg["rho"],
        sinkhorn_eps=fine_cfg["sinkhorn_eps"], T=fine_cfg["T"],
        inner_steps=fine_cfg["inner_steps"], mode="wasserstein",
    )
    p_fine, _ = run_jko(p0_fine, energy_fine, C_fine, K_fine, fine_jko)

    # Map p_fine back to size-M support
    p_full = np.zeros(M, dtype=np.float32)
    p_full[pool_idx] = p_fine

    if return_internal:
        return p_full, pool_idx, {
            "assignments": assignments, "centroids": centroids,
            "p_coarse": p_coarse, "top_groups": top_groups,
        }
    return p_full, pool_idx


# -----------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 32
    # synthesise 500 chunks in 25 clusters of 20 each
    centers = rng.normal(size=(25, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    Z = []
    relevance = []
    for ci, c in enumerate(centers):
        for j in range(20):
            z = c + 0.1 * rng.normal(size=d)
            z /= np.linalg.norm(z)
            Z.append(z)
            # cluster 0 is the relevant cluster (high relevance)
            r = 0.9 - 0.5 * (ci > 0) + 0.05 * rng.normal()
            relevance.append(max(0.0, min(1.0, r)))
    Z = np.stack(Z).astype(np.float32)
    relevance = np.array(relevance, dtype=np.float32)
    M = len(Z)
    print(f"Synthetic test: M={M} chunks, true relevant cluster has indices 0..19")

    import time
    t0 = time.time()
    p, pool = mr_jko(Z, relevance, G=25, G_keep=4)
    t_mr = time.time() - t0
    print(f"MR-JKO  time={t_mr*1000:.0f}ms, refined pool size={len(pool)}, mass on rel cluster (0..19): {p[:20].sum():.3f}")
    print(f"  top-10 chunks: {np.argsort(-p)[:10].tolist()}")

    # Compare to vanilla JKO on full pool
    from jko import JKOConfig, run_jko
    C = cost_matrix_cosine(Z).astype(np.float32)
    K = redundancy_kernel(Z).astype(np.float32)
    p0 = softmax_np(relevance, tau=0.2)
    cfg = JKOConfig(h=2.0, lam=0.1, rho=0.05, sinkhorn_eps=0.2, T=3, inner_steps=25)
    t0 = time.time()
    p_vanilla, _ = run_jko(p0, -relevance, C, K, cfg)
    t_van = time.time() - t0
    print(f"Vanilla JKO time={t_van*1000:.0f}ms, mass on rel cluster: {p_vanilla[:20].sum():.3f}")
    print(f"  top-10 chunks: {np.argsort(-p_vanilla)[:10].tolist()}")
    print(f"Speedup MR vs vanilla: {t_van / max(t_mr, 1e-6):.2f}x")
