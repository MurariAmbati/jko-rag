"""Shared utilities for the JKO-RAG stability-theory verification experiments.

These experiments test the central theoretical claim of the paper: that the
Wasserstein proximal makes the JKO retrieval map *geometrically* more stable
than the KL proximal, because the entropic-OT proximal Hessian penalises
cross-cluster mass transport while the KL proximal Hessian is diagonal
(geometry-blind).

Everything here is wired to match the *existing* stability protocol used in
`run_stability_multi.py` so the new numbers are directly comparable to the
paper's stability table:

  - rerank-only energy:  energy = -minmax(rerank)        (alpha_e=0, gamma_e=1)
  - initial distribution: p0 = softmax(minmax(rerank), tau=0.1)
  - cost matrix:          C_ij = (1 - cos<z_i,z_j>)^2
  - redundancy kernel:    K_ij = max(0, cos<z_i,z_j>)
  - W_C metric:           entropic OT loss, eps=0.1, n_iter=80  (same as
                          w_distance_on_full_pool, but on a *fixed* pool)

No query re-encoding is required: we perturb the *energy vector* under a
controlled norm, which is exactly the object the single-step linear-response
theorem characterises. The candidate pool is held fixed, isolating the
energy-landscape-shift component of paraphrase sensitivity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from jko import JKOConfig, jko_step, log_sinkhorn_loss
from retrieval import (
    cost_matrix_cosine,
    redundancy_kernel,
    softmax_np,
    normalize_minmax,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = ROOT / "indices"
RESULTS_DIR = ROOT / "results"

# Base stability config (matches run_stability_multi.jko_distribution).
BASE_CFG = dict(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1, T=3, inner_steps=25)


# -----------------------------------------------------------------------------
# Data loading (cached candidate pools; no model needed)
# -----------------------------------------------------------------------------
def _index_dir(dataset: str) -> Path:
    sub = INDEX_ROOT / dataset
    return sub if (sub / "doc_ids.json").exists() else INDEX_ROOT


def load_pools(dataset: str = "scifact") -> dict:
    """Load cached test-split candidate pools + embeddings + qrels.

    Returns a dict with:
        cand_idx (Q, M) int   - corpus indices per query
        rerank   (Q, M) float - cross-encoder scores on the pool
        dense    (Q, M) float - dense cosine scores on the pool
        emb      (N, d) float - L2-normalised corpus embeddings
        doc_ids  list[str]
        qids     list[str]    - query ids aligned with rows of cand_idx
        qrels    dict[qid -> dict[doc_id -> rel]]
    """
    base = _index_dir(dataset)
    cache_path = base / "candidates_test.npz"
    if not cache_path.exists():
        cache_path = INDEX_ROOT / "candidates_test.npz"
    cache = np.load(cache_path, allow_pickle=True)
    emb = np.load(base / "embeddings.npy")
    with open(base / "doc_ids.json") as f:
        doc_ids = json.load(f)

    # qrels via data_loader (test split)
    from data_loader import load_dataset
    ds = load_dataset(dataset)
    qrels = ds.qrels.get("test", {})

    return {
        "dataset": dataset,
        "cand_idx": cache["cand_idx"],
        "rerank": cache["rerank"],
        "dense": cache["dense_pool"],
        "emb": emb,
        "doc_ids": doc_ids,
        "qids": [str(x) for x in cache["q_ids"]],
        "qrels": qrels,
    }


@dataclass
class QueryInputs:
    cand: np.ndarray   # (M,) corpus indices
    Z: np.ndarray      # (M, d) embeddings
    C: np.ndarray      # (M, M) cost matrix
    K: np.ndarray      # (M, M) redundancy kernel
    energy: np.ndarray # (M,) energy = -relevance
    p0: np.ndarray     # (M,) initial distribution
    qid: str


def build_inputs(pools: dict, ci: int, alpha_e: float = 0.0,
                 gamma_e: float = 1.0, tau0: float = 0.1) -> QueryInputs:
    """Build (C, K, energy, p0) for query-row ci, matching the stability protocol."""
    cand = pools["cand_idx"][ci]
    Z = pools["emb"][cand]
    C = cost_matrix_cosine(Z)
    K = redundancy_kernel(Z)
    r = alpha_e * normalize_minmax(pools["dense"][ci]) + gamma_e * normalize_minmax(pools["rerank"][ci])
    energy = (-r).astype(np.float32)
    p0 = softmax_np(r, tau=tau0).astype(np.float32)
    return QueryInputs(cand=cand, Z=Z, C=C, K=K, energy=energy, p0=p0, qid=pools["qids"][ci])


# -----------------------------------------------------------------------------
# Free energy and the W_C stability metric
# -----------------------------------------------------------------------------
def free_energy(p: np.ndarray, energy: np.ndarray, K: np.ndarray,
                lam: float, rho: float) -> float:
    """F(p) = <p,E> + lam <p, log p> + (rho/2) p^T K p  (the JKO objective sans prox)."""
    p = np.clip(p, 1e-30, None)
    data = float(p @ energy)
    ent = float(lam * (p * np.log(p)).sum())
    red = float(0.5 * rho * (p @ (K @ p)))
    return data + ent + red


def wc(p: np.ndarray, q: np.ndarray, C: np.ndarray,
       eps: float = 0.1, n_iter: int = 80) -> float:
    """Entropic-OT distance W_C(p, q) on a FIXED pool (same metric as stability table)."""
    pt = torch.tensor(np.clip(p, 1e-12, None), dtype=torch.float32)
    qt = torch.tensor(np.clip(q, 1e-12, None), dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)
    with torch.no_grad():
        w = log_sinkhorn_loss(torch.log(pt), torch.log(qt), Ct, eps=eps, n_iter=n_iter)
    return float(w)


# -----------------------------------------------------------------------------
# JKO flow (with optional per-outer-step trace)
# -----------------------------------------------------------------------------
def make_cfg(h: float, T: int, mode: str, **over) -> JKOConfig:
    d = dict(BASE_CFG)
    d.update(over)
    d["h"] = h
    d["T"] = T
    d["mode"] = mode
    return JKOConfig(**d)


def flow(energy: np.ndarray, C: np.ndarray, K: np.ndarray, p0: np.ndarray,
         cfg: JKOConfig, device: str = "cpu") -> np.ndarray:
    """Run T JKO outer steps, return p_T. (Thin wrapper to keep call sites uniform.)"""
    p = torch.tensor(p0, dtype=torch.float32, device=device)
    e = torch.tensor(energy, dtype=torch.float32, device=device)
    Ct = torch.tensor(C, dtype=torch.float32, device=device)
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    for _ in range(cfg.T):
        p, _ = jko_step(p, e, Ct, Kt, cfg)
    return p.cpu().numpy()


def flow_trace(energy: np.ndarray, C: np.ndarray, K: np.ndarray, p0: np.ndarray,
               cfg: JKOConfig, device: str = "cpu") -> list[np.ndarray]:
    """Run T JKO outer steps, return [p_0, p_1, ..., p_T] (intermediates included)."""
    p = torch.tensor(p0, dtype=torch.float32, device=device)
    e = torch.tensor(energy, dtype=torch.float32, device=device)
    Ct = torch.tensor(C, dtype=torch.float32, device=device)
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    traj = [p.cpu().numpy().copy()]
    for _ in range(cfg.T):
        p, _ = jko_step(p, e, Ct, Kt, cfg)
        traj.append(p.cpu().numpy().copy())
    return traj


# -----------------------------------------------------------------------------
# Graph-Laplacian frequency bands of the OT geometry
# -----------------------------------------------------------------------------
def laplacian_spectrum(C: np.ndarray, eps: float = 0.1):
    """Symmetric normalised Laplacian of the Gibbs affinity Gamma = exp(-C/eps).

    Returns (evals, U) with evals ascending in [0, 2]. Low eval = smooth /
    intra-cluster mode; high eval = oscillatory / cross-cluster mode. The
    eigenvectors form an orthonormal frequency basis for energy perturbations.
    """
    Gamma = np.exp(-C / eps).astype(np.float64)
    np.fill_diagonal(Gamma, 0.0)            # no self-affinity
    deg = Gamma.sum(axis=1)
    deg = np.clip(deg, 1e-12, None)
    dinv = 1.0 / np.sqrt(deg)
    Lsym = np.eye(len(C)) - (dinv[:, None] * Gamma * dinv[None, :])
    Lsym = 0.5 * (Lsym + Lsym.T)            # symmetrise numerical noise
    evals, U = np.linalg.eigh(Lsym)
    return evals, U


def band_slices(M: int, n_bands: int) -> list[tuple[int, int]]:
    """Partition eigenvector indices [0, M) into n_bands contiguous frequency bands."""
    edges = np.linspace(0, M, n_bands + 1).astype(int)
    return [(int(edges[b]), int(edges[b + 1])) for b in range(n_bands)]


def random_band_perturbation(U: np.ndarray, lo: int, hi: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Unit-norm, mean-zero energy perturbation drawn from eigenvectors [lo:hi)."""
    k = hi - lo
    coef = rng.standard_normal(k)
    dE = U[:, lo:hi] @ coef
    dE = dE - dE.mean()                     # project to tangent of the simplex
    nrm = np.linalg.norm(dE)
    if nrm < 1e-12:
        return np.zeros_like(dE)
    return (dE / nrm).astype(np.float32)


# -----------------------------------------------------------------------------
# Convenience: select a reproducible subset of queries
# -----------------------------------------------------------------------------
def select_queries(pools: dict, n: int, seed: int = 0) -> list[int]:
    """Return row-indices of n queries that have at least one relevant doc in qrels."""
    qids = pools["qids"]
    qrels = pools["qrels"]
    eligible = [i for i, q in enumerate(qids)
                if q in qrels and any(v > 0 for v in qrels[q].values())]
    rng = np.random.default_rng(seed)
    if n >= len(eligible):
        return eligible
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())
