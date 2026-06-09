"""JKO-RAG step solver.

Solves, for each retrieval refinement step:

    p_{t+1} = argmin_{p in simplex}  (1/2h) * Prox(p, p_t)
                                     + sum_i p_i E_i
                                     + lambda * sum_i p_i log p_i
                                     + (rho/2) * p^T K p

The proximal operator Prox(p, p_t) is one of:

  - "wasserstein": entropic-OT cost W^2_{C,eps}(p, p_t), via log-domain Sinkhorn.
  - "kl":          D_KL(p || p_t), classical mirror descent / Bregman proximal.
  - "noproximal":  no proximal term (gradient flow in plain Euclidean geometry).
  - "bregman":     CONVEX COMBINATION alpha_prox * W^2 + (1-alpha_prox) * KL.
                   This is a novel interpolation that lets us continuously
                   morph between KL (alpha=0) and Wasserstein (alpha=1).
                   alpha_prox is exposed as a JKOConfig hyperparameter.

The log-domain Sinkhorn solver also exposes its DUAL VARIABLES (f, g):
  - f_i is the "potential at source i" (related to how "expensive" it is to send mass out of i)
  - g_j is the "potential at target j"
These dual potentials are calibrated proxies for per-document confidence; we
expose them via `log_sinkhorn_with_duals` and use them in DUAL-RANK
(see src/dual_rank.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class JKOConfig:
    h: float = 0.5            # outer step size (proximal weight)
    lam: float = 0.05         # entropy coefficient on -H(p)
    rho: float = 0.05         # redundancy penalty coefficient
    sinkhorn_eps: float = 0.1 # entropic OT regularization
    sinkhorn_iter: int = 60   # log-domain Sinkhorn iterations (total)
    sinkhorn_iter_grad: int = 8 # of which, last K are autograd-tracked
    inner_steps: int = 25     # Adam steps per JKO outer step
    inner_lr: float = 0.1     # Adam lr on the softmax logits
    T: int = 3                # number of outer JKO iterations
    mode: str = "wasserstein" # "wasserstein" | "kl" | "noproximal" | "bregman"
    alpha_prox: float = 0.5   # interpolation for mode="bregman": alpha*W^2 + (1-alpha)*KL


def log_sinkhorn_loss(
    log_p: torch.Tensor,        # (M,) log of source distribution
    log_q: torch.Tensor,        # (M,) log of target distribution
    C: torch.Tensor,            # (M, M) cost matrix, nonnegative
    eps: float = 0.1,
    n_iter: int = 60,
    n_iter_grad: int = 3,
) -> torch.Tensor:
    """Entropic OT cost <T, C> with marginals p and q, computed in log-domain.

    For speed, the first (n_iter - n_iter_grad) Sinkhorn updates run with no
    autograd graph (warmstart). The last n_iter_grad iterations are tracked
    so gradients flow into log_p / log_q. This is much faster than backpropping
    through all iterations and is the standard trick for differentiable Sinkhorn
    (envelope-theorem style).
    """
    loss, _f, _g = _log_sinkhorn_core(log_p, log_q, C, eps, n_iter, n_iter_grad)
    return loss


def log_sinkhorn_with_duals(
    log_p: torch.Tensor, log_q: torch.Tensor, C: torch.Tensor,
    eps: float = 0.1, n_iter: int = 60, n_iter_grad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as log_sinkhorn_loss but ALSO returns the dual potentials (f, g).

    By Sinkhorn duality, f_i ≈ eps * (log p_i - log a_i) where a_i is the
    normalizing constant in the optimal coupling row i. Equivalently:

        f_i = -eps * log (sum_j exp((g_j - C_ij) / eps))

    so f_i is large when row i has lots of low-cost targets to transport to.
    We expose f and g for use in DUAL-RANK (per-document confidence).
    """
    return _log_sinkhorn_core(log_p, log_q, C, eps, n_iter, n_iter_grad)


def _log_sinkhorn_core(log_p, log_q, C, eps, n_iter, n_iter_grad):
    M = C.shape[0]
    minus_C_over_eps = -C / eps
    f = torch.zeros(M, device=C.device, dtype=C.dtype)
    g = torch.zeros(M, device=C.device, dtype=C.dtype)

    n_warm = max(0, n_iter - n_iter_grad)
    with torch.no_grad():
        log_p_d = log_p.detach()
        log_q_d = log_q.detach()
        for _ in range(n_warm):
            f = eps * (log_p_d - torch.logsumexp(minus_C_over_eps + (g / eps)[None, :], dim=1))
            g = eps * (log_q_d - torch.logsumexp(minus_C_over_eps + (f / eps)[:, None], dim=0))
    for _ in range(n_iter_grad):
        f = eps * (log_p - torch.logsumexp(minus_C_over_eps + (g / eps)[None, :], dim=1))
        g = eps * (log_q - torch.logsumexp(minus_C_over_eps + (f / eps)[:, None], dim=0))
    log_T = (f[:, None] + g[None, :] - C) / eps
    T = torch.exp(log_T)
    loss = (T * C).sum()
    return loss, f.detach(), g.detach()


def jko_step(
    p_prev: torch.Tensor,   # (M,) distribution at time t
    energy: torch.Tensor,   # (M,) per-chunk energy E_i (lower = more relevant)
    C: torch.Tensor,        # (M, M) cost matrix
    K: torch.Tensor,        # (M, M) redundancy kernel
    cfg: JKOConfig,
    init_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Solve one JKO proximal step. Returns (p_next, info)."""
    M = p_prev.shape[0]
    device = p_prev.device
    dtype = p_prev.dtype

    log_p_prev = torch.log(p_prev.clamp_min(1e-30))

    if init_logits is None:
        theta = torch.log(p_prev.clamp_min(1e-30)).clone().detach()
    else:
        theta = init_logits.clone().detach()
    theta.requires_grad_(True)

    optimizer = torch.optim.Adam([theta], lr=cfg.inner_lr)

    info = {"losses": []}
    for step in range(cfg.inner_steps):
        optimizer.zero_grad()
        log_p = F.log_softmax(theta, dim=0)
        p = log_p.exp()

        loss = (p * energy).sum()
        loss = loss + cfg.lam * (p * log_p).sum()
        loss = loss + 0.5 * cfg.rho * (p @ (K @ p))

        if cfg.mode == "wasserstein":
            w2 = log_sinkhorn_loss(
                log_p, log_p_prev, C,
                eps=cfg.sinkhorn_eps,
                n_iter=cfg.sinkhorn_iter,
                n_iter_grad=cfg.sinkhorn_iter_grad,
            )
            loss = loss + (1.0 / (2.0 * cfg.h)) * w2
        elif cfg.mode == "kl":
            kl = (p * (log_p - log_p_prev)).sum()
            loss = loss + (1.0 / (2.0 * cfg.h)) * kl
        elif cfg.mode == "bregman":
            # NEW: convex interpolation alpha * W^2 + (1 - alpha) * KL
            a = float(cfg.alpha_prox)
            a = max(0.0, min(1.0, a))
            kl = (p * (log_p - log_p_prev)).sum()
            if a > 0.0:
                w2 = log_sinkhorn_loss(
                    log_p, log_p_prev, C,
                    eps=cfg.sinkhorn_eps,
                    n_iter=cfg.sinkhorn_iter,
                    n_iter_grad=cfg.sinkhorn_iter_grad,
                )
                proximal = a * w2 + (1.0 - a) * kl
            else:
                proximal = kl
            loss = loss + (1.0 / (2.0 * cfg.h)) * proximal
        elif cfg.mode == "noproximal":
            pass
        else:
            raise ValueError(f"unknown mode {cfg.mode}")

        loss.backward()
        optimizer.step()
        info["losses"].append(float(loss.detach()))

    with torch.no_grad():
        p_next = F.softmax(theta, dim=0)
    return p_next.detach(), info


def run_jko(
    p0: np.ndarray,
    energy: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    cfg: JKOConfig,
    device: str = "cpu",
) -> tuple[np.ndarray, list[dict]]:
    """Run T JKO outer iterations starting from p0. Returns (p_T, info_per_step)."""
    p = torch.tensor(p0, dtype=torch.float32, device=device)
    e = torch.tensor(energy, dtype=torch.float32, device=device)
    Ct = torch.tensor(C, dtype=torch.float32, device=device)
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    infos: list[dict] = []
    for _ in range(cfg.T):
        p, info = jko_step(p, e, Ct, Kt, cfg)
        infos.append(info)
    return p.cpu().numpy(), infos


def run_jko_with_duals(
    p0: np.ndarray, energy: np.ndarray, C: np.ndarray, K: np.ndarray,
    cfg: JKOConfig, device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run JKO and ALSO return the final-step Sinkhorn dual potentials (f, g).

    The dual variables encode per-chunk "transport potential". Specifically:
       f_i  large  ⇔  source chunk i is hard to move mass out of (high outgoing cost)
       g_j  large  ⇔  target chunk j is hard to receive mass into

    For DUAL-RANK confidence (see src/dual_rank.py), we use:
        confidence(i) = sigmoid((f_i - median(f)) / (mad(f) + eps))
    intuitively: items the OT problem has trouble "displacing" are more
    confidently held by the retrieved distribution.

    Returns (p_T, f, g). If mode != wasserstein/bregman, f and g are zeros.
    """
    p = torch.tensor(p0, dtype=torch.float32, device=device)
    e = torch.tensor(energy, dtype=torch.float32, device=device)
    Ct = torch.tensor(C, dtype=torch.float32, device=device)
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    for _ in range(cfg.T):
        p, _ = jko_step(p, e, Ct, Kt, cfg)

    # One final Sinkhorn iteration to extract duals (no autograd needed)
    if cfg.mode in ("wasserstein", "bregman"):
        log_p = torch.log(p.clamp_min(1e-30))
        # Use uniform as the "target" so duals reflect how p deviates from uniform
        # (alternatively could use p0 or some reference; uniform = uninformative prior)
        log_q = torch.full_like(log_p, -np.log(len(p)))
        _, f, g = log_sinkhorn_with_duals(
            log_p, log_q, Ct,
            eps=cfg.sinkhorn_eps, n_iter=cfg.sinkhorn_iter, n_iter_grad=0,
        )
        return p.cpu().numpy(), f.cpu().numpy(), g.cpu().numpy()
    else:
        M = len(p)
        return p.cpu().numpy(), np.zeros(M, dtype=np.float32), np.zeros(M, dtype=np.float32)


# -----------------------------------------------------------------------------
# Sanity / self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M = 30

    Z = []
    centers = rng.normal(size=(3, 16))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    for c in centers:
        for _ in range(10):
            z = c + 0.1 * rng.normal(size=16)
            z /= np.linalg.norm(z)
            Z.append(z)
    Z = np.stack(Z).astype(np.float32)
    sim = Z @ Z.T
    C = (1.0 - sim) ** 2
    K = np.maximum(sim, 0.0)

    energy = rng.normal(size=M).astype(np.float32)
    energy[:10] -= 1.5

    p0 = np.ones(M) / M
    p0[:10] *= 1.5
    p0 /= p0.sum()

    print("=== JKO modes (cluster mass: c0, c1, c2) ===")
    for mode, extra in [("wasserstein", {}), ("kl", {}), ("noproximal", {}),
                         ("bregman", {"alpha_prox": 0.5}),
                         ("bregman", {"alpha_prox": 0.0}),
                         ("bregman", {"alpha_prox": 1.0})]:
        cfg = JKOConfig(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1, T=3, inner_steps=30,
                         mode=mode, **extra)
        p, _ = run_jko(p0, energy, C, K, cfg)
        cm = [p[:10].sum(), p[10:20].sum(), p[20:].sum()]
        tag = f"{mode}" + (f"(a={extra.get('alpha_prox', 1.0):.1f})" if mode == "bregman" else "")
        print(f"  {tag:<22s} -> [{cm[0]:.3f}, {cm[1]:.3f}, {cm[2]:.3f}], top5={np.sort(p)[-5:].sum():.3f}")

    print("\n=== Dual potentials test (Wasserstein) ===")
    cfg = JKOConfig(h=0.5, lam=0.05, rho=0.05, sinkhorn_eps=0.1, T=3, inner_steps=30)
    p, f, g = run_jko_with_duals(p0, energy, C, K, cfg)
    print(f"  f stats: min={f.min():.3f}, median={np.median(f):.3f}, max={f.max():.3f}")
    print(f"  top-3 by f: chunks {np.argsort(-f)[:3].tolist()}")
    print(f"  top-3 by p: chunks {np.argsort(-p)[:3].tolist()}")
