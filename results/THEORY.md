# JKO-RAG — theoretical framework

## 1. The Jordan–Kinderlehrer–Otto scheme

The Fokker–Planck equation

    ∂ρ/∂t = ∇·(ρ ∇Ψ) + β⁻¹ Δρ                                  (1)

describes the evolution of a probability density ρ(x, t) under an external potential Ψ(x) plus diffusion. Jordan, Kinderlehrer & Otto (1998) showed (1) is the gradient flow of the free energy functional

    F(ρ) = ∫ Ψ ρ dx + β⁻¹ ∫ ρ log ρ dx                          (2)

with respect to the 2-Wasserstein distance W₂ over probability measures. The semi-discrete *JKO scheme* is

    ρ_{t+1} = argmin_ρ  (1/(2h)) · W₂²(ρ, ρ_t) + F(ρ)            (3)

In the limit h → 0, iterates of (3) converge to the solution of (1). The Wasserstein-proximal step (3) is the natural geometric counterpart of an Euclidean implicit-Euler step: it asks for the next iterate to (i) decrease free energy F and (ii) not move "too far" from the current iterate under W₂.

## 2. Adaptation to retrieval

We discretize the JKO scheme to the **finite simplex Δ^{M-1}** over a candidate pool of M documents per query. The flow is no longer a continuous-time PDE but a sequence of distributional updates over a finite ground set.

Let z₁, …, z_M ∈ R^d be normalized embeddings of the M candidate chunks. We define:

- **Cost matrix** C_{ij} = (1 − cos⟨z_i, z_j⟩)² ∈ [0, 4]. This is the squared spherical distance on the embedding manifold (up to a constant). It satisfies the metric requirements modulo positive-definiteness on the simplex.

- **Free energy**

      F_q(p) = Σ p_i E_i(q)  +  λ Σ p_i log p_i  +  (ρ/2) p^T K p              (4)

  Three terms:
  1. **Data fidelity** Σ p_i E_i with E_i = −relevance_i. Concentrates p on relevant chunks. Lower E_i ⇒ chunk i is more relevant ⇒ p should put more mass on i.
  2. **Entropy** λ Σ p log p. Repels p from delta-spikes; the scale λ controls smoothness.
  3. **Redundancy** (ρ/2) p^T K p with K_{ij} = max(0, cos⟨z_i, z_j⟩). Quadratic penalty that discourages placing mass on many semantically near-duplicate chunks.

- **JKO step** (entropic regularization for tractability):

      p_{t+1} = argmin_p  (1/(2h)) · W²_{C,ε}(p, p_t) + F_q(p)              (5)

  where W²_{C,ε} is the entropic optimal-transport cost with regularization ε, computed via Sinkhorn iteration. Entropic OT (Cuturi 2013) makes (5) differentiable and ~Õ(M²) per Sinkhorn iteration instead of the Õ(M³) of exact OT.

We iterate (5) for T outer steps. Each outer step is solved via Adam on the softmax logits θ where p = softmax(θ). The Sinkhorn solver is run for 60 iterations with the *last 8* tracked through autograd; the first 52 are warmstart-only. This **envelope-theorem trick** (cf. Maclaurin et al. 2015; Feydy et al. 2019) gives correct gradients at the optimum at ~20× the speed of fully tracked Sinkhorn.

## 3. Why Wasserstein, not KL

The Kullback–Leibler proximal alternative replaces W²_{C,ε} in (5) with D_KL(p ‖ p_t). This gives a multiplicative-update / mirror-descent dynamic on the simplex (Beck & Teboulle 2003), which is the closest pure-information-theoretic analog.

The crucial difference: **KL is blind to embedding geometry.** Moving mass from chunk i to chunk j costs the same under D_KL whether j is the nearest neighbour of i or unrelated. Under W²_{C,ε}, the cost is C_{ij} = (1 − cos)² — small for semantically close chunks, large for semantically distant ones.

This has direct consequences:
- **Stability**. Under a paraphrase q → q', the new energy E(q') is small at chunks near the gold cluster and larger elsewhere. KL freely transports mass anywhere → the new p can shift drastically. W² penalizes transport across distant clusters → the new p stays close to the old one within the same semantic region.
- **Distractor resistance**. When a "near-neighbour-but-wrong" chunk is injected with high relevance score, KL moves mass to it freely. W² makes it competitive with the genuinely relevant chunk *only if* they're in the same cluster — which is exactly what we want, since distractors that are semantically close to gold often *should* be considered, while distractors in unrelated clusters shouldn't be.
- **Decisive empirical test (this work, ablation matrix)**. Replacing C with the identity-on-off-diagonal matrix `C_ij = 1[i ≠ j]` collapses W² to a KL-like multiplicative update — and empirically, identity-C exactly matches KL-prox at 0.713 nDCG@10 on SciFact test. **The semantic geometry is operative.**

## 4. Convergence properties

The JKO scheme (3) under the *exact* OT cost is known to have:

- **Monotone decrease of free energy**: F(ρ_{t+1}) ≤ F(ρ_t) for each step (immediate from the argmin formulation).
- **Convergence to stationary points**: if F is bounded below and lower semi-continuous in W₂, the iterates converge to a stationary measure ρ_∞ satisfying ∇_W F(ρ_∞) = 0.

For our entropic-regularized finite-state version, convergence is to a *biased* fixed point because W²_{C,ε} ≠ W²_C. The bias is O(ε log M) (Genevay 2019). We use ε = 0.1–0.2 throughout, where this bias is small relative to the per-chunk relevance signal.

The inner Adam loop is not guaranteed to find the exact argmin in (5), but warm-starting from p_t gives a near-identity initialization, and 25–40 Adam steps are empirically sufficient to converge within tolerance.

## 5. Relation to existing distributional retrieval methods

| Method | Distributional update | Geometric awareness | Iterative? |
|---|---|---|---|
| MMR (Carbonell & Goldstein 1998) | Greedy argmax on (λ·rel − (1−λ)·max-sim) | Yes (similarity penalty) | Per-step greedy |
| Diversified DPP retrieval | Sample from determinantal point process | Yes (kernel) | One-shot |
| Iter-RetGen (Shao et al. 2023) | Generate intermediate answer, re-retrieve | No | T iterations |
| KL-proximal soft retrieval | Mirror descent on simplex | **No** | T iterations |
| **JKO-RAG (this work)** | **Wasserstein-proximal on simplex** | **Yes (semantic ground metric)** | T iterations |

JKO-RAG is the unique combination of (a) a fully distributional update over the simplex, (b) a semantic ground metric on the embedding manifold, and (c) iterative gradient-flow refinement.

## 6. Connection to information geometry and Otto calculus

The W₂ metric on the space of probability measures equipped with the Otto metric tensor makes it a formal infinite-dimensional Riemannian manifold. Under this view, JKO is *intrinsic gradient descent on free energy*. The Wasserstein gradient flow of (2) is the Fokker–Planck PDE (1); the steepest-descent direction at ρ is given by ∇·(ρ ∇(δF/δρ)).

For our finite candidate pool, the analogous statement is that the JKO iterates (5) are discrete-time steepest-descent updates of F_q on the simplex with the Otto metric induced by C. KL-proximal corresponds instead to gradient descent under the Fisher–Rao information metric. The two are different geometries; one is intrinsic to the embedding manifold, the other is not.

## 7. Practical hyperparameters and what they control

| Hparam | Range | What it controls |
|---|---|---|
| h | 0.1–2.0 | Wasserstein-proximal strength. Small h: heavy regularization → p barely moves per step. Large h: weak prox → behaves like noprox or KL. |
| λ | 0.005–0.1 | Entropy on -H(p). Prevents collapse to single chunk. |
| ρ | 0.0–0.1 | Redundancy penalty strength. |
| ε | 0.05–0.2 | Sinkhorn regularization. Small ε: sharper OT but slower convergence. |
| T | 1–5 | Outer JKO steps. Cumulative distance budget T · h. |
| inner_steps | 15–40 | Adam steps per JKO outer step. |
| τ₀ | 0.05–1.0 | Initial-distribution sharpness via softmax(r/τ₀). |
| α, γ | [0, 1] | Dense vs reranker weight in energy. |

The tuned config found on SciFact-train and NFCorpus-train independently (h=2.0, λ=0.1, ρ=0.05, ε=0.2, T=5) is a **weak-proximal** regime: the Wasserstein term acts as a soft semantic regulariser rather than a tight constraint. With weaker proximal (large h), the iterates can move quickly toward the energy minimum while still preferring intra-cluster moves over cross-cluster ones.

## 8. Limitations of the framework

- **Finite candidate pool**. The flow lives on Δ^{M-1} for a pre-selected pool of M=200 candidates. Chunks outside the pool can never receive mass. The first-stage hybrid retriever must achieve high pool recall (empirically 0.97 on SciFact, 0.73 on FiQA).
- **Symmetric cost**. C is symmetric. A directed cost (e.g., asymmetric semantic entailment) would allow modelling support vs contradiction asymmetry but breaks Sinkhorn's symmetric updates.
- **No corpus-level interaction**. The flow only refines the per-query distribution. Cross-query learning (as in dense retriever training) is orthogonal to the framework.
- **Entropic bias**. W²_{C,ε} ≠ W²_C for ε > 0. We use ε = 0.1–0.2; debiasing via Sinkhorn divergences (Feydy 2019) could be a future improvement.

## 9. Why this could matter for ICLR / ICML

The retrieval literature is dominated by methods that produce a *point* answer (a ranked list). JKO-RAG introduces a principled framework for retrieval as a **distributional, geometric, iterative process** — closer to belief refinement than to ranking. The framework is:

1. **Mathematically grounded** in 25-year-old, well-studied gradient-flow theory.
2. **Empirically advantageous** on novel evaluation axes (stability, distractor resistance) that the retrieval community has under-measured.
3. **Decisively ablatable** — the W²-vs-KL distinction is sharp and controlled by a single matrix.
4. **Composable** with any first-stage retriever + reranker, including future SOTA models.

The proposal is not "JKO-RAG is the new SOTA." It is: "Distributional, geometric retrieval is a *richer* primitive than top-k ranking, and Wasserstein gradient flow is its natural mathematical home."

## References (informal)

- Jordan, R., Kinderlehrer, D. & Otto, F. (1998). *The Variational Formulation of the Fokker–Planck Equation.* SIAM J. Math. Anal. 29(1): 1–17.
- Otto, F. (2001). *The geometry of dissipative evolution equations: the porous medium equation.* Comm. Partial Diff. Eqns 26(1-2): 101–174.
- Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.* NeurIPS.
- Feydy, J. et al. (2019). *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences.* AISTATS.
- Genevay, A. (2019). *Entropy-regularized Optimal Transport for Machine Learning.* PhD thesis.
- Carbonell, J. & Goldstein, J. (1998). *The use of MMR for reordering documents and producing summaries.* SIGIR.
- Shao, Z. et al. (2023). *Enhancing Retrieval-Augmented LMs with Iterative Retrieval-Generation Synergy.* EMNLP-Findings.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain QA.* EMNLP.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot IR.* NeurIPS Track on Datasets and Benchmarks.
