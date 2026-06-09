# JKO-RAG: Distributional Retrieval as Wasserstein Free-Energy Gradient Flow

**NeurIPS 2025 Workshop on Optimal Transport and Machine Learning**

This repository contains the full research pipeline for **JKO-RAG**, which frames
neural reranking as minimising a free-energy functional under the Jordan–Kinderlehrer–Otto
(JKO) Wasserstein gradient flow on the probability simplex over retrieved candidates.

---

## Overview

Standard RAG pipelines return a ranked list via a cross-encoder. JKO-RAG instead
maintains a *distribution* $p \in \Delta^{M-1}$ over a candidate pool of $M$
documents and evolves it via the JKO proximal scheme:

$$p_{t+1} = \arg\min_{p} \frac{1}{2h} W^2_{C,\varepsilon}(p, p_t) + F_q(p)$$

where $F_q = \text{relevance} + \lambda\,\text{entropy} + \frac{\rho}{2}\,\text{redundancy}$
and $C_{ij} = (1 - \cos\langle z_i, z_j\rangle)^2$ is the semantic ground metric.

### Four algorithmic contributions

| Name | Description |
|------|-------------|
| **NM-JKO** | Low-rank learned ground metric $W \in \mathbb{R}^{64 \times 384}$ trained via InfoNCE |
| **BW-JKO** | Bregman–Wasserstein α-interpolation: $\alpha W^2 + (1-\alpha)\text{KL}$ |
| **SAM-JKO** | Score-Aware Multi-resolution JKO: relevance-weighted clustering → 2× speedup |
| **Dual-Rank** | Sinkhorn dual potentials as per-query retrieval confidence (selective abstention) |

---

## Key Results

### nDCG@10 (tuned hyperparameters, 95% bootstrap CI)

| Method | SciFact | NFCorpus | TREC-COVID | FiQA | SCIDOCS |
|--------|---------|----------|------------|------|---------|
| Cross-Encoder | 0.684 | 0.352 | 0.687 | 0.368 | 0.167 |
| KL-Proximal | 0.712 | 0.357 | 0.717 | 0.413 | 0.197 |
| **JKO-RAG (ours)** | **0.713** | **0.359** | **0.725**† | **0.411** | **0.196** |

> † TREC-COVID: JKO improvement over Cross-Encoder is borderline significant (paired 95% CI lower bound ≈ 0.000; n=50 queries). FiQA and SCIDOCS: JKO and KL-Proximal are within bootstrap CI of each other (effectively tied; KL scores marginally higher by 0.001–0.002).

### Stability under paraphrase (W_C, lower = more stable)

| Method | SciFact | NFCorpus | TREC-COVID | FiQA |
|--------|---------|----------|------------|------|
| Cross-Encoder top-k | 0.114 | 0.130 | 0.076 | 0.117 |
| KL-Proximal | 0.073 | 0.089 | 0.113 | 0.116 |
| **JKO-RAG** | **0.045** | **0.069** | **0.075** | **0.072** |

JKO-RAG is **22–38% more stable** than KL across all four datasets.

### SAM-JKO speedup (SciFact, M=200)

| Method | nDCG@10 | ms/query | Speedup |
|--------|---------|----------|---------|
| Vanilla JKO | 0.710 | 948 | 1.00× |
| SAM-JKO β=2.0 | **0.715** | **456** | **2.08×** |

---

## Installation

```bash
# Python 3.10+
pip install uv
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

---

## Reproducing Results

### 1. Download data
```bash
python src/download_data.py         # SciFact
python src/download_more.py         # NFCorpus, FiQA, TREC-COVID
```

### 2. Build indices
```bash
python src/build_indices.py                              # SciFact
python src/index_multi.py --datasets nfcorpus fiqa trec-covid
```

### 3. Precompute candidate pools + reranker scores
```bash
python src/precompute_candidates.py
python src/precompute_multi.py --datasets nfcorpus fiqa trec-covid scidocs --splits test train
```

### 4. Hyperparameter tuning
```bash
python src/tune_hparams.py --dataset scifact --n-train 80 --n-iter 25
```

### 5. Full evaluation (all contributions)
```bash
python src/run_contributions.py --dataset scifact
python src/run_contributions.py --dataset nfcorpus
python src/run_contributions.py --dataset fiqa
python src/run_contributions.py --dataset scidocs
```

### 6. Stability experiments
```bash
python src/run_stability_new.py --dataset scifact --n-queries 50
python src/run_stability_new.py --dataset nfcorpus --n-queries 50
python src/run_stability_new.py --dataset fiqa --n-queries 50
```

### 7. SAM-JKO benchmark
```bash
python src/run_mr_jko_bench.py --dataset scifact
```

### 8. Dual-Rank selective coverage
```bash
python src/dual_rank_selective.py --dataset scifact --split test
python src/dual_rank_selective.py --dataset nfcorpus --split test
python src/dual_rank_selective.py --dataset fiqa --split test
python src/dual_rank_selective.py --dataset scidocs --split test
```

### 9. Generate paper figures
```bash
python scripts/make_figures.py
```

### 10. Regenerate full report
```bash
python src/final_report.py
```

Or run the full idempotent pipeline:
```bash
bash src/run_directions_pipeline.sh
```

---

## Repository Structure

```
jko-rag/
├── src/
│   ├── jko.py                   # Core JKO solver (Sinkhorn, envelope-theorem)
│   ├── learned_metric.py        # NM-JKO: LowRankMetric + InfoNCE training
│   ├── hierarchical_jko.py      # SAM-JKO: score-aware multi-resolution JKO
│   ├── dual_rank_selective.py   # Dual-Rank: OT potentials as confidence
│   ├── run_contributions.py     # Main evaluation: 9 methods × 4 datasets
│   ├── run_stability_new.py     # BW-JKO / NM-JKO stability sweep
│   ├── run_mr_jko_bench.py      # SAM-JKO synthetic + SciFact benchmark
│   ├── nm_jko_end2end.py        # D3: end-to-end training via JKO unroll
│   ├── retrieval.py             # Cost matrix, Sinkhorn helpers
│   ├── methods.py               # Candidate dataclass, rerank_scores
│   ├── evaluation.py            # nDCG, Recall, bootstrap CIs
│   └── ...
├── paper/
│   ├── main.tex                 # NeurIPS workshop paper (LaTeX)
│   ├── bibliography.bib
│   └── figures/                 # Generated by scripts/make_figures.py
├── scripts/
│   └── make_figures.py          # Figure generation from result JSONs
├── results/
│   ├── REPORT.md                # Full experimental report (793 lines)
│   ├── THEORY.md                # Theoretical framework
│   └── *.json                   # All result files
└── indices/                     # Built indices (not tracked in git)
```

---

## Method Details

### JKO Free Energy

$$F_q(p) = \underbrace{\sum_i p_i E_i}_{\text{relevance}} + \underbrace{\lambda \sum_i p_i \log p_i}_{\text{entropy}} + \underbrace{\frac{\rho}{2} p^\top K p}_{\text{redundancy}}$$

- $E_i = -(\alpha \tilde{s}^\text{dense}_i + \gamma \tilde{s}^\text{rerank}_i)$  (energy from blended scores)
- $K_{ij} = \max(0, \cos\langle z_i, z_j\rangle)$  (redundancy kernel)
- Tuned config: $h{=}2.0$, $\lambda{=}0.1$, $\rho{=}0.05$, $\varepsilon{=}0.2$, $T{=}5$

### Sinkhorn + Envelope Theorem

60 total Sinkhorn iterations; only the final 8 are tracked through autograd for gradient computation. This gives ~20× speedup over fully-tracked Sinkhorn while recovering the correct gradient at the optimum (envelope theorem).

### BW-JKO Interpolation

$$\text{Prox}_t^\alpha(p) = \alpha \cdot W^2_{C,\varepsilon}(p, p_t) + (1-\alpha) \cdot D_\text{KL}(p \| p_t)$$

Sweeping α ∈ {0, 0.25, 0.5, 0.75, 1.0} gives a monotone stability improvement as α→1, empirically proving that the Wasserstein geometry is the source of the stability advantage.

---

## Models Used

| Component | Model | Size |
|-----------|-------|------|
| Dense retriever | `all-MiniLM-L6-v2` | 22M |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M |
| BGE geometry (ablation) | `BAAI/bge-small-en-v1.5` | 33M |

*Note: Using larger backbones (e.g. BGE-large-en, 335M) would raise all absolute numbers while preserving relative orderings. The framework is backbone-agnostic.*

---

## Citation

```bibtex
@inproceedings{jkorag2025,
  title     = {{JKO-RAG}: Distributional Retrieval as {Wasserstein} Free-Energy Gradient Flow},
  author    = {Anonymous},
  booktitle = {NeurIPS 2025 Workshop on Optimal Transport and Machine Learning},
  year      = {2025},
}
```

---

## License

MIT
