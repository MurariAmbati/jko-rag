# JKO-RAG results inventory

## Primary outputs
- `REPORT.md` — comprehensive Markdown report (~350 lines, all tables and analysis)
- `stage1_bars.png` — bar charts of nDCG@10, Recall@10/20, diversity per method
- `w_vs_kl.png` — per-query scatter and histogram of W vs KL nDCG@10
- `stability_bars.png` — stability comparison across methods

## Per-dataset Stage 1 (retrieval) results

### Default hyperparameters
- `stage1.json` — SciFact (original run)
- `stage1_nfcorpus.json` — NFCorpus
- `stage1_trec-covid.json` — TREC-COVID
- `stage1_fiqa.json` — FiQA

### Tuned hyperparameters (SciFact-train config, transferred)
- `stage1_scifact_tuned.json`
- `stage1_nfcorpus_tuned.json`
- `stage1_trec-covid_tuned.json`
- `stage1_fiqa_tuned.json`

### Per-dataset tuned (FiQA-own config)
- `stage1_fiqa_fiqatuned.json` — demonstrates overfitting; SciFact-trans is better

## Hyperparameter tuning (random search on train splits)
- `best_hparams_scifact.json` — 25 configs × 80 train queries
- `best_hparams_nfcorpus.json` — 20 configs × 60 train queries (same optimum as SciFact)
- `best_hparams_fiqa.json` — 20 configs × 60 train queries (different, but overfit)

## Stability under query perturbation (Stage 3a)
- `stability.json` — SciFact, 60 queries × 3 perturbations
- `stability_nfcorpus.json` — NFCorpus, 60 queries × 3 perturbations
- `stability_trec-covid.json` — TREC-COVID, 50 queries × 3 perturbations
- `stability_fiqa.json` — FiQA, 60 queries × 3 perturbations

## Distractor-injection robustness (Stage 3b)
- `distractors_scifact.json` — N ∈ {0, 10, 30}, 150 queries
- `distractors_nfcorpus.json` — N ∈ {0, 10, 30}, 100 queries
- `distractors_fiqa.json` — N ∈ {0, 10, 30}, 100 queries

## Generation (Stage 2)
- `stage2_scifact.json` — FLAN-T5-base on 200 untuned claims, 3-way label accuracy

## Ablations
- `ablations_scifact_test.json` — 9-way ablation matrix on SciFact test
- `blend_ablation.json` — W vs KL vs NoProx with identical blended energy

## Reproducibility
- `*.log` — full stdout/stderr for every run

## Indices (under `../indices/`)
- `<dataset>/embeddings.npy` — corpus dense embeddings (L2-normalized)
- `<dataset>/bm25.pkl` — BM25Okapi + tokenized corpus
- `<dataset>/doc_ids.json`, `doc_texts.json` — corpus metadata
- `<dataset>/q_embeddings_<split>.npy`, `q_ids_<split>.json`
- `<dataset>/candidates_<split>.npz` — precomputed (BM25, dense, rerank) scores

## Data (under `../data/`)
- `<dataset>/<dataset>/{corpus,queries}.jsonl` + `qrels/{train,dev,test}.tsv`
- Downloaded from BEIR's public URL at `public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/`
- `scifact_orig/` — original SciFact release with SUPPORT/CONTRADICT/NEI labels (for Stage 2)
