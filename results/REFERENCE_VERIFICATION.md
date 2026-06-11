# Reference & competitor-number verification

Independent web verification (Google Scholar / DBLP / Semantic Scholar / arXiv /
ACL Anthology / official proceedings) of every cited reference and of the BEIR
nDCG@10 numbers used to position this work. Date: 2026-06-11.

## References: all 24 cited entries are REAL

No fabricated citations. Metadata errors found and **fixed** in `bibliography.bib`:

| Key | Issue found | Fix applied |
|---|---|---|
| `santambrogio2015ot` | Typed `@article` with garbled journal "Birkäuser, NY", vol 55, p 94 | Retyped `@book`, *Progress in Nonlinear Diff. Eqns.* vol **87**, Birkhäuser Cham, 2015 |
| `wang2023e5` (E5) | `@inproceedings` w/ arXiv booktitle; author "Xinlong" | `@misc` arXiv:2212.03533 (2022); corrected to **Xiaolong** Huang |
| `nogueira2019mono` | `@inproceedings` w/ arXiv booktitle | `@misc` arXiv:1901.04085 (2019) — arXiv-only, never venue-published |
| `genevay2019` | School "Université Paris Dauphine" | "Université PSL (Paris-Dauphine)" — PSL is the degree-granting body |
| `chen2022dpp` | Key year wrong (really NeurIPS **2018**) | Key renamed `chen2018dpp` (year field was already 2018) |
| `wangscifact2020` | Key implies first author "Wang"; really **Wadden** | Key renamed `wadden2020scifact` (author field was already correct) |
| `wilson2014spectral` | Added speculatively, never cited, irrelevant | Removed |

All other entries (`jko1998`, `cuturi2013sinkhorn`, `feydy2019aistats`,
`ambrosio2008gradient`, `luise2018differential`, `eisenberger2022unified`,
`maclaurin2015grad`, `beck2003mirror`, `cohen2019certified`, `carbonell1998mmr`,
`thakur2021beir`, `shao2023iter`, `karpukhin2020dpr`, `kusner2015wmd`,
`reimers2019sbert`, `rrf2009`, `lewis2020rag`, `santambrogio2017euclidean`)
verified correct as written (authors / venue / year all match the primary source).

## BEIR nDCG@10 competitor numbers (primary-sourced)

Used in the "Positioning against the BEIR landscape" paragraph. Verified against
the BEIR paper Table 2 (arXiv:2104.08663), model papers, and MTEB raw results.

| Dataset | BM25 (BEIR paper) | Strong systems | Best / SOTA | SOTA source |
|---|---|---|---|---|
| SciFact | 0.665 | ColBERTv2 0.693 · BGE-large 0.746 · E5-large 0.726 | **0.777** monoT5-3B | Rosa et al. 2206.02873 T1 |
| NFCorpus | 0.325 | BGE-large 0.381 · E5-large 0.361 | **~0.40** InPars-v2 0.399 / monoT5-3B 0.384 | InPars-v2 2301.01820 T1 |
| TREC-COVID | 0.656 | BGE-large 0.747 · E5-large 0.783 · monoT5-3B 0.795 | **~0.82–0.85** InPars-v2 0.823 / RankT5 ~0.846 | InPars-v2 2301.01820 T1 |
| FiQA | 0.236 | GTR-XXL 0.467 · BGE-large 0.450 | **0.514** monoT5-3B | Rosa et al. 2206.02873 T1 |
| SCIDOCS | 0.158 | BGE-large 0.226 · E5-large 0.201 | **0.226** BGE-large-en-v1.5 | MTEB raw results |

Notes / corrections to first-pass (from-memory) claims:
- BM25 baselines were exactly right (BEIR-paper Anserini multifield numbers).
- **TREC-COVID SOTA is achieved by cross-encoder rerankers, NOT E5-large**
  (E5-large tops out at 0.783; the 0.82–0.85 ceiling is InPars-v2 / RankT5-3B).
- NFCorpus ceiling is ~0.40 (rerankers), a touch higher than first stated (~0.38).
- SCIDOCS: rerankers do *not* help (monoT5-3B 0.197 < BGE-large 0.226); dense
  embedders win this dataset.

**Conclusion.** JKO-RAG's MiniLM-backbone scores are below large-model SOTA, as
expected and as now stated explicitly in the paper. JKO-RAG is a
backbone-agnostic framework; the contribution is the geometry + stability theory,
not the absolute nDCG.
