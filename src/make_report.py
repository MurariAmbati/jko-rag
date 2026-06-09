"""Generate a Markdown report from results/stage1.json and stability.json."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def fmt_ci(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"


def fmt_diff(d: dict) -> str:
    sig = "**" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else ""
    return f"{sig}{d['diff']:+.4f}{sig} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]"


def main():
    s1 = json.loads((RESULTS_DIR / "stage1.json").read_text())
    has_stab = (RESULTS_DIR / "stability.json").exists()
    stab = json.loads((RESULTS_DIR / "stability.json").read_text()) if has_stab else None

    lines: list[str] = []
    lines.append("# JKO-RAG: Stage 1 Results on SciFact\n")
    lines.append(f"- Test queries: **{s1['n_queries']}**")
    lines.append(f"- Total runtime for all methods: **{s1['elapsed_sec']:.1f}s** "
                 f"(~{s1['elapsed_sec']/s1['n_queries']:.2f}s/query for all methods combined)")
    lines.append(f"- Candidate pool size M=200, top-k=20, JKO T=3, h=0.5, "
                 f"sinkhorn_eps=0.1, lambda=0.05, rho=0.05")
    lines.append("")
    lines.append("## Method roster")
    descs = {
        "bm25": "BM25 over full corpus (5,183 docs)",
        "dense": "all-MiniLM-L6-v2 dense retrieval over full corpus",
        "hybrid_rrf": "BM25 + dense fused with reciprocal-rank fusion on a M=200 pool",
        "rerank": "hybrid pool → cross-encoder (ms-marco-MiniLM-L-6-v2) top-k",
        "mmr": "MMR with lambda_mmr=0.5 over reranker scores on the pool",
        "noprox_rerank": "Free-energy with reranker energy, no proximal term (ablation)",
        "kl_rerank": "Free-energy with reranker energy, KL-proximal step (ablation)",
        "jko_rerank": "Free-energy with reranker energy, **Wasserstein**-proximal step",
        "jko_blend": "Wasserstein-proximal, energy = 0.4*dense + 0.6*rerank",
        "jko_blend_dense": "Wasserstein-proximal, energy = 0.7*dense + 0.3*rerank",
    }
    for m, d in descs.items():
        lines.append(f"- **{m}**: {d}")
    lines.append("")

    lines.append("## Headline retrieval metrics\n")
    lines.append("Mean per-query score with 95% bootstrap CI (n_boot=2000).\n")
    lines.append("| Method | nDCG@10 | Recall@5 | Recall@10 | Recall@20 | Diversity@10 |")
    lines.append("|---|---|---|---|---|---|")
    for m, s in s1["summary"].items():
        lines.append(
            f"| `{m}` | {fmt_ci(s['ndcg@10'])} | {fmt_ci(s['recall@5'])} | "
            f"{fmt_ci(s['recall@10'])} | {fmt_ci(s['recall@20'])} | {fmt_ci(s['diversity@10'])} |"
        )
    lines.append("")

    lines.append("## Paired bootstrap differences vs `rerank` baseline\n")
    lines.append("Positive = our method beats `rerank`. **Bold** = 95% CI excludes zero.\n")
    for metric in ["ndcg@10", "recall@10", "recall@20"]:
        lines.append(f"### {metric}\n")
        lines.append("| Method | Δ vs rerank | 95% CI |")
        lines.append("|---|---|---|")
        for m, d in s1["paired_vs_rerank"][metric].items():
            lines.append(f"| `{m}` | {fmt_diff(d)} | |")
        lines.append("")

    # decisive ablation: JKO vs KL with same energy
    lines.append("## Decisive ablation: Wasserstein vs KL proximal\n")
    lines.append("Same energy (reranker), same hyperparameters, only the proximal term differs.\n")
    if "jko_rerank" in s1["per_query"] and "kl_rerank" in s1["per_query"]:
        import numpy as np
        from evaluation import paired_bootstrap_diff
        for metric in ["ndcg@10", "recall@10", "recall@20", "diversity@10"]:
            a = s1["per_query"]["jko_rerank"][metric]
            b = s1["per_query"]["kl_rerank"][metric]
            diff, lo, hi = paired_bootstrap_diff(a, b)
            sig = "**" if (lo > 0 or hi < 0) else ""
            lines.append(f"- {metric}: jko − kl = {sig}{diff:+.4f}{sig} [{lo:+.4f}, {hi:+.4f}]")
    lines.append("")

    if stab:
        lines.append("## Stage 3: Retrieval-distribution stability under query perturbation\n")
        lines.append("Lower W_C(p_T(q), p_T(q')) = the retrieval distribution moves less when the query is perturbed.\n")
        lines.append("Perturbations: drop a stopword, append a hedge, lowercase + strip punctuation.\n")
        lines.append("| Method | Mean W_C over perturbations |")
        lines.append("|---|---|")
        for m, v in sorted(stab["per_method_mean_over_perturbations"].items(), key=lambda x: x[1]):
            lines.append(f"| `{m}` | {v:.4f} |")
        lines.append("")
        lines.append("Per-perturbation breakdown:\n")
        lines.append("| Method | drop_stop | hedge | lower_nop |")
        lines.append("|---|---|---|---|")
        for m, pdict in stab["summary"].items():
            parts = []
            for p in ["drop_stop", "hedge", "lower_nop"]:
                v = pdict.get(p, {}).get("mean", float("nan"))
                parts.append(f"{v:.4f}")
            lines.append(f"| `{m}` | {parts[0]} | {parts[1]} | {parts[2]} |")
        lines.append("")

    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote {RESULTS_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
