"""Download additional BEIR datasets (NFCorpus, TREC-COVID, FiQA) in parallel.

Skips HotpotQA/Natural Questions: corpora are 5.2M / 2.7M passages — encoding
on CPU would take ~24h+ each. Including them would require either GPU or
subsampling that breaks the standard benchmark protocol.

NFCorpus:    3,633 docs, 323 test queries, biomedical
TREC-COVID:  171,332 docs, 50 test queries, biomedical
FiQA-2018:   57,638 docs, 648 test queries, financial QA
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASETS = {
    "nfcorpus":    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "trec-covid":  "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
    "fiqa":        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
}


def download_one(name: str, url: str) -> tuple[str, Path]:
    target = DATA_DIR / f"{name}.zip"
    if target.exists() and target.stat().st_size > 100_000:
        return name, target
    print(f"[{name}] downloading {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return name, target


def extract_one(name: str, zip_path: Path) -> None:
    import zipfile
    out = DATA_DIR / name
    out.mkdir(exist_ok=True)
    expected = out / name / "corpus.jsonl"
    if expected.exists():
        return
    print(f"[{name}] extracting")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)


def stats_one(name: str) -> dict:
    base = DATA_DIR / name / name
    corpus_p = base / "corpus.jsonl"
    queries_p = base / "queries.jsonl"
    qrels_p = base / "qrels" / "test.tsv"
    n_corpus = sum(1 for _ in open(corpus_p, encoding="utf-8"))
    n_q = sum(1 for _ in open(queries_p, encoding="utf-8"))
    n_qrels_q = set()
    n_rel = 0
    with open(qrels_p, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            qid, did, rel = line.strip().split("\t")
            if int(rel) > 0:
                n_qrels_q.add(qid)
                n_rel += 1
    return {"docs": n_corpus, "queries_total": n_q, "test_queries": len(n_qrels_q), "test_qrels": n_rel}


def main():
    print("Downloading datasets in parallel...")
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(download_one, n, u): n for n, u in DATASETS.items()}
        for fut in as_completed(futs):
            name, path = fut.result()
            print(f"[{name}] OK: {path} ({path.stat().st_size/1e6:.1f} MB)")

    for name in DATASETS:
        extract_one(name, DATA_DIR / f"{name}.zip")

    print("\nDataset statistics:")
    print(f"{'dataset':<14s}  {'docs':>10s}  {'q_total':>10s}  {'test_q':>10s}  {'test_qrels':>12s}")
    for name in DATASETS:
        st = stats_one(name)
        print(f"{name:<14s}  {st['docs']:>10d}  {st['queries_total']:>10d}  "
              f"{st['test_queries']:>10d}  {st['test_qrels']:>12d}")


if __name__ == "__main__":
    main()
