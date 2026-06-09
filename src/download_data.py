"""Download SciFact dataset from BEIR.

SciFact has:
- 5,183 abstracts (corpus)
- 1,109 train + 300 test claims (queries)
- expert-annotated evidence labels (qrels)

We use the test split for evaluation.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SCIFACT_DIR = DATA_DIR / "scifact"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {target}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(target, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 14):
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    print(f"Extracting {zip_path} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)


def load_scifact():
    corpus_path = SCIFACT_DIR / "scifact" / "corpus.jsonl"
    queries_path = SCIFACT_DIR / "scifact" / "queries.jsonl"
    qrels_test = SCIFACT_DIR / "scifact" / "qrels" / "test.tsv"
    qrels_train = SCIFACT_DIR / "scifact" / "qrels" / "train.tsv"

    corpus = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {
                "title": d.get("title", ""),
                "text": d.get("text", ""),
            }

    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]

    def load_qrels(p: Path) -> dict[str, dict[str, int]]:
        qrels: dict[str, dict[str, int]] = {}
        if not p.exists():
            return qrels
        with open(p, encoding="utf-8") as f:
            header = f.readline()  # skip header
            for line in f:
                qid, did, rel = line.strip().split("\t")
                qrels.setdefault(qid, {})[did] = int(rel)
        return qrels

    return corpus, queries, load_qrels(qrels_test), load_qrels(qrels_train)


def main():
    zip_path = DATA_DIR / "scifact.zip"
    if not zip_path.exists():
        download(BEIR_URL, zip_path)
    if not (SCIFACT_DIR / "scifact" / "corpus.jsonl").exists():
        extract_zip(zip_path, SCIFACT_DIR)

    corpus, queries, qrels_test, qrels_train = load_scifact()
    print(f"corpus: {len(corpus)} docs")
    print(f"queries: {len(queries)} total")
    print(f"qrels test: {len(qrels_test)} queries with judgments")
    print(f"qrels train: {len(qrels_train)} queries with judgments")
    avg_rel = sum(len(v) for v in qrels_test.values()) / max(1, len(qrels_test))
    print(f"avg relevant docs per test query: {avg_rel:.2f}")
    # peek
    sample_qid = next(iter(qrels_test))
    print(f"\nSample test query [{sample_qid}]: {queries[sample_qid]}")
    print(f"Relevant docs: {qrels_test[sample_qid]}")
    sample_did = next(iter(qrels_test[sample_qid]))
    print(f"\nSample doc [{sample_did}]:")
    print(f"  title: {corpus[sample_did]['title']}")
    print(f"  text:  {corpus[sample_did]['text'][:200]}...")


if __name__ == "__main__":
    main()
