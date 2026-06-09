"""Multi-dataset BEIR loader.

A BEIR dataset on disk has the layout:
    data/<name>/<name>/corpus.jsonl
    data/<name>/<name>/queries.jsonl
    data/<name>/<name>/qrels/{train,dev,test}.tsv
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, dict[str, int]]]  # split -> qid -> did -> rel

    def docs(self) -> int:
        return len(self.corpus)

    def n_queries(self, split: str = "test") -> int:
        return len(self.qrels.get(split, {}))


def _load_corpus(p: Path) -> dict[str, dict[str, str]]:
    out: dict = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    return out


def _load_queries(p: Path) -> dict[str, str]:
    out: dict = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d["text"]
    return out


def _load_qrels(p: Path) -> dict[str, dict[str, int]]:
    out: dict = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, did, rel = parts
            out.setdefault(qid, {})[did] = int(rel)
    return out


def load_dataset(name: str) -> BeirDataset:
    base = DATA_DIR / name / name
    if not base.exists():
        # SciFact was downloaded with a different layout (data/scifact/scifact)
        # Detect and try the standard BEIR layout
        alt = DATA_DIR / "scifact" / "scifact" if name == "scifact" else None
        if alt and alt.exists():
            base = alt
        else:
            raise FileNotFoundError(f"Dataset not found at {base}")

    corpus = _load_corpus(base / "corpus.jsonl")
    queries = _load_queries(base / "queries.jsonl")
    qrels = {}
    for split in ("train", "dev", "test"):
        q = _load_qrels(base / "qrels" / f"{split}.tsv")
        if q:
            qrels[split] = q
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def summary_table(names: list[str]) -> None:
    print(f"{'dataset':<14s}  {'docs':>10s}  {'train_q':>10s}  {'dev_q':>10s}  {'test_q':>10s}")
    for n in names:
        try:
            d = load_dataset(n)
            tr = d.n_queries("train"); dv = d.n_queries("dev"); te = d.n_queries("test")
            print(f"{n:<14s}  {d.docs():>10d}  {tr:>10d}  {dv:>10d}  {te:>10d}")
        except Exception as e:
            print(f"{n:<14s}  ERROR: {e}")


if __name__ == "__main__":
    summary_table(["scifact", "nfcorpus", "trec-covid", "fiqa"])
