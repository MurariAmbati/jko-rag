"""Download the ORIGINAL SciFact dataset (Allen AI's release with labels and rationale).

The BEIR version flattens this to (query, doc, relevance) tuples, losing
the SUPPORTS/REFUTES/NEI labels and the rationale sentence ids. We need
those for Stage 2 generation evaluation.

Source: https://scifact.apps.allenai.org/
Mirror: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "scifact_orig"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def download(url, target):
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(target, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))


def extract_tar(p, out):
    print(f"Extracting {p} -> {out}")
    with tarfile.open(p, "r:gz") as t:
        t.extractall(out)


def main():
    tar_path = DATA_DIR / "data.tar.gz"
    if not tar_path.exists():
        download(URL, tar_path)
    extract_tar(tar_path, DATA_DIR)

    # Show structure
    print("\nContents:")
    for p in sorted(DATA_DIR.rglob("*.jsonl")):
        print(f"  {p.relative_to(DATA_DIR)} — {p.stat().st_size//1024} KB")

    # Sanity check
    for split in ("train", "dev"):
        p = DATA_DIR / "data" / f"claims_{split}.jsonl"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                first = json.loads(f.readline())
            print(f"\nSample {split} claim:")
            print(json.dumps(first, indent=2)[:600])
            break


if __name__ == "__main__":
    main()
