#!/usr/bin/env python3
"""Build an evaluation set big enough to measure with, from a published corpus.

    python benchmarks/corpus.py --out benchmarks/scifact.jsonl

Why this exists: `benchmarks/queries.jsonl` has four queries per class with
relevant sets of one or two documents, so a class mean can only move in steps
of 0.125. `hybrid_search.evaluation.step` computes that from the file. Nothing
measured on it means anything, which is what made the weight question
unanswerable.

The corpus is **SciFact**, from BEIR. It is downloaded on demand into a
gitignored cache and never committed: it is CC BY-NC and redistributing it
inside an MIT repository would be wrong, and 2.8MB of someone else's data is
not this repository's to carry.

Two classes, built differently and honestly labelled:

- **natural** comes straight from SciFact's own test queries and its own
  human relevance judgements. This is a real, published, human-judged set.
- **identifier** is *constructed*, because no general corpus ships the kind of
  query this repository's identifier class means: a statute reference, an
  ISIN, a product code. It is built by finding alphanumeric tokens that occur
  in exactly one document, which gives an objective relevance judgement and
  tests exactly the mechanism the class is about, lexical exact match where
  an embedding has nothing to work with. It is **not** human-judged and does
  not stand in for user satisfaction. It stands in for "can this system find
  a string".

That distinction is the one thing not to lose when reading the numbers.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DEFAULT_CACHE = Path(".corpus-cache")

# An identifier-shaped token: some letters, then digits, optionally more.
# Deliberately narrow. A bare number matches dates, counts and p-values and
# would produce queries whose "one relevant document" is an accident.
IDENTIFIER = re.compile(r"\b[A-Za-z]{1,10}[-]?\d{2,}[A-Za-z0-9-]*\b")
MIN_IDENTIFIER_LENGTH = 5


def download(cache: Path = DEFAULT_CACHE, url: str = SCIFACT_URL) -> Path:
    """Fetch and unzip once. The cache is gitignored and safe to delete."""
    target = cache / "scifact"
    if (target / "corpus.jsonl").exists():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(cache)
    return target


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_documents(root: Path) -> list[dict]:
    """BEIR's corpus.jsonl, in this repository's document shape."""
    out = []
    for row in read_jsonl(root / "corpus.jsonl"):
        text = " ".join(part for part in (row.get("title", ""), row.get("text", "")) if part)
        out.append({"doc_id": row["_id"], "text": text, "source": "scifact"})
    return out


def load_qrels(root: Path, split: str = "test") -> dict[str, set[str]]:
    """Human relevance judgements, keyed by query id. Zero-graded rows drop."""
    judged: dict[str, set[str]] = defaultdict(set)
    lines = (root / "qrels" / f"{split}.tsv").read_text().splitlines()
    for line in lines[1:]:  # the first row is a header
        query_id, doc_id, score = line.split("\t")
        if int(score) > 0:
            judged[query_id].add(doc_id)
    return dict(judged)


def natural_cases(root: Path, split: str = "test") -> list[dict]:
    """SciFact's own queries and its own judgements. Nothing invented."""
    qrels = load_qrels(root, split)
    cases = []
    for row in read_jsonl(root / "queries.jsonl"):
        relevant = qrels.get(row["_id"])
        if relevant:
            cases.append(
                {"query": row["text"], "class": "natural", "relevant": sorted(relevant)}
            )
    return cases


def identifier_cases(documents: list[dict], limit: int = 150) -> list[dict]:
    """Tokens that occur in exactly one document, as exact-match queries.

    Sorted by token so the set is identical on every machine. A sample taken
    in corpus order would change whenever BEIR reordered its file, and a
    benchmark whose query set moves is not a benchmark.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for doc in documents:
        for token in IDENTIFIER.findall(doc["text"]):
            if len(token) >= MIN_IDENTIFIER_LENGTH:
                owners[token].add(doc["doc_id"])
    unique = sorted(token for token, docs in owners.items() if len(docs) == 1)
    return [
        {"query": token, "class": "identifier", "relevant": sorted(owners[token])}
        for token in unique[:limit]
    ]


def build(cache: Path = DEFAULT_CACHE, identifier_limit: int = 150) -> tuple[list[dict], list[dict]]:
    root = download(cache)
    documents = load_documents(root)
    return documents, natural_cases(root) + identifier_cases(documents, identifier_limit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/scifact.jsonl"))
    ap.add_argument("--corpus-out", type=Path, default=Path("benchmarks/scifact-corpus.jsonl"))
    ap.add_argument("--identifier-limit", type=int, default=150)
    args = ap.parse_args()

    documents, cases = build(args.cache, args.identifier_limit)
    args.out.write_text("".join(json.dumps(c) + "\n" for c in cases))
    args.corpus_out.write_text("".join(json.dumps(d) + "\n" for d in documents))

    from hybrid_search.evaluation import resolution

    print(f"{len(documents)} documents -> {args.corpus_out}")
    print(f"{len(cases)} queries -> {args.out}")
    for line in resolution(cases):
        print("  " + line.readable)


if __name__ == "__main__":
    main()
