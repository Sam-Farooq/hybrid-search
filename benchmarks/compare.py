#!/usr/bin/env python3
"""Recall@10 for BM25 alone, dense alone, fused, and optionally reranked.

    python benchmarks/compare.py --queries benchmarks/queries.jsonl
    python benchmarks/compare.py --queries benchmarks/scifact.jsonl --sweep
    python benchmarks/compare.py --queries benchmarks/scifact.jsonl --rerank

The point of the benchmark is not the fused number. It is the two query
classes: fusion is roughly a wash on natural-language questions and a large
win on anything containing an identifier, which is where dense alone falls
over.

**Read the resolution line before the table.** It is printed first on
purpose. Recall@k is quantised by the size of each relevant set, so a class
mean cannot move by less than one step, and a difference smaller than that is
rounding rather than a result. On the eight-query set this repository shipped
with, that step is 0.125, which is why nothing measured there separated
anything. `benchmarks/corpus.py` builds a set two orders of magnitude finer.

This needs Elasticsearch and Qdrant up; `docker-compose.yml` has both.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient
from tabulate import tabulate

from hybrid_search.dense import DenseIndex
from hybrid_search.evaluation import (
    is_distinguishable,
    recall_at,
    resolution,
    sweep,
)
from hybrid_search.fusion import reciprocal_rank_fusion
from hybrid_search.rerank import CrossEncoderReranker
from hybrid_search.rerank import apply as apply_rerank
from hybrid_search.sparse import SparseIndex

SWEEP_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="benchmarks/queries.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--sweep", action="store_true", help="grid search the retriever weights")
    ap.add_argument("--rerank", action="store_true", help="add a cross-encoder column")
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.queries).read_text().splitlines() if l.strip()]

    steps = {r.name: r.step for r in resolution(cases)}
    print("Resolution, before any number below is believed:")
    for line in resolution(cases):
        print("  " + line.readable)
    print()

    es = AsyncElasticsearch("http://localhost:9200")
    qd = AsyncQdrantClient(url="http://localhost:6333")
    sparse, dense = SparseIndex(es), DenseIndex(qd)
    reranker = CrossEncoderReranker() if args.rerank else None

    totals: dict[str, dict[str, list[float]]] = {}
    cached_runs: dict[str, dict[str, list[str]]] = {}
    relevant_by_id: dict[str, set[str]] = {}

    for index, case in enumerate(cases):
        klass = case.get("class", "general")
        relevant = set(case["relevant"])
        query_id = f"q{index}"
        s = await sparse.search(case["query"], limit=args.depth)
        d = await dense.search(case["query"], limit=args.depth)
        f = reciprocal_rank_fusion({"bm25": s, "dense": d})

        # Kept so the weight sweep can run over a single retrieval rather than
        # holding both services up for the length of a grid search.
        cached_runs[query_id] = {
            "bm25": [h.doc_id for h in s],
            "dense": [h.doc_id for h in d],
        }
        relevant_by_id[query_id] = relevant

        bucket = totals.setdefault(klass, {"bm25": [], "dense": [], "fused": [], "reranked": []})
        bucket["bm25"].append(recall_at([h.doc_id for h in s], relevant, args.k))
        bucket["dense"].append(recall_at([h.doc_id for h in d], relevant, args.k))
        bucket["fused"].append(recall_at([h.doc_id for h in f], relevant, args.k))
        if reranker is not None:
            r = await apply_rerank(reranker, case["query"], f, depth=args.depth)
            bucket["reranked"].append(recall_at([h.doc_id for h in r], relevant, args.k))

    columns = ["bm25", "dense", "fused"] + (["reranked"] if reranker else [])
    rows = [
        [klass, len(v["bm25"])] + [f"{sum(v[c]) / len(v[c]):.3f}" for c in columns]
        for klass, v in sorted(totals.items())
    ]
    print(tabulate(rows, headers=["class", "n", *columns]))

    if args.sweep:
        print()
        results = sweep(cached_runs, relevant_by_id, grid=SWEEP_GRID, k=args.k)
        equal = next(r for r in results if r.weights["dense"] == 1.0)
        best = results[0]
        print(tabulate(
            [[f"{r.weights['dense']:.2f}", f"{r.score:.4f}"] for r in results],
            headers=["dense weight (bm25 = 1)", f"mean recall@{args.k}"],
        ))
        # One step, taken as the coarsest class, because the sweep is scored
        # over every query at once.
        coarsest = max(steps.values()) if steps else 0.0
        verdict = is_distinguishable(best, equal, coarsest)
        print()
        print(f"best {best.weights['dense']:.2f} at {best.score:.4f}, "
              f"equal weighting at {equal.score:.4f}, "
              f"smallest visible difference {coarsest:.4f}")
        print("verdict:", "a real difference" if verdict else
              "NOT distinguishable from equal weighting on this query set")

    await es.close()


if __name__ == "__main__":
    asyncio.run(main())
