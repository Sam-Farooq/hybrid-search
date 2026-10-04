#!/usr/bin/env python3
"""Recall@10 for BM25 alone, dense alone, and fused.

    python benchmarks/compare.py --queries benchmarks/queries.jsonl

The point of the benchmark is not the fused number. It is the two query
classes: fusion is roughly a wash on natural-language questions and a large
win on anything containing an identifier, which is where dense alone falls
over.
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
from hybrid_search.engine import HybridEngine
from hybrid_search.fusion import reciprocal_rank_fusion
from hybrid_search.sparse import SparseIndex


def recall_at(hits: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(hits[:k]) & relevant) / len(relevant)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="benchmarks/queries.jsonl")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    es = AsyncElasticsearch("http://localhost:9200")
    qd = AsyncQdrantClient(url="http://localhost:6333")
    sparse, dense = SparseIndex(es), DenseIndex(qd)
    engine = HybridEngine(sparse, dense)

    cases = [json.loads(l) for l in Path(args.queries).read_text().splitlines() if l.strip()]
    totals: dict[str, dict[str, list[float]]] = {}

    for case in cases:
        klass = case.get("class", "general")
        relevant = set(case["relevant"])
        s = await sparse.search(case["query"], limit=50)
        d = await dense.search(case["query"], limit=50)
        f = reciprocal_rank_fusion({"bm25": s, "dense": d})

        bucket = totals.setdefault(klass, {"bm25": [], "dense": [], "fused": []})
        bucket["bm25"].append(recall_at([h.doc_id for h in s], relevant, args.k))
        bucket["dense"].append(recall_at([h.doc_id for h in d], relevant, args.k))
        bucket["fused"].append(recall_at([h.doc_id for h in f], relevant, args.k))

    rows = [
        [klass, len(v["bm25"]),
         f"{sum(v['bm25']) / len(v['bm25']):.3f}",
         f"{sum(v['dense']) / len(v['dense']):.3f}",
         f"{sum(v['fused']) / len(v['fused']):.3f}"]
        for klass, v in sorted(totals.items())
    ]
    print(tabulate(rows, headers=["class", "n", "bm25", "dense", "fused"]))
    await es.close()


if __name__ == "__main__":
    asyncio.run(main())
