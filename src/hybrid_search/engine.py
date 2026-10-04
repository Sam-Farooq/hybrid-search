"""Both retrievers, concurrently, fused."""
from __future__ import annotations

import asyncio
import logging

from hybrid_search.dense import DenseIndex
from hybrid_search.fusion import FusedHit, reciprocal_rank_fusion
from hybrid_search.sparse import SparseIndex

log = logging.getLogger(__name__)

# Each retriever returns this many before fusion. Deeper lists help: a
# document that BM25 puts at 40 and dense puts at 3 should surface, and it
# cannot if BM25 only returned 10.
CANDIDATE_DEPTH = 50


class HybridEngine:
    def __init__(self, sparse: SparseIndex, dense: DenseIndex,
                 weights: dict[str, float] | None = None):
        self.sparse = sparse
        self.dense = dense
        self.weights = weights or {}

    async def index(self, docs: list[dict]) -> dict[str, int]:
        sparse_n, dense_n = await asyncio.gather(
            self.sparse.bulk_index(docs), self.dense.upsert(docs)
        )
        return {"sparse": sparse_n, "dense": dense_n}

    async def search(self, query: str, limit: int = 10,
                     depth: int = CANDIDATE_DEPTH) -> list[FusedHit]:
        sparse_hits, dense_hits = await asyncio.gather(
            self.sparse.search(query, limit=depth),
            self.dense.search(query, limit=depth),
            return_exceptions=True,
        )

        runs = {}
        # A half-degraded search beats a 500. If Elasticsearch is down the
        # dense half still answers, and the response says which ran.
        if isinstance(sparse_hits, Exception):
            log.warning("sparse retrieval failed: %s", sparse_hits)
        else:
            runs["bm25"] = sparse_hits
        if isinstance(dense_hits, Exception):
            log.warning("dense retrieval failed: %s", dense_hits)
        else:
            runs["dense"] = dense_hits

        if not runs:
            raise RuntimeError("both retrievers failed")

        return reciprocal_rank_fusion(runs, weights=self.weights, limit=limit)
