"""The stage after fusion, and why fusion needs one.

RRF orders by position agreement and deliberately never reads the retrievers'
scores, which is what makes it indifferent to BM25 and cosine living on
incomparable scales. The cost, stated in fusion.py and now acted on, is that
it cannot tell a confident first place from a marginal one: a document both
retrievers put near the top outranks one that a single retriever is certain
about, whatever either of them actually thought.

That trade is right for recall and wrong for the top 3, which is the part a
user sees. A cross-encoder fixes it by doing the one thing neither retriever
nor the fusion does: reading the query and the document together.

Deliberately a separate stage rather than a change to fusion. Fusion stays
pure, cheap and testable, and the expensive model only ever sees the fused
window rather than the corpus.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Protocol, runtime_checkable

from hybrid_search.fusion import FusedHit

log = logging.getLogger(__name__)

# Small, fast, and trained on exactly this task. Larger ms-marco cross-encoders
# score a little better and cost several times the latency, which is the wrong
# trade inside a request.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How deep into the fused list to rerank. 50 matches engine.CANDIDATE_DEPTH:
# reranking fewer than the retrievers returned would throw away the recall the
# depth was chosen to buy.
DEFAULT_RERANK_DEPTH = 50


@runtime_checkable
class Reranker(Protocol):
    """Reorder a fused window. Returns the same hits, never a different set."""

    async def rerank(self, query: str, hits: list[FusedHit]) -> list[FusedHit]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder over (query, document) pairs.

    The model is loaded on first use, not in __init__, so constructing an
    engine costs nothing and a process that never reranks never pays for the
    weights. `model` is injectable so the tests can drive this without a
    download.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        model: object | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model

    def _loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(self, query: str, hits: list[FusedHit]) -> list[FusedHit]:
        if not hits:
            return []
        # A hit with no text cannot be scored against the query. That happens
        # when a retriever returned an id without a payload, and scoring the
        # empty string would quietly rank it on nothing.
        if not any(hit.text for hit in hits):
            log.warning("no hit carries text, so there is nothing to rerank")
            return list(hits)

        pairs = [(query, hit.text) for hit in hits]

        def _run():
            return self._loaded().predict(pairs, batch_size=self.batch_size)

        scores = await asyncio.to_thread(_run)
        scored = [replace(hit, rerank_score=float(score)) for hit, score in zip(hits, scores)]
        # Stable sort on the score alone, so hits the model cannot separate
        # keep the fused order rather than being shuffled by doc_id.
        scored.sort(key=lambda h: -h.rerank_score)
        return scored


async def apply(
    reranker: Reranker | None,
    query: str,
    fused: list[FusedHit],
    depth: int = DEFAULT_RERANK_DEPTH,
) -> list[FusedHit]:
    """Rerank the top `depth` and leave the tail where it is.

    The tail is kept rather than dropped because the caller's limit is applied
    after this, and a reranker that is given 50 should not be able to shorten
    a list of 200.

    A reranker that raises degrades to the fused order, the same way a dead
    retriever degrades to the other one. The warning is the signal; a request
    that 500s because the re-ranking model is unavailable is a worse answer
    than a slightly worse ordering.
    """
    if reranker is None or not fused:
        return fused
    window, tail = fused[:depth], fused[depth:]
    try:
        return await reranker.rerank(query, window) + tail
    except Exception as exc:  # noqa: BLE001 - degrading on purpose, see above
        log.warning("reranking failed, falling back to the fused order: %s", exc)
        return fused
