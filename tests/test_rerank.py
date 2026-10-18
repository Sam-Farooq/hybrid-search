"""The re-ranking stage, driven with a stub model.

No download and no services. The one test that loads a real cross-encoder is
marked `model` and deselected by default, because a test that silently
downloads 80MB the first time CI runs is not a unit test.

asyncio.run rather than pytest-asyncio, so the dev dependencies stay at two.
"""
from __future__ import annotations

import asyncio

import pytest

from hybrid_search.engine import HybridEngine
from hybrid_search.fusion import FusedHit, Hit
from hybrid_search.rerank import CrossEncoderReranker, Reranker, apply


def fused(*doc_ids: str, text: str = "a document") -> list[FusedHit]:
    return [
        FusedHit(doc_id=d, rrf_score=1.0 / (i + 1), ranks={"bm25": i + 1}, text=f"{text} {d}")
        for i, d in enumerate(doc_ids)
    ]


class ScriptedReranker:
    """Scores each hit by a lookup, so a test can state the intended order."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.saw: list[str] = []

    async def rerank(self, query: str, hits: list[FusedHit]) -> list[FusedHit]:
        self.saw = [h.doc_id for h in hits]
        ranked = [
            FusedHit(h.doc_id, h.rrf_score, h.ranks, h.text, h.metadata, self.scores.get(h.doc_id, 0.0))
            for h in hits
        ]
        ranked.sort(key=lambda h: -h.rerank_score)
        return ranked


class Exploding:
    async def rerank(self, query: str, hits: list[FusedHit]) -> list[FusedHit]:
        raise RuntimeError("the model host is down")


class FakeSparse:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    async def search(self, query: str, limit: int = 50) -> list[Hit]:
        return self.hits[:limit]


class FakeDense(FakeSparse):
    pass


def test_the_protocol_is_satisfied_by_the_real_implementation():
    assert isinstance(CrossEncoderReranker(model=object()), Reranker)


def test_reranking_reorders_the_window_and_keeps_both_scores():
    hits = fused("a", "b", "c")
    out = asyncio.run(apply(ScriptedReranker({"c": 9.0, "a": 5.0, "b": 1.0}), "q", hits))

    assert [h.doc_id for h in out] == ["c", "a", "b"]
    assert [h.rerank_score for h in out] == [9.0, 5.0, 1.0]
    # The fused score survives, because the two disagreeing is the interesting
    # case and overwriting it would hide that.
    assert out[0].rrf_score == pytest.approx(1 / 3)


def test_only_the_top_of_the_list_is_reranked_and_the_tail_keeps_its_place():
    hits = fused(*[f"d-{i}" for i in range(10)])
    scripted = ScriptedReranker({"d-2": 100.0})

    out = asyncio.run(apply(scripted, "q", hits, depth=3))

    assert scripted.saw == ["d-0", "d-1", "d-2"]
    assert [h.doc_id for h in out[:3]] == ["d-2", "d-0", "d-1"]
    # Everything past the window is untouched, and nothing was dropped.
    assert [h.doc_id for h in out[3:]] == [f"d-{i}" for i in range(3, 10)]
    assert len(out) == 10


def test_no_reranker_leaves_the_fused_order_exactly_as_it_was():
    hits = fused("a", "b", "c")

    assert asyncio.run(apply(None, "q", hits)) == hits


def test_a_reranker_that_raises_degrades_to_the_fused_order():
    """Same rule as a dead retriever: a worse ordering beats a 500."""
    hits = fused("a", "b", "c")

    out = asyncio.run(apply(Exploding(), "q", hits))

    assert [h.doc_id for h in out] == ["a", "b", "c"]
    assert all(h.rerank_score is None for h in out)


def test_hits_with_no_text_are_returned_rather_than_scored_on_nothing():
    hits = [FusedHit(doc_id="a", rrf_score=1.0, ranks={"bm25": 1}, text="")]

    out = asyncio.run(CrossEncoderReranker(model=object()).rerank("q", hits))

    assert [h.doc_id for h in out] == ["a"]
    assert out[0].rerank_score is None


def test_an_empty_window_is_not_handed_to_the_model():
    class Forbidden:
        def predict(self, pairs, batch_size=32):
            raise AssertionError("the model should not have been called")

    assert asyncio.run(CrossEncoderReranker(model=Forbidden()).rerank("q", [])) == []


def test_the_cross_encoder_scores_query_document_pairs_and_sorts_by_the_result():
    seen = {}

    class FakeModel:
        def predict(self, pairs, batch_size=32):
            seen["pairs"] = pairs
            seen["batch_size"] = batch_size
            return [0.1, 0.9, 0.5]

    out = asyncio.run(
        CrossEncoderReranker(model=FakeModel(), batch_size=8).rerank("why", fused("a", "b", "c"))
    )

    assert seen["pairs"] == [("why", "a document a"), ("why", "a document b"),
                             ("why", "a document c")]
    assert seen["batch_size"] == 8
    assert [h.doc_id for h in out] == ["b", "c", "a"]


def test_documents_the_model_cannot_separate_keep_the_fused_order():
    class Flat:
        def predict(self, pairs, batch_size=32):
            return [0.5] * len(pairs)

    out = asyncio.run(CrossEncoderReranker(model=Flat()).rerank("q", fused("a", "b", "c")))

    assert [h.doc_id for h in out] == ["a", "b", "c"]


def test_the_engine_reranks_the_fused_window_and_applies_the_limit_afterwards():
    """The bug this ordering avoids.

    Fusing with limit=2 and then reranking would hand the model the two
    documents RRF already liked, which is the ordering the stage exists to
    doubt. The engine fuses everything, reranks the window, then truncates.
    """
    docs = [Hit(doc_id=f"d-{i}", score=1.0, text=f"text {i}") for i in range(6)]
    scripted = ScriptedReranker({"d-5": 99.0})
    engine = HybridEngine(FakeSparse(docs), FakeDense(docs), reranker=scripted)

    out = asyncio.run(engine.search("q", limit=2))

    assert scripted.saw == [f"d-{i}" for i in range(6)]
    assert [h.doc_id for h in out] == ["d-5", "d-0"]


def test_the_engine_without_a_reranker_behaves_as_it_did():
    docs = [Hit(doc_id=f"d-{i}", score=1.0, text=f"text {i}") for i in range(6)]
    engine = HybridEngine(FakeSparse(docs), FakeDense(docs))

    out = asyncio.run(engine.search("q", limit=3))

    assert [h.doc_id for h in out] == ["d-0", "d-1", "d-2"]
    assert all(h.rerank_score is None for h in out)


@pytest.mark.model
def test_a_real_cross_encoder_puts_the_answer_above_the_decoy():
    """The one test that loads the actual model.

    Two documents that BM25 and dense would both rank by overlap, where the
    decoy shares more words with the query than the answer does. That is the
    case RRF cannot fix, because neither retriever's position ordering knows
    which one answers the question.
    """
    hits = [
        FusedHit(
            doc_id="decoy",
            rrf_score=0.9,
            ranks={"bm25": 1},
            text="The capital allowance regime was reviewed. The capital city was not discussed.",
        ),
        FusedHit(
            doc_id="answer",
            rrf_score=0.1,
            ranks={"dense": 9},
            text="Paris is the capital of France.",
        ),
    ]

    out = asyncio.run(CrossEncoderReranker().rerank("what is the capital of France", hits))

    assert [h.doc_id for h in out] == ["answer", "decoy"]
    assert out[0].rerank_score > out[1].rerank_score
