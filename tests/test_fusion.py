"""Fusion is pure, so it gets tested properly. The retrievers are integration
work and live behind docker compose."""
import pytest

from hybrid_search.fusion import Hit, reciprocal_rank_fusion


def hits(*ids):
    return [Hit(doc_id=i, score=0.0, text=i) for i in ids]


def test_agreement_at_rank_one_wins():
    fused = reciprocal_rank_fusion({"bm25": hits("a", "b"), "dense": hits("a", "c")})
    assert fused[0].doc_id == "a"
    assert fused[0].ranks == {"bm25": 1, "dense": 1}


def test_a_document_found_by_both_beats_one_found_by_one_higher():
    # 'b' is second in both: 1/62 + 1/62 = 0.03226
    # 'a' is first in one only:       1/61 = 0.01639
    fused = reciprocal_rank_fusion({"bm25": hits("a", "b"), "dense": hits("c", "b")})
    assert fused[0].doc_id == "b"


def test_scores_are_never_read_only_positions():
    # Identical rankings, wildly different scores. RRF must not notice.
    a = [Hit("x", 1000.0), Hit("y", 999.0)]
    b = [Hit("x", 0.001), Hit("y", 0.0009)]
    assert [h.doc_id for h in reciprocal_rank_fusion({"r1": a, "r2": b})] == ["x", "y"]


def test_weights_shift_the_balance():
    runs = {"bm25": hits("a", "b"), "dense": hits("b", "a")}
    unweighted = reciprocal_rank_fusion(runs)
    assert unweighted[0].doc_id == "a"          # tie, broken by doc_id
    weighted = reciprocal_rank_fusion(runs, weights={"dense": 3.0})
    assert weighted[0].doc_id == "b"


def test_ordering_is_stable_across_calls():
    runs = {"bm25": hits("a", "b", "c"), "dense": hits("c", "b", "a")}
    first = [h.doc_id for h in reciprocal_rank_fusion(runs)]
    for _ in range(20):
        assert [h.doc_id for h in reciprocal_rank_fusion(runs)] == first


def test_larger_k_flattens_the_gap_between_ranks():
    runs = {"bm25": hits("a", "b")}
    tight = reciprocal_rank_fusion(runs, k=1)
    loose = reciprocal_rank_fusion(runs, k=1000)
    assert tight[0].rrf_score - tight[1].rrf_score > loose[0].rrf_score - loose[1].rrf_score


def test_single_retriever_preserves_its_own_order():
    fused = reciprocal_rank_fusion({"bm25": hits("a", "b", "c")})
    assert [h.doc_id for h in fused] == ["a", "b", "c"]
    assert all(not h.found_by_all for h in fused)


def test_limit_truncates_after_sorting_not_before():
    runs = {"bm25": hits("a", "b", "c"), "dense": hits("c", "c2", "c3")}
    assert [h.doc_id for h in reciprocal_rank_fusion(runs, limit=1)] == ["c"]


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="rank offset"):
        reciprocal_rank_fusion({"bm25": hits("a")}, k=0)
