"""The arithmetic behind both measurement issues, checked against the set
that provoked them.

Nothing here needs a service. That is the point: the reason eight queries
cannot separate two retrievers is a property of the query file, not of the
retrieval, and it was computable all along.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_search.evaluation import (
    SweepResult,
    fuse_weighted,
    is_distinguishable,
    mean_recall,
    recall_at,
    resolution,
    step,
    sweep,
)

QUERIES = Path(__file__).resolve().parents[1] / "benchmarks" / "queries.jsonl"


def cases():
    return [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]


def test_recall_counts_the_relevant_set_not_the_hits():
    assert recall_at(["a", "b", "c"], {"a", "z"}, k=10) == 0.5
    assert recall_at(["a", "b", "c"], {"a"}, k=10) == 1.0
    assert recall_at(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_an_empty_relevant_set_scores_zero_rather_than_dividing_by_it():
    assert recall_at(["a"], set(), k=10) == 0.0


def test_the_step_is_one_over_n_times_the_lcm():
    # Four queries, relevant sets of 2, 1, 2, 1: lcm 2, so 1/(4*2).
    assert step([2, 1, 2, 1]) == pytest.approx(0.125)
    # Ten queries all with a single relevant document move in tenths.
    assert step([1] * 10) == pytest.approx(0.1)
    # A set with a 3 in it gets finer, not coarser.
    assert step([2, 3]) == pytest.approx(1 / 12)


def test_the_shipped_query_file_can_only_move_in_eighths():
    """Reproduces the number in the issue from the data rather than quoting it.

    If someone adds queries, this test changes, and that is the signal that
    the benchmark's resolution changed.
    """
    by_name = {r.name: r for r in resolution(cases())}

    assert by_name["natural"].queries == 4
    assert by_name["natural"].step == pytest.approx(0.125)
    assert by_name["identifier"].queries == 4
    assert by_name["identifier"].step == pytest.approx(0.125)


def test_relevant_ids_are_counted_once_however_often_they_are_listed():
    """A judgement file with a repeated id is not a query with two answers,
    and counting the list rather than the set would make the step look finer
    than it is."""
    assert resolution([{"class": "n", "relevant": ["d", "d", "d"]}])[0].relevant_lcm == 1


def test_a_hundred_queries_per_class_is_about_two_orders_finer():
    """The target the corpus loader is sized against."""
    hundred = [
        {"class": "natural", "relevant": [f"d-{i}-{j}" for j in range(1 + i % 3)]}
        for i in range(100)
    ]

    # Relevant sets of 1, 2 and 3 give lcm 6, so 1/(100*6).
    assert resolution(hundred)[0].step == pytest.approx(1 / 600)


def test_weighting_changes_the_fused_order_when_the_runs_disagree():
    runs = {"q1": {"bm25": ["a", "b"], "dense": ["b", "a"]}}

    assert fuse_weighted(runs, {"bm25": 1.0, "dense": 1.0})["q1"] == ["a", "b"]
    assert fuse_weighted(runs, {"bm25": 1.0, "dense": 5.0})["q1"] == ["b", "a"]


def test_the_sweep_returns_its_best_weighting_first():
    """A case equal weighting genuinely loses.

    Each query has the answer in the dense run only and a distractor in the
    sparse run only, both at rank 1. At equal weights the two tie on score
    and on best rank, so the stable doc_id tie-break decides it, and the ids
    are chosen so that the distractor wins. Only a dense weight above 1
    separates them.
    """
    runs = {
        "q1": {"bm25": ["aa-distractor"], "dense": ["zz-answer"]},
        "q2": {"bm25": ["aa-other"], "dense": ["zz-target"]},
    }
    relevant = {"q1": {"zz-answer"}, "q2": {"zz-target"}}

    results = sweep(runs, relevant, grid=[0.25, 1.0, 4.0], k=1)

    assert results[0].weights["dense"] == 4.0
    assert results[0].score == 1.0
    assert results[-1].score == 0.0


def test_a_margin_smaller_than_one_step_is_not_a_result():
    """The question issue #2 left open, answered rather than restated.

    A sweep always produces a winner. Whether that winner means anything is a
    separate question, and it is this one.
    """
    best = SweepResult(weights={"bm25": 1.0, "dense": 2.0}, score=0.80)
    equal = SweepResult(weights={"bm25": 1.0, "dense": 1.0}, score=0.75)

    # On the shipped eight-query set nothing below 0.125 is visible, so a five
    # point margin is indistinguishable from zero.
    assert is_distinguishable(best, equal, 0.125) is False
    # On a set of a hundred queries the same margin is real.
    assert is_distinguishable(best, equal, 1 / 600) is True


def test_mean_recall_averages_over_queries_not_over_documents():
    runs = {"q1": ["a"], "q2": ["x"]}
    relevant = {"q1": {"a", "b", "c"}, "q2": {"x"}}

    # q1 scores 1/3 and q2 scores 1, so the mean is 2/3 and not 2/4.
    assert mean_recall(runs, relevant) == pytest.approx(2 / 3)
