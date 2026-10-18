"""The corpus builder's judgement rules, without downloading anything.

The one test that fetches SciFact is marked `network` and deselected by
default. It was run while this was written and its output is quoted in the
README; it is kept so that a change to BEIR's layout fails somewhere
sensible rather than six months later.
"""
from __future__ import annotations

import pytest
from corpus import IDENTIFIER, build, identifier_cases

from hybrid_search.evaluation import resolution


def docs(*pairs: tuple[str, str]) -> list[dict]:
    return [{"doc_id": d, "text": t, "source": "test"} for d, t in pairs]


def test_a_token_in_two_documents_is_not_a_usable_query():
    """The judgement has to be objective, and a token shared by two documents
    has no single right answer."""
    cases = identifier_cases(docs(("d1", "see RTS28 here"), ("d2", "also RTS28")))

    assert cases == []


def test_a_token_in_exactly_one_document_becomes_a_query_for_it():
    cases = identifier_cases(docs(("d1", "the marker CD8942 appears"), ("d2", "nothing here")))

    assert cases == [{"query": "CD8942", "class": "identifier", "relevant": ["d1"]}]


def test_bare_numbers_are_not_identifiers():
    """A year, a count or a p-value owned by one document would produce a
    query whose single relevant document is an accident."""
    assert IDENTIFIER.findall("in 2019 we found 42 cases") == []


def test_short_tokens_are_rejected_even_when_unique():
    # "p53" is unique here and is four characters, under the floor.
    assert identifier_cases(docs(("d1", "p53 and ABC1234"), ("d2", "other"))) == [
        {"query": "ABC1234", "class": "identifier", "relevant": ["d1"]}
    ]


def test_the_query_set_is_the_same_on_every_machine():
    """Sorted by token, not taken in corpus order.

    A sample that moved when BEIR reordered its file would make two runs of
    the benchmark incomparable, which is the failure this whole exercise is
    about.
    """
    forward = identifier_cases(docs(("d1", "ZZZ9999"), ("d2", "AAA1111")))
    backward = identifier_cases(docs(("d2", "AAA1111"), ("d1", "ZZZ9999")))

    assert forward == backward
    assert [c["query"] for c in forward] == ["AAA1111", "ZZZ9999"]


def test_the_limit_takes_the_first_tokens_in_sort_order():
    many = docs(*[(f"d{i}", f"TOK{1000 + i}") for i in range(10)])

    cases = identifier_cases(many, limit=3)

    assert [c["query"] for c in cases] == ["TOK1000", "TOK1001", "TOK1002"]


@pytest.mark.network
def test_the_real_corpus_is_two_orders_of_magnitude_finer_than_the_shipped_set():
    """Downloads SciFact. The numbers below were produced by running this."""
    documents, cases = build()
    steps = {r.name: r for r in resolution(cases)}

    assert len(documents) == 5183
    assert steps["natural"].queries == 300
    assert steps["identifier"].queries == 150
    # The shipped eight-query set steps in 0.125 for both classes.
    assert steps["natural"].step < 0.001
    assert steps["identifier"].step < 0.01
