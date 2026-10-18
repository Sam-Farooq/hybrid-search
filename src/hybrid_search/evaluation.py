"""Scoring, and the arithmetic that says whether a score difference is real.

Two issues on this repository said the same thing in different words: eight
queries cannot separate two retrievers, and the weights were never tuned
because nothing could show a difference. Both are the same property, and it
is computable from the query set alone without running a single search.

**Recall@k is quantised.** A query with r relevant documents can only score
0, 1/r, 2/r ... 1. A class mean over n such queries therefore moves in steps
of 1 / (n * lcm(r_1..r_n)) and nothing finer exists. Any measured difference
smaller than one step is not a small effect, it is zero plus rounding.

So `step()` is the number to look at before believing any comparison in this
repository, and `benchmarks/resolution.py` prints it for a query file.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import lcm


def recall_at(hits: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Fraction of the relevant set retrieved in the top k."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(hits[:k]) & relevant) / len(relevant)


def step(relevant_sizes: Sequence[int]) -> float:
    """The smallest non-zero difference two mean recalls can have.

    Worked for the set this repository shipped with: the natural class has
    four queries with relevant sets of 2, 1, 2 and 1, so lcm is 2 and the
    mean moves in steps of 1/(4*2) = 0.125. A weighting that improved recall
    by three points could not be seen at all.
    """
    sizes = [s for s in relevant_sizes if s > 0]
    if not sizes:
        return 0.0
    return 1.0 / (len(sizes) * lcm(*sizes))


@dataclass(frozen=True)
class ClassResolution:
    name: str
    queries: int
    relevant_lcm: int
    step: float

    @property
    def readable(self) -> str:
        return f"{self.name}: {self.queries} queries, smallest visible difference {self.step:.4f}"


def resolution(cases: Iterable[Mapping]) -> list[ClassResolution]:
    """Per class, how fine a difference the query set can show."""
    by_class: dict[str, list[int]] = {}
    for case in cases:
        name = case.get("class", "general")
        by_class.setdefault(name, []).append(len(set(case["relevant"])))
    out = []
    for name, sizes in sorted(by_class.items()):
        usable = [s for s in sizes if s > 0]
        out.append(
            ClassResolution(
                name=name,
                queries=len(sizes),
                relevant_lcm=lcm(*usable) if usable else 0,
                step=step(sizes),
            )
        )
    return out


def mean_recall(
    runs: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Iterable[str]],
    k: int = 10,
) -> float:
    """Mean recall@k over queries, keyed by query id."""
    if not runs:
        return 0.0
    return sum(recall_at(hits, relevant[qid], k) for qid, hits in runs.items()) / len(runs)


def fuse_weighted(
    per_query_runs: Mapping[str, Mapping[str, Sequence[str]]],
    weights: Mapping[str, float],
    k_rrf: int = 60,
    limit: int = 10,
) -> dict[str, list[str]]:
    """Apply RRF with a given weighting to already-retrieved runs.

    Takes doc id lists rather than Hits so a sweep can run over a cached
    retrieval, which is what makes tuning possible without holding both
    services up for the length of a grid search.
    """
    fused: dict[str, list[str]] = {}
    for qid, runs in per_query_runs.items():
        scores: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        for run_name, doc_ids in runs.items():
            weight = weights.get(run_name, 1.0)
            for position, doc_id in enumerate(doc_ids, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k_rrf + position)
                best_rank[doc_id] = min(best_rank.get(doc_id, position), position)
        order = sorted(scores, key=lambda d: (-scores[d], best_rank[d], d))
        fused[qid] = order[:limit]
    return fused


@dataclass(frozen=True)
class SweepResult:
    weights: dict[str, float]
    score: float


def sweep(
    per_query_runs: Mapping[str, Mapping[str, Sequence[str]]],
    relevant: Mapping[str, Iterable[str]],
    grid: Sequence[float],
    retrievers: Sequence[str] = ("bm25", "dense"),
    k: int = 10,
) -> list[SweepResult]:
    """Score every weighting on the grid, best first.

    Only the ratio matters to RRF, so one retriever is pinned at 1.0 and the
    other varies. A grid over both would spend most of its time re-evaluating
    the same ordering under a different name.
    """
    head, *rest = retrievers
    results = []
    for value in grid:
        weights = {head: 1.0}
        for name in rest:
            weights[name] = value
        fused = fuse_weighted(per_query_runs, weights, limit=k)
        results.append(SweepResult(weights=weights, score=mean_recall(fused, relevant, k)))
    results.sort(key=lambda r: -r.score)
    return results


def is_distinguishable(best: SweepResult, baseline: SweepResult, resolution_step: float) -> bool:
    """Is the winner's margin bigger than the smallest difference that exists?

    This is the question issue #2 left open. Without it a sweep always
    produces a winner, and a winner that beat equal weighting by less than one
    quantisation step has not beaten it at all.
    """
    return abs(best.score - baseline.score) >= resolution_step
