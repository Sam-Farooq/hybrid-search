"""Reciprocal rank fusion.

    score(d) = sum over retrievers of  1 / (k + rank(d))

The property that makes RRF worth using is that it never looks at the
retrievers' own scores, only at the positions. BM25 returns unbounded TF-IDF
sums; cosine similarity returns something in [-1, 1]. Any attempt to combine
those numerically needs normalisation, and every normalisation scheme is
wrong somewhere: min-max is destroyed by one outlier, z-score assumes a
distribution neither retriever has, and a fixed scale factor has to be
retuned whenever the corpus changes.

Ranks have none of those problems. The cost is that RRF cannot tell a
confident first place from a marginal one.

k=60 is the value from Cormack, Clarke and Buettcher (SIGIR 2009) and it has
held up on every corpus I have tried it on. It is large enough that the
difference between rank 1 and rank 2 is small (1/61 vs 1/62), which is what
stops a single retriever dominating the fused list.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

DEFAULT_K = 60


@dataclass
class Hit:
    doc_id: str
    score: float
    text: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class FusedHit:
    doc_id: str
    rrf_score: float
    ranks: dict[str, int]
    text: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def found_by_all(self) -> bool:
        return len(self.ranks) > 1


def reciprocal_rank_fusion(
    runs: dict[str, list[Hit]],
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse named result lists. Each list must already be in rank order."""
    if k <= 0:
        raise ValueError("k must be positive; it is the rank offset, not a result count")

    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    payload: dict[str, Hit] = {}

    for run_name, hits in runs.items():
        weight = weights.get(run_name, 1.0)
        for position, hit in enumerate(hits, start=1):
            scores[hit.doc_id] += weight / (k + position)
            ranks[hit.doc_id][run_name] = position
            # First writer wins. The retrievers return the same text, and
            # overwriting means the payload depends on dict ordering.
            payload.setdefault(hit.doc_id, hit)

    fused = [
        FusedHit(
            doc_id=doc_id,
            rrf_score=score,
            ranks=ranks[doc_id],
            text=payload[doc_id].text,
            metadata=payload[doc_id].metadata,
        )
        for doc_id, score in scores.items()
    ]
    # Tie-break on the best single rank, then on doc_id so the order is stable
    # across runs. Without the second key, two documents with identical scores
    # swap places between calls and the eval numbers wobble.
    fused.sort(key=lambda f: (-f.rrf_score, min(f.ranks.values()), f.doc_id))
    return fused[:limit] if limit else fused
