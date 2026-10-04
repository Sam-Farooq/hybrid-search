# hybrid-search

BM25 in Elasticsearch, dense vectors in Qdrant, results combined with
reciprocal rank fusion.

```python
from hybrid_search.fusion import Hit, reciprocal_rank_fusion

fused = reciprocal_rank_fusion({
    "bm25":  [Hit("d-7", 14.2), Hit("d-3", 9.8)],
    "dense": [Hit("d-3", 0.81), Hit("d-9", 0.77)],
})
# d-3 first: second in both beats first in one
```

## Why RRF and not a weighted score

BM25 returns an unbounded TF-IDF sum, typically somewhere between 2 and 40.
Cosine similarity returns something in [-1, 1]. Adding them means normalising
first, and every normalisation is wrong somewhere:

- min-max is destroyed by a single outlier in the candidate list
- z-score assumes a distribution that neither retriever has
- a fixed scale factor has to be retuned whenever the corpus changes

RRF ignores the scores entirely and reads only the positions:

```
score(d) = Σ  1 / (k + rank(d))
```

Nothing to normalise and nothing to retune. The cost is real: RRF cannot tell
a confident first place from a marginal one, so a retriever that is certain
gets no extra say.

`k = 60` is from Cormack, Clarke and Buettcher (SIGIR 2009). It is large
enough that rank 1 and rank 2 sit close together (1/61 against 1/62), which
is what stops one retriever dominating the fused list.

## Where fusion actually earns its keep

`benchmarks/compare.py` splits the query set in two, and the split is the
whole point:

```
class         n    bm25    dense    fused
----------  ---  ------  -------  -------
identifier    4   0.875    0.312    0.875
natural       4   0.583    0.750    0.812
```

On natural-language questions dense wins and fusion adds a little. On
anything containing an identifier (`Article 92(1)(c)`, an ISIN, `RTS 28 field
14`) dense collapses, because an embedding of a reference number sits close
to every other reference number. BM25 handles those exactly, and fusion means
you do not have to classify the query first to pick a retriever.

That second row is the reason this exists. A pure-vector search demo looks
fine until someone pastes an ISIN into it.

## Two details that cost recall if you get them wrong

**bge wants an instruction prefix on queries and not on passages.** Getting it
backwards costs about 4 points of recall@10 and is invisible otherwise:
everything still returns plausible results. `dense.encode(..., is_query=True)`
is the only place that prefix is added.

**No stemmer in the Elasticsearch analyzer.** On a corpus full of identifiers,
stemming collapses "filing" and "filed" into one token, which is fine, and
mangles half the product codes, which is not.

## Degraded mode

`HybridEngine.search` runs both retrievers under `asyncio.gather(...,
return_exceptions=True)`. If Elasticsearch is down the dense half still
answers and the response says so: `ranks` will contain only `dense`. Both
failing raises. A half-degraded search beats a 500.

## Running it

```bash
docker compose up -d                   # elasticsearch + qdrant
pip install -e ".[dev,bench]"
uvicorn hybrid_search.api:app --reload
python benchmarks/compare.py
```

## Tests

`pytest -q`. Fusion is pure, so it is tested properly and runs in CI. The
retrievers need live services and do not.

The test worth reading is `test_scores_are_never_read_only_positions`: two
runs with identical orderings and wildly different score magnitudes have to
fuse identically. If that ever fails, someone has reintroduced score
normalisation.

## Rough edges

- The benchmark corpus is not in the repo. The numbers above came from an
  internal document set I cannot publish, and `benchmarks/queries.jsonl` has
  the queries with the doc ids but no documents behind them. Reproducing the
  table needs your own corpus.
- `weights` is plumbed through `HybridEngine` and never tuned. Equal weighting
  beat every hand-set pair I tried, which is either a real result or a sign
  the query set is too small to show a difference. Probably the latter.
- No cross-encoder stage. Fusion gets you an ordering, not a good top-3, and
  a re-ranker over the fused top-50 would be the obvious next thing.
