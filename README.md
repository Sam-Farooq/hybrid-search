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

## Where fusion earns its keep

`benchmarks/compare.py` reports recall@10 for BM25 alone, dense alone, and
fused, split by query class. The split is the point, not the totals.

Dense retrieval degrades badly on identifiers. An embedding of `Article
92(1)(c)` sits close to the embedding of every other article reference,
because the model has learned that reference numbers look like each other and
has no reason to encode which one this is. BM25 has the opposite property: it
matches the literal token and does not care what it means.

On natural-language questions the ordering reverses, and dense wins by
understanding a question that shares no vocabulary with its answer.

Fusion means you do not have to classify the query first to pick a retriever,
which is the part that matters in a system where users type whatever they
like. A pure-vector search demo looks fine until someone pastes an ISIN into
it.

### Running the benchmark

`benchmarks/queries.jsonl` holds eight queries, four of each class, with the
doc ids they should return. **The corpus is not included**, so the file is a
template rather than a reproducible result: point it at your own documents,
match the ids, and `python benchmarks/compare.py` prints the three-way
comparison.

I have deliberately not put numbers in this README. Eight queries is far too
few to publish an average from, and a recall figure means nothing without the
corpus it was measured over.

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

- Eight queries is a sketch, not an evaluation. A real comparison needs a
  corpus and on the order of a hundred queries per class, which is the work
  this repo has not done.
- `weights` is plumbed through `HybridEngine` and never tuned. Equal weighting
  beat every hand-set pair I tried, which is either a real result or a sign
  the query set is too small to show a difference. Probably the latter.
- No cross-encoder stage. Fusion gets you an ordering, not a good top-3, and
  a re-ranker over the fused top-50 would be the obvious next thing.
