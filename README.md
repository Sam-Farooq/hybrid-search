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

`pytest -q`. Fusion, the re-ranking stage and the resolution arithmetic are
pure, so they are tested properly and run in CI. The retrievers need live
services and do not.

Two tiers are deselected by default, so `pytest -q` downloads nothing:

```bash
pytest -q                      36 tests, no network, no services
pytest -q -m network           downloads SciFact, 2.8MB
pytest -q -m model             loads a real cross-encoder, 80MB on first run
```

The test worth reading is `test_scores_are_never_read_only_positions`: two
runs with identical orderings and wildly different score magnitudes have to
fuse identically. If that ever fails, someone has reintroduced score
normalisation.

The second one worth reading is
`test_the_shipped_query_file_can_only_move_in_eighths`. It derives the 0.125
figure below from `benchmarks/queries.jsonl` rather than quoting it, so
adding queries changes the test and that is the signal the benchmark's
resolution moved.

## Measuring anything at all

Three rough edges on this repository were one problem wearing three hats, and
the problem is arithmetic rather than retrieval.

**Recall@k is quantised.** A query with `r` relevant documents can only score
`0, 1/r, 2/r ... 1`, so a class mean over `n` queries moves in steps of
`1 / (n * lcm(r...))` and nothing finer exists. For the eight-query set this
repository shipped with, that is **0.125**. Every comparison made on it,
including every weighting tried by hand, was reading rounding.

`hybrid_search.evaluation.step` computes it, `benchmarks/compare.py` prints it
above the table before any number is believed, and
`benchmarks/corpus.py` builds a set fine enough to use:

```
$ python benchmarks/corpus.py
5183 documents -> benchmarks/scifact-corpus.jsonl
450 queries -> benchmarks/scifact.jsonl
  identifier: 150 queries, smallest visible difference 0.0067
  natural: 300 queries, smallest visible difference 0.0001
```

The corpus is **SciFact**, from BEIR, downloaded on demand into a gitignored
cache and never committed: it is CC BY-NC and redistributing it inside an MIT
repository would be wrong.

The two classes are not built the same way and the difference matters:

- **natural** is SciFact's own test queries with its own human relevance
  judgements. A real, published, human-judged set.
- **identifier** is *constructed*, because no general corpus ships statute
  references or ISINs. Tokens occurring in exactly one document become
  queries for that document, which gives an objective judgement and tests
  exactly what the class is about, lexical exact match where an embedding has
  nothing to work with. It is **not** human-judged and does not stand in for
  user satisfaction. It stands in for "can this system find a string".

**Weights.** `HybridEngine` takes them and `benchmarks/compare.py --sweep`
now searches them, over a cached retrieval so a grid search does not need
both services up for its whole length. The sweep also answers the question
that made the earlier attempt useless: it compares the winner's margin
against the resolution and prints `NOT distinguishable from equal weighting
on this query set` when the margin is smaller than one step. A sweep always
produces a winner; whether the winner means anything is a different question.

**No numbers are published here.** Running the benchmark needs Elasticsearch
and Qdrant, and the machine this work was done on has neither. The harness is
tested, the corpus build is reproduced above from a real run, and the recall
table is left to whoever runs it against the services. A table of unmeasured
numbers was removed from this README once already.

## The re-ranking stage

RRF orders by position agreement and never reads the retrievers' scores,
which is what makes it indifferent to BM25 and cosine living on incomparable
scales. The cost is that it cannot tell a confident first place from a
marginal one: good for recall, mediocre for the top 3, which is the part a
user sees.

`hybrid_search.rerank` is the stage that fixes it, and it is a separate stage
rather than a change to fusion, so fusion stays pure and the expensive model
only ever sees the fused window:

```python
engine = HybridEngine(sparse, dense, reranker=CrossEncoderReranker())
```

The engine fuses everything, reranks the top 50, then applies the caller's
limit. That order matters: fusing to a limit of 10 and reranking those would
hand the model the ten documents RRF already liked, which is the ordering the
stage exists to doubt.

A reranker that raises degrades to the fused order with a warning, on the
same rule as a dead retriever: a worse ordering beats a 500.

## Rough edges

- No numbers in this README, for the reason above: the services were not
  available where the work was done.
- The identifier class is constructed rather than judged, which is stated
  wherever it is used but is still a weaker claim than the natural class.
- The sweep searches one ratio with the other retriever pinned at 1.0. Only
  the ratio changes the RRF ordering, so a two-dimensional grid would spend
  most of its time re-evaluating the same ordering under a different name,
  but that stops being true the moment a third retriever is added.
