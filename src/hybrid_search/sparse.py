"""BM25 half, in Elasticsearch."""
from __future__ import annotations

from elasticsearch import AsyncElasticsearch, helpers

from hybrid_search.fusion import Hit

MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "custom",
                    "tokenizer": "standard",
                    # No stemmer. On a corpus full of identifiers and statute
                    # references, stemming turns "filing" and "filed" into the
                    # same token and also mangles half the product codes.
                    "filter": ["lowercase", "asciifolding"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            # Exact-match field alongside the analysed one. Queries that look
            # like identifiers are routed here, where BM25 beats anything dense.
            "text_raw": {"type": "keyword", "ignore_above": 256},
            "source": {"type": "keyword"},
            "section": {"type": "keyword"},
        }
    },
}


class SparseIndex:
    def __init__(self, client: AsyncElasticsearch, index: str = "documents"):
        self.client = client
        self.index = index

    async def ensure(self) -> None:
        if not await self.client.indices.exists(index=self.index):
            await self.client.indices.create(index=self.index, body=MAPPING)

    async def bulk_index(self, docs: list[dict]) -> int:
        actions = [
            {
                "_index": self.index,
                "_id": d["doc_id"],
                "text": d["text"],
                "text_raw": d["text"][:256],
                "source": d.get("source", ""),
                "section": d.get("section", ""),
            }
            for d in docs
        ]
        ok, _ = await helpers.async_bulk(self.client, actions, refresh=True)
        return ok

    async def search(self, query: str, limit: int = 50) -> list[Hit]:
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"text": {"query": query, "boost": 1.0}}},
                        # Phrase match scores contiguous terms higher. Cheap,
                        # and it is most of why BM25 still wins on quoted
                        # statute references.
                        {"match_phrase": {"text": {"query": query, "boost": 2.0}}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": limit,
        }
        resp = await self.client.search(index=self.index, body=body)
        return [
            Hit(
                doc_id=h["_id"],
                score=h["_score"],
                text=h["_source"]["text"],
                metadata={"source": h["_source"].get("source", "")},
            )
            for h in resp["hits"]["hits"]
        ]
