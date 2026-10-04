"""Dense half, in Qdrant."""
from __future__ import annotations

import asyncio
from functools import lru_cache

from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer

from hybrid_search.fusion import Hit

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384


@lru_cache(maxsize=1)
def _encoder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


async def encode(texts: list[str], is_query: bool = False) -> list[list[float]]:
    # bge wants an instruction prefix on queries but not on passages. Getting
    # this backwards costs about 4 points of recall@10 and is invisible
    # otherwise, because everything still returns plausible-looking results.
    prepared = [f"Represent this sentence for searching relevant passages: {t}"
                for t in texts] if is_query else texts

    def _run():
        return _encoder().encode(prepared, normalize_embeddings=True).tolist()

    return await asyncio.to_thread(_run)


class DenseIndex:
    def __init__(self, client: AsyncQdrantClient, collection: str = "documents"):
        self.client = client
        self.collection = collection

    async def ensure(self) -> None:
        existing = {c.name for c in (await self.client.get_collections()).collections}
        if self.collection not in existing:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
            )

    async def upsert(self, docs: list[dict]) -> int:
        vectors = await encode([d["text"] for d in docs])
        await self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=d["doc_id"], vector=v,
                    payload={"text": d["text"], "source": d.get("source", "")},
                )
                for d, v in zip(docs, vectors)
            ],
        )
        return len(docs)

    async def search(self, query: str, limit: int = 50) -> list[Hit]:
        vector = (await encode([query], is_query=True))[0]
        resp = await self.client.query_points(
            collection_name=self.collection, query=vector,
            limit=limit, with_payload=True,
        )
        return [
            Hit(doc_id=str(p.id), score=p.score, text=p.payload["text"],
                metadata={"source": p.payload.get("source", "")})
            for p in resp.points
        ]
