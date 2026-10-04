from __future__ import annotations

from contextlib import asynccontextmanager

from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from hybrid_search.dense import DenseIndex
from hybrid_search.engine import HybridEngine
from hybrid_search.sparse import SparseIndex

_engine: HybridEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    es = AsyncElasticsearch("http://localhost:9200")
    qd = AsyncQdrantClient(url="http://localhost:6333")
    sparse, dense = SparseIndex(es), DenseIndex(qd)
    await sparse.ensure()
    await dense.ensure()
    _engine = HybridEngine(sparse, dense)
    yield
    await es.close()


app = FastAPI(title="hybrid-search", version="0.9.1", lifespan=lifespan)


class Doc(BaseModel):
    doc_id: str
    text: str
    source: str = ""
    section: str = ""


class SearchResult(BaseModel):
    doc_id: str
    rrf_score: float
    ranks: dict[str, int]
    found_by_all: bool
    text: str


@app.post("/index")
async def index(docs: list[Doc]) -> dict:
    return await _engine.index([d.model_dump() for d in docs])


@app.get("/search")
async def search(q: str, limit: int = Field(10, ge=1, le=100)) -> list[SearchResult]:
    hits = await _engine.search(q, limit=limit)
    return [
        SearchResult(doc_id=h.doc_id, rrf_score=round(h.rrf_score, 6), ranks=h.ranks,
                     found_by_all=h.found_by_all, text=h.text)
        for h in hits
    ]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
