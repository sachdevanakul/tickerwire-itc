"""
src/api/models.py
Request and response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Journalist's question")
    fiscal_year: Optional[str] = Field(None, description="Optional: FY22/FY23/FY24/FY25")
    stream: bool = Field(True, description="Stream response via SSE")


class ChunkCitation(BaseModel):
    chunk_id: str
    citation: str
    page: int
    fiscal_year: str
    relevance_score: float


class QueryResponse(BaseModel):
    query_id: str
    action: str
    answer: str
    citations: List[ChunkCitation] = []
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    chroma_chunks: int = 0
    bm25_loaded: bool = False
