"""
src/api/main.py
FastAPI application. Exposes POST /query (SSE streaming) and GET /health.
"""
from __future__ import annotations
import os, time, uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

from openai import AsyncOpenAI

from src.api.models import QueryRequest, HealthResponse
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CohereReranker
from src.agent.router import route_query
from src.agent.generator import generate_stream
from src.utils.logging import setup_logging, get_logger
from src.utils.tracing import setup_tracing, get_tracer

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# ── Globals (initialized at startup) ────────────────────────────────────────
_retriever: HybridRetriever = None
_reranker: CohereReranker = None
_openai: AsyncOpenAI = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _reranker, _openai
    logger.info("startup_begin")
    setup_tracing()
    _openai = AsyncOpenAI()
    _retriever = HybridRetriever()
    _reranker = CohereReranker()
    logger.info("startup_complete")
    yield
    logger.info("shutdown")


app = FastAPI(title="TickerWire-ITC Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── /health ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        count = _retriever.dense.collection.count()
        return HealthResponse(status="ok", chroma_chunks=count, bm25_loaded=True)
    except Exception as e:
        return HealthResponse(status=f"degraded: {e}")


# ── /query ───────────────────────────────────────────────────────────────────
@app.post("/query")
async def query_endpoint(req: QueryRequest):
    query_id = str(uuid.uuid4())[:8]
    tracer = get_tracer()

    async def event_stream():
        start = time.perf_counter()

        with tracer.start_as_current_span("agent.pipeline") as span:
            span.set_attribute("query_id", query_id)
            span.set_attribute("query", req.query)

            # ── 1. Route ──────────────────────────────────────────────────
            with tracer.start_as_current_span("agent.route"):
                decision = await route_query(req.query, _openai)
            span.set_attribute("action", decision.action)

            # ── 2. Retrieve (skip for clarify/refuse) ─────────────────────
            chunks = []
            if decision.action in ("direct_answer", "retrieve_then_answer"):
                with tracer.start_as_current_span("retrieval.hybrid"):
                    top_k = int(os.getenv("DENSE_TOP_K", "20"))
                    chunks = await _retriever.retrieve(req.query, top_k=top_k)

                with tracer.start_as_current_span("retrieval.rerank"):
                    rerank_n = int(os.getenv("RERANK_TOP_N", "5"))
                    chunks = _reranker.rerank(req.query, chunks, top_n=rerank_n)

            # ── 3. Stream generation ───────────────────────────────────────
            with tracer.start_as_current_span("generation.stream"):
                # Emit metadata event first
                meta = {"query_id": query_id, "action": decision.action, "chunks": len(chunks)}
                yield f"data: {json.dumps({'type': 'meta', **meta})}\n\n"

                full_text = []
                async for token in generate_stream(
                    query=req.query,
                    chunks=chunks,
                    action=decision.action,
                    client=_openai,
                    clarification_question=decision.clarification_question,
                    refuse_reason=decision.reason if decision.action == "refuse" else None,
                ):
                    full_text.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                latency_ms = round((time.perf_counter() - start) * 1000, 1)

                # Emit done event with citations
                citations = [
                    {
                        "chunk_id": c["chunk_id"],
                        "citation": c.get("citation", ""),
                        "page": c.get("page", 0),
                        "fiscal_year": c.get("fiscal_year", ""),
                        "score": round(c.get("rerank_score", c.get("rrf_score", 0)), 4),
                    }
                    for c in chunks
                ]
                yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'latency_ms': latency_ms})}\n\n"

                logger.info(
                    "query_complete",
                    query_id=query_id,
                    action=decision.action,
                    chunks=len(chunks),
                    latency_ms=latency_ms,
                    answer_len=len("".join(full_text)),
                )

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"X-Query-ID": query_id, "Cache-Control": "no-cache"})
