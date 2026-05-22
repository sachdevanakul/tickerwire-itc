"""
src/api/main.py
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    QueryRequest,
    HealthResponse,
)

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CohereReranker

from src.agent.router import route_query
from src.agent.generator import generate_stream

from src.utils.logging import (
    setup_logging,
    get_logger,
)

from src.utils.tracing import (
    setup_tracing,
    get_tracer,
)

setup_logging(
    os.getenv("LOG_LEVEL", "INFO")
)

logger = get_logger(__name__)

_retriever: HybridRetriever = None
_reranker: CohereReranker = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global _retriever, _reranker

    logger.info("startup_begin")

    setup_tracing()

    _retriever = HybridRetriever()
    _reranker = CohereReranker()

    logger.info("startup_complete")

    yield

    logger.info("shutdown")


app = FastAPI(
    title="TickerWire-ITC Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():

    return {
        "message": "TickerWire ITC Assistant API running"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health():

    try:

        count = (
            _retriever
            .dense
            .collection
            .count()
        )

        return HealthResponse(
            status="ok",
            chroma_chunks=count,
            bm25_loaded=True,
        )

    except Exception as e:

        return HealthResponse(
            status=f"degraded: {e}"
        )


@app.post("/query")
async def query_endpoint(
    req: QueryRequest
):

    start = time.perf_counter()

    query_id = str(uuid.uuid4())[:8]

    tracer = get_tracer()

    with tracer.start_as_current_span(
        "agent.pipeline"
    ) as span:

        span.set_attribute(
            "query_id",
            query_id,
        )

        span.set_attribute(
            "query",
            req.query,
        )

        # Route

        decision = await route_query(
            req.query
        )

        # Retrieve

        chunks = []

        if decision.action in (
            "direct_answer",
            "retrieve_then_answer",
        ):

            top_k = int(
                os.getenv(
                    "DENSE_TOP_K",
                    "20",
                )
            )

            chunks = await _retriever.retrieve(
                req.query,
                top_k=top_k,
            )

            rerank_n = int(
                os.getenv(
                    "RERANK_TOP_N",
                    "5",
                )
            )

            chunks = _reranker.rerank(
                req.query,
                chunks,
                top_n=rerank_n,
            )

        # Generate answer

        answer_parts = []

        async for token in generate_stream(
            query=req.query,
            chunks=chunks,
            action=decision.action,
            clarification_question=(
                decision.clarification_question
            ),
            refuse_reason=(
                decision.reason
                if decision.action == "refuse"
                else None
            ),
        ):

            answer_parts.append(token)

        answer = "".join(answer_parts)

        latency_ms = round(
            (
                time.perf_counter()
                - start
            ) * 1000,
            1,
        )

        citations = [
            {
                "chunk_id": c["chunk_id"],
                "citation": c.get(
                    "citation",
                    "",
                ),
                "page": c.get(
                    "page",
                    0,
                ),
                "fiscal_year": c.get(
                    "fiscal_year",
                    "",
                ),
                "score": round(
                    c.get(
                        "rerank_score",
                        c.get(
                            "rrf_score",
                            0,
                        ),
                    ),
                    4,
                ),
            }
            for c in chunks
        ]

        logger.info(
            "query_complete",
            query_id=query_id,
            action=decision.action,
            chunks=len(chunks),
            latency_ms=latency_ms,
        )

        return {
            "query_id": query_id,
            "action": decision.action,
            "answer": answer,
            "citations": citations,
            "latency_ms": latency_ms,
        }