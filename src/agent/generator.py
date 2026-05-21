"""
src/agent/generator.py
Generates streamed answers with citations. Handles all 4 action types.
"""
from __future__ import annotations
import os
from typing import AsyncIterator, List, Dict, Optional
from openai import AsyncOpenAI
from src.agent.cost_guard import truncate_chunks_to_limit
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_DIRECT = """You are TickerWire's ITC research assistant. Answer journalist queries accurately using only the provided context.

Rules:
1. Answer ONLY from the provided context chunks. Do not use outside knowledge.
2. Every factual claim MUST be followed by a citation: [AR-FY{year}, p.{page}]
3. If context lacks enough data, say exactly what's missing.
4. Quote exact figures from reports — never approximate unless the report does.
5. Keep answers concise: 3–6 sentences for simple queries.
6. Do not speculate about future performance."""

SYSTEM_SYNTHESIS = """You are TickerWire's ITC research assistant specializing in year-over-year analysis.

Rules:
1. Answer ONLY from the provided context chunks.
2. For comparisons, use a markdown table: Metric | FY__ | FY__ | Change
3. Cite every figure: [AR-FY{year}, p.{page}]
4. Calculated metrics (growth %, margins) are allowed if inputs are cited — label as "Calculated".
5. If a year's data is missing, state it explicitly rather than omitting.
6. End with a one-sentence factual summary."""

CLARIFY_TEMPLATE = """To give you the most accurate answer, I need one clarification:

**{question}**

Available fiscal years in our database: FY2022, FY2023, FY2024, FY2025."""

REFUSE_TEMPLATE = """This query falls outside the TickerWire-ITC knowledge base.

**Reason:** {reason}

Our system covers ITC Limited's Annual Reports for FY22–FY25 only. If you believe this should be answerable from the reports, please rephrase your question."""


def _format_context(chunks: List[Dict]) -> str:
    parts = []
    for chunk in chunks:
        header = f"[Chunk: {chunk['chunk_id']} | {chunk.get('citation', 'Unknown')} | Type: {chunk.get('block_type', 'prose')}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


async def generate_stream(
    query: str,
    chunks: List[Dict],
    action: str,
    client: AsyncOpenAI,
    clarification_question: Optional[str] = None,
    refuse_reason: Optional[str] = None,
) -> AsyncIterator[str]:
    """Yield answer tokens as a stream. Handles all 4 action types."""

    # ── Non-retrieval actions: template-based, no LLM call ──────────────────
    if action == "clarify":
        yield CLARIFY_TEMPLATE.format(question=clarification_question or "Which fiscal year are you referring to?")
        return

    if action == "refuse":
        yield REFUSE_TEMPLATE.format(reason=refuse_reason or "Query is outside the available corpus.")
        return

    # ── Retrieval-based actions: stream from GPT-4o-mini ────────────────────
    system = SYSTEM_SYNTHESIS if action == "retrieve_then_answer" else SYSTEM_DIRECT
    safe_chunks = truncate_chunks_to_limit(chunks, query, system)
    context = _format_context(safe_chunks)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    logger.info("generating", action=action, chunks_used=len(safe_chunks))

    stream = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        max_tokens=800,
        temperature=0.1,
        stream=True,
    )

    async for event in stream:
        delta = event.choices[0].delta
        if delta.content:
            yield delta.content
