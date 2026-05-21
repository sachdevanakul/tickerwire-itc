"""
src/retrieval/hyde.py
HyDE: Hypothetical Document Embeddings
Generates a hypothetical answer to improve dense retrieval against long financial passages.
"""

from __future__ import annotations

import os
from openai import AsyncOpenAI
from src.utils.logging import get_logger

logger = get_logger(__name__)

HYDE_PROMPT = """You are a financial analyst who has read ITC Limited's annual reports thoroughly.
Write a concise, factual passage (120–160 words) that would appear in an ITC annual report and directly answer the question below.
Use specific financial terminology, include plausible numbers (you may approximate), and write in the formal register of an Indian annual report.
Do NOT say "I don't know" or hedge — write as if this is a real passage from the report.
This passage will be used for document retrieval only, not shown to users.

Question: {query}

Write the hypothetical passage:"""


async def generate_hypothetical_document(query: str, client: AsyncOpenAI) -> str:
    """
    Generate a hypothetical document that answers the query.
    This is embedded and used for retrieval instead of (or alongside) the raw query.
    """
    if not os.getenv("HYDE_ENABLED", "true").lower() == "true":
        return query

    try:
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "user", "content": HYDE_PROMPT.format(query=query)}
            ],
            max_tokens=250,
            temperature=0.3,
        )
        hypothetical = response.choices[0].message.content.strip()
        logger.info("hyde_generated", query_len=len(query), hypo_len=len(hypothetical))
        return hypothetical

    except Exception as e:
        logger.warning("hyde_failed", error=str(e), fallback="raw_query")
        return query  # Graceful fallback


async def embed_text(text: str, client: AsyncOpenAI) -> list[float]:
    """Embed text using OpenAI embeddings."""
    response = await client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        input=text,
    )
    return response.data[0].embedding
