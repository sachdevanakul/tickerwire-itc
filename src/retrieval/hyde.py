"""
src/retrieval/hyde.py
HyDE: Hypothetical Document Embeddings
"""

from __future__ import annotations

import os

from groq import AsyncGroq
from sentence_transformers import SentenceTransformer

from src.utils.logging import get_logger

logger = get_logger(__name__)

_encoder = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

HYDE_PROMPT = """
You are a financial analyst who has read ITC Limited annual reports thoroughly.

Write a concise factual passage that directly answers the question below.

Question: {query}

Write the hypothetical passage:
"""


async def generate_hypothetical_document(
    query: str,
) -> str:

    if (
        os.getenv(
            "HYDE_ENABLED",
            "true",
        ).lower()
        != "true"
    ):
        return query

    try:

        client = AsyncGroq(
            api_key=os.getenv(
                "GROQ_API_KEY"
            )
        )

        response = await client.chat.completions.create(
            model=os.getenv(
                "GROQ_MODEL",
                "llama-3.1-8b-instant",
            ),
            messages=[
                {
                    "role": "user",
                    "content": HYDE_PROMPT.format(
                        query=query
                    ),
                }
            ],
            max_tokens=250,
            temperature=0.3,
        )

        hypothetical = (
            response
            .choices[0]
            .message.content
            .strip()
        )

        logger.info(
            "hyde_generated",
            query_len=len(query),
            hypo_len=len(hypothetical),
        )

        return hypothetical

    except Exception as e:

        logger.warning(
            "hyde_failed",
            error=str(e),
            fallback="raw_query",
        )

        return query


async def embed_text(
    text: str,
) -> list[float]:

    return _encoder.encode(
        text,
        normalize_embeddings=True,
    ).tolist()