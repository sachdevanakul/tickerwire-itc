"""
src/retrieval/reranker.py
Cohere reranker: takes top-20 hybrid candidates → returns top-5.
"""
from __future__ import annotations
import os
from typing import List, Dict
import cohere
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CohereReranker:
    def __init__(self):
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        self.model = "rerank-english-v3.0"

    def rerank(self, query: str, chunks: List[Dict], top_n: int = 5) -> List[Dict]:
        """Rerank chunks using Cohere cross-encoder. Returns top_n chunks."""
        if not chunks:
            return []

        documents = [c["text"] for c in chunks]

        results = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        reranked = []
        for result in results.results:
            chunk = chunks[result.index].copy()
            chunk["rerank_score"] = result.relevance_score
            reranked.append(chunk)

        logger.info("reranked", input=len(chunks), output=len(reranked),
                    top_score=reranked[0]["rerank_score"] if reranked else 0)
        return reranked
