"""
src/retrieval/hybrid.py
Hybrid retrieval: Dense + BM25 merged with Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations
import asyncio
from typing import List, Dict
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hyde import generate_hypothetical_document, embed_text
from openai import AsyncOpenAI
from src.utils.logging import get_logger

logger = get_logger(__name__)
RRF_K = 60  # Standard RRF constant


def _reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """Merge multiple ranked lists using RRF. Returns merged list sorted by RRF score."""
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = chunk

    merged = sorted(chunk_map.values(), key=lambda c: scores[c["chunk_id"]], reverse=True)
    for chunk in merged:
        chunk["rrf_score"] = scores[chunk["chunk_id"]]

    return merged


class HybridRetriever:
    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.openai_client = AsyncOpenAI()

    async def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Full hybrid retrieval pipeline:
        1. Generate HyDE hypothetical document
        2. Run dense + BM25 concurrently
        3. Merge with RRF
        """
        # Step 1: HyDE expansion (run concurrently with BM25)
        hyde_task = generate_hypothetical_document(query, self.openai_client)
        bm25_task = asyncio.get_event_loop().run_in_executor(None, self.bm25.retrieve, query, top_k)

        hypothetical_doc, bm25_results = await asyncio.gather(hyde_task, bm25_task)

        # Step 2: Dense retrieval on HyDE-expanded query
        dense_results = await self.dense.retrieve(hypothetical_doc, top_k=top_k)

        # Step 3: Merge with RRF
        merged = _reciprocal_rank_fusion([dense_results, bm25_results])
        logger.info("hybrid_merged", total=len(merged), returning=top_k)
        return merged[:top_k]
