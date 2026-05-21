"""
src/retrieval/bm25.py
BM25 keyword retrieval using rank_bm25.
"""
from __future__ import annotations
import os, pickle
from typing import List, Dict
from rank_bm25 import BM25Okapi
from src.utils.logging import get_logger

logger = get_logger(__name__)

class BM25Retriever:
    def __init__(self, bm25_path: str = "./data/bm25.pkl"):
        self.bm25_path = bm25_path
        self._bm25: BM25Okapi = None
        self._chunks: List[Dict] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {self.bm25_path}. Run scripts/ingest.py first.")
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        logger.info("bm25_loaded", corpus_size=len(self._chunks))

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """Retrieve top_k chunks using BM25 keyword scoring."""
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        chunks = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self._chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["source"] = "bm25"
            chunks.append(chunk)

        logger.info("bm25_retrieved", count=len(chunks), top_score=chunks[0]["score"] if chunks else 0)
        return chunks
