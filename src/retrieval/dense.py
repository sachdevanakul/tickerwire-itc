"""
src/retrieval/dense.py
Dense vector retrieval using ChromaDB.
"""
from __future__ import annotations
import os
from typing import List, Dict
import chromadb
from openai import AsyncOpenAI
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DenseRetriever:
    def __init__(self, persist_dir: str = None):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("itc_annual_reports")
        self.openai = AsyncOpenAI()

    async def retrieve(self, query_text: str, top_k: int = 20) -> List[Dict]:
        """Embed query and retrieve top_k chunks by cosine similarity."""
        response = await self.openai.embeddings.create(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            input=query_text,
        )
        query_embedding = response.data[0].embedding

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": doc,
                "score": 1 - dist,  # cosine distance → similarity
                "fiscal_year": meta.get("fiscal_year"),
                "page": meta.get("page"),
                "citation": meta.get("citation"),
                "block_type": meta.get("block_type"),
                "source": "dense",
            })

        logger.info("dense_retrieved", count=len(chunks), top_score=chunks[0]["score"] if chunks else 0)
        return chunks
