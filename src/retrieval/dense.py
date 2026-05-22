"""
src/retrieval/dense.py
Dense vector retrieval using ChromaDB + local SentenceTransformer embeddings.
"""
from __future__ import annotations
import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from src.utils.logging import get_logger

logger = get_logger(__name__)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

class DenseRetriever:
    def __init__(self, persist_dir: str = None):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("itc_annual_reports")
        self.encoder = SentenceTransformer(EMBED_MODEL)

    async def retrieve(self, query_text: str, top_k: int = 20) -> List[Dict]:
        """Embed query and retrieve top_k chunks by cosine similarity."""
        # SentenceTransformer is sync — fine to call directly in async context for CPU work
        query_embedding = self.encoder.encode(
            query_text, normalize_embeddings=True
        ).tolist()

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
