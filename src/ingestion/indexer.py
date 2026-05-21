# """
# src/ingestion/indexer.py
# Embeds chunks and stores them in ChromaDB (dense) and BM25 index (keyword).
# Also maintains a KPI metadata store for deterministic lookups.
# """

# from __future__ import annotations

# import json
# import pickle
# from pathlib import Path
# from typing import List, Dict, Any

# import chromadb
# from openai import OpenAI
# from rank_bm25 import BM25Okapi

# from src.ingestion.chunker import Chunk
# from src.utils.logging import get_logger

# logger = get_logger(__name__)

# EMBED_MODEL = "text-embedding-3-small"
# EMBED_BATCH_SIZE = 100
# CHROMA_COLLECTION = "itc_annual_reports"


# class ReportIndexer:
#     """Indexes chunks into ChromaDB and BM25 for hybrid retrieval."""

#     def __init__(self, persist_dir: str = "./data/chroma", bm25_path: str = "./data/bm25.pkl"):
#         self.persist_dir = Path(persist_dir)
#         self.persist_dir.mkdir(parents=True, exist_ok=True)
#         self.bm25_path = Path(bm25_path)
#         self.bm25_path.parent.mkdir(parents=True, exist_ok=True)

#         self.openai = OpenAI()
#         self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
#         self.collection = self.chroma_client.get_or_create_collection(
#             name=CHROMA_COLLECTION,
#             metadata={"hnsw:space": "cosine"},
#         )
#         self._chunks: List[Chunk] = []

#     def _embed_batch(self, texts: List[str]) -> List[List[float]]:
#         """Embed a batch of texts using OpenAI embeddings."""
#         response = self.openai.embeddings.create(
#             model=EMBED_MODEL,
#             input=texts,
#         )
#         return [item.embedding for item in response.data]

#     def index_chunks(self, chunks: List[Chunk]) -> None:
#         """Embed and index all chunks."""
#         logger.info("indexing_start", total_chunks=len(chunks))
#         self._chunks = chunks

#         # Check which chunks are already indexed
#         existing_ids = set(self.collection.get(include=[])["ids"])

#         new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
#         if not new_chunks:
#             logger.info("all_chunks_already_indexed", skipped=len(chunks))
#         else:
#             logger.info("indexing_new_chunks", count=len(new_chunks))
#             self._index_chroma(new_chunks)

#         # Always rebuild BM25 (fast)
#         self._build_bm25(chunks)
#         logger.info("indexing_complete")

#     def _index_chroma(self, chunks: List[Chunk]) -> None:
#         """Batch-embed and upsert chunks into ChromaDB."""
#         for i in range(0, len(chunks), EMBED_BATCH_SIZE):
#             batch = chunks[i: i + EMBED_BATCH_SIZE]
#             texts = [c.text for c in batch]
#             ids = [c.chunk_id for c in batch]
#             metadatas = [
#                 {
#                     "fiscal_year": c.fiscal_year,
#                     "source_file": c.source_file,
#                     "page": c.page,
#                     "block_type": c.block_type,
#                     "token_count": c.token_count,
#                     "citation": c.citation,
#                 }
#                 for c in batch
#             ]

#             embeddings = self._embed_batch(texts)

#             self.collection.upsert(
#                 ids=ids,
#                 embeddings=embeddings,
#                 documents=texts,
#                 metadatas=metadatas,
#             )
#             logger.info("indexed_batch", start=i, end=i + len(batch))

#     def _build_bm25(self, chunks: List[Chunk]) -> None:
#         """Build and persist BM25 index."""
#         tokenized_corpus = [c.text.lower().split() for c in chunks]
#         bm25 = BM25Okapi(tokenized_corpus)

#         with open(self.bm25_path, "wb") as f:
#             pickle.dump({"bm25": bm25, "chunks": [c.to_dict() for c in chunks]}, f)

#         logger.info("bm25_built", corpus_size=len(chunks))

#     def get_collection(self) -> chromadb.Collection:
#         return self.collection

#     def load_bm25(self) -> tuple[BM25Okapi, List[Dict]]:
#         with open(self.bm25_path, "rb") as f:
#             data = pickle.load(f)
#         return data["bm25"], data["chunks"]


"""
src/ingestion/indexer.py
Embeds chunks and stores in ChromaDB (dense) and BM25 (keyword).
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Dict

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.utils.logging import get_logger

logger = get_logger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100
CHROMA_COLLECTION = "itc_annual_reports"


class ReportIndexer:
    def __init__(self, persist_dir: str = "./data/chroma", bm25_path: str = "./data/bm25.pkl"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_path = Path(bm25_path)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)

        self.openai = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks: List[Chunk] = []

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.openai.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def index_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            logger.error("no_chunks_to_index", hint="PDF download likely failed — check URLs")
            raise ValueError("No chunks to index. Ensure PDFs downloaded successfully.")

        logger.info("indexing_start", total_chunks=len(chunks))
        self._chunks = chunks

        existing_ids = set(self.collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("all_chunks_already_indexed", skipped=len(chunks))
        else:
            logger.info("indexing_new_chunks", count=len(new_chunks))
            self._index_chroma(new_chunks)

        self._build_bm25(chunks)
        logger.info("indexing_complete", total=len(chunks))

    def _index_chroma(self, chunks: List[Chunk]) -> None:
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i: i + EMBED_BATCH_SIZE]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [{
                "fiscal_year": c.fiscal_year,
                "source_file": c.source_file,
                "page": c.page,
                "block_type": c.block_type,
                "token_count": c.token_count,
                "citation": c.citation,
            } for c in batch]

            embeddings = self._embed_batch(texts)
            self.collection.upsert(ids=ids, embeddings=embeddings,
                                   documents=texts, metadatas=metadatas)
            logger.info("indexed_batch", start=i, end=i + len(batch))

    def _build_bm25(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 index with 0 chunks.")
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": [c.to_dict() for c in chunks]}, f)
        logger.info("bm25_built", corpus_size=len(chunks))

    def get_collection(self):
        return self.collection

    def load_bm25(self):
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["chunks"]
# EOF
# echo "indexer fixed"