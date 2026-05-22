from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import chromadb
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.ingestion.chunker import Chunk
from src.utils.logging import get_logger

logger = get_logger(__name__)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Much better for CPU
EMBED_BATCH_SIZE = 64

# TEMP LIMIT FOR ASSIGNMENT DEMO
MAX_CHUNKS = 1000

CHROMA_COLLECTION = "itc_annual_reports"


class ReportIndexer:
    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        bm25_path: str = "./data/bm25.pkl",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.bm25_path = Path(bm25_path)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("loading_embedding_model", model=EMBED_MODEL)

        self.encoder = SentenceTransformer(
            EMBED_MODEL,
            device="cpu",
        )

        # Better CPU inference
        torch.set_grad_enabled(False)

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        self._chunks: List[Chunk] = []

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.encoder.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def index_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            logger.error(
                "no_chunks_to_index",
                hint="PDF download likely failed — check URLs",
            )
            raise ValueError(
                "No chunks to index. Ensure PDFs downloaded successfully."
            )

        # LIMIT FOR FASTER ITERATION
        if len(chunks) > MAX_CHUNKS:
            logger.warning(
                "chunk_limit_applied",
                original=len(chunks),
                limited_to=MAX_CHUNKS,
            )
            chunks = chunks[:MAX_CHUNKS]

        logger.info(
            "indexing_start",
            total_chunks=len(chunks),
        )

        self._chunks = chunks

        existing_ids = set(
            self.collection.get(include=[])["ids"]
        )

        new_chunks = [
            c for c in chunks
            if c.chunk_id not in existing_ids
        ]

        if not new_chunks:
            logger.info(
                "all_chunks_already_indexed",
                skipped=len(chunks),
            )
        else:
            logger.info(
                "indexing_new_chunks",
                count=len(new_chunks),
            )

            self._index_chroma(new_chunks)

        self._build_bm25(chunks)

        logger.info(
            "indexing_complete",
            total=len(chunks),
        )

    def _index_chroma(self, chunks: List[Chunk]) -> None:
        for i in tqdm(
            range(0, len(chunks), EMBED_BATCH_SIZE),
            desc="Embedding batches",
        ):
            batch = chunks[i : i + EMBED_BATCH_SIZE]

            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]

            metadatas = [
                {
                    "fiscal_year": c.fiscal_year,
                    "source_file": c.source_file,
                    "page": c.page,
                    "block_type": c.block_type,
                    "token_count": c.token_count,
                    "citation": c.citation,
                }
                for c in batch
            ]

            embeddings = self._embed_batch(texts)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            logger.info(
                "indexed_batch",
                start=i,
                end=i + len(batch),
            )

    def _build_bm25(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError(
                "Cannot build BM25 index with 0 chunks."
            )

        tokenized_corpus = [
            c.text.lower().split()
            for c in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        with open(self.bm25_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": bm25,
                    "chunks": [c.to_dict() for c in chunks],
                },
                f,
            )

        logger.info(
            "bm25_built",
            corpus_size=len(chunks),
        )

    def get_collection(self):
        return self.collection

    def load_bm25(self):
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)

        return data["bm25"], data["chunks"]