"""
src/ingestion/chunker.py
Chunks parsed text blocks into retrieval-ready units.
- Prose: sliding window (512 tokens, 64-token overlap)
- Tables: kept atomic (entire table as one chunk)
- Headings: prepended to following prose for context
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, asdict
from typing import List

import tiktoken

from src.ingestion.parser import TextBlock
from src.utils.logging import get_logger

logger = get_logger(__name__)

CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 64
MIN_CHUNK_TOKENS = 50

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A retrieval-ready text chunk with metadata."""
    chunk_id: str
    text: str
    fiscal_year: str
    source_file: str
    page: int
    block_type: str  # prose | table | heading+prose
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return f"AR-{self.fiscal_year}, p.{self.page}"


def _tokenize(text: str) -> List[int]:
    return enc.encode(text)


def _detokenize(tokens: List[int]) -> str:
    return enc.decode(tokens)


def _chunk_prose(block: TextBlock) -> List[Chunk]:
    """Sliding window chunking for prose blocks."""
    tokens = _tokenize(block.text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]

        if len(chunk_tokens) < MIN_CHUNK_TOKENS:
            # Too small — merge with previous chunk if possible
            if chunks:
                prev = chunks[-1]
                merged_text = prev.text + " " + _detokenize(chunk_tokens)
                merged_tokens = _tokenize(merged_text)
                chunks[-1] = Chunk(
                    chunk_id=prev.chunk_id,
                    text=merged_text,
                    fiscal_year=block.fiscal_year,
                    source_file=block.source_file,
                    page=block.page,
                    block_type=block.block_type,
                    token_count=len(merged_tokens),
                )
            break

        chunk_text = _detokenize(chunk_tokens)
        chunks.append(Chunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
            text=chunk_text,
            fiscal_year=block.fiscal_year,
            source_file=block.source_file,
            page=block.page,
            block_type=block.block_type,
            token_count=len(chunk_tokens),
        ))

        # Advance by chunk size minus overlap
        start += CHUNK_SIZE_TOKENS - OVERLAP_TOKENS

    return chunks


def _chunk_table(block: TextBlock) -> List[Chunk]:
    """Tables are kept atomic. If too large, split at row boundaries."""
    tokens = _tokenize(block.text)

    if len(tokens) <= CHUNK_SIZE_TOKENS * 2:
        return [Chunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
            text=block.text,
            fiscal_year=block.fiscal_year,
            source_file=block.source_file,
            page=block.page,
            block_type="table",
            token_count=len(tokens),
        )]

    # Split large tables at row boundaries (lines)
    rows = block.text.split("\n")
    sub_tables = []
    current_rows: List[str] = []
    current_tokens = 0

    for row in rows:
        row_tokens = len(_tokenize(row))
        if current_tokens + row_tokens > CHUNK_SIZE_TOKENS and current_rows:
            sub_tables.append("\n".join(current_rows))
            current_rows = [row]
            current_tokens = row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens

    if current_rows:
        sub_tables.append("\n".join(current_rows))

    return [
        Chunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
            text=sub_text,
            fiscal_year=block.fiscal_year,
            source_file=block.source_file,
            page=block.page,
            block_type="table",
            token_count=len(_tokenize(sub_text)),
        )
        for sub_text in sub_tables
    ]


def chunk_blocks(blocks: List[TextBlock]) -> List[Chunk]:
    """
    Convert parsed text blocks into retrieval chunks.
    Headings are prepended to the next prose block for context.
    """
    chunks: List[Chunk] = []
    pending_heading: str = ""

    for block in blocks:
        if block.block_type == "heading":
            pending_heading = block.text
            continue

        if block.block_type == "table":
            table_chunks = _chunk_table(block)
            if pending_heading and table_chunks:
                # Prepend heading to first table chunk for context
                first = table_chunks[0]
                enriched_text = f"[{pending_heading}]\n{first.text}"
                table_chunks[0] = Chunk(
                    chunk_id=first.chunk_id,
                    text=enriched_text,
                    fiscal_year=first.fiscal_year,
                    source_file=first.source_file,
                    page=first.page,
                    block_type="table",
                    token_count=len(_tokenize(enriched_text)),
                )
                pending_heading = ""
            chunks.extend(table_chunks)

        else:  # prose
            if pending_heading:
                block = TextBlock(
                    page=block.page,
                    block_type="prose",
                    text=f"{pending_heading}. {block.text}",
                    fiscal_year=block.fiscal_year,
                    source_file=block.source_file,
                )
                pending_heading = ""
            chunks.extend(_chunk_prose(block))

    logger.info("chunking_complete", total_chunks=len(chunks))
    return chunks
