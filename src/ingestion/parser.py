"""
src/ingestion/parser.py
Parses ITC Annual Report PDFs into structured text blocks.
Uses pdfplumber for tables, PyMuPDF for prose flow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber
import fitz  # PyMuPDF
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextBlock:
    """A parsed block of text from a PDF page."""
    page: int
    block_type: str  # "prose" | "table" | "heading"
    text: str
    fiscal_year: str
    source_file: str
    bbox: Optional[tuple] = None


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove PDF artefacts."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d)\s+(\d{3})", r"\1\2", text)  # Fix split numbers: "1 234" → "1234"
    return text.strip()


def _extract_tables(page: pdfplumber.page.Page, fiscal_year: str, source_file: str) -> List[TextBlock]:
    """Extract tables from a page using pdfplumber."""
    blocks = []
    tables = page.extract_tables()

    for table in tables:
        if not table:
            continue
        # Flatten table to pipe-delimited text representation
        rows = []
        for row in table:
            cells = [str(c or "").strip() for c in row]
            rows.append(" | ".join(cells))
        table_text = "\n".join(rows)

        if len(table_text.strip()) < 20:
            continue  # Skip near-empty tables

        blocks.append(TextBlock(
            page=page.page_number,
            block_type="table",
            text=table_text,
            fiscal_year=fiscal_year,
            source_file=source_file,
        ))

    return blocks


def _extract_prose(doc: fitz.Document, page_num: int, fiscal_year: str, source_file: str) -> List[TextBlock]:
    """Extract prose text blocks from a page using PyMuPDF."""
    page = doc[page_num]
    blocks = []

    raw_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    current_prose = []

    for block in raw_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block
            continue

        block_text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
            block_text += "\n"

        cleaned = _clean_text(block_text)
        if not cleaned or len(cleaned) < 15:
            continue

        # Detect headings by font size (approximate)
        font_sizes = [
            span.get("size", 10)
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10
        block_type = "heading" if avg_font_size >= 13 else "prose"

        current_prose.append(cleaned)

    if current_prose:
        blocks.append(TextBlock(
            page=page_num + 1,
            block_type="prose",
            text=" ".join(current_prose),
            fiscal_year=fiscal_year,
            source_file=source_file,
        ))

    return blocks


def parse_pdf(pdf_path: Path, fiscal_year: str) -> List[TextBlock]:
    """
    Parse an annual report PDF into structured text blocks.
    Handles tables separately for higher fidelity.
    """
    logger.info("parsing_pdf", path=str(pdf_path), fiscal_year=fiscal_year)
    all_blocks: List[TextBlock] = []
    source_file = pdf_path.name

    # Track pages with tables to avoid double-extracting prose
    pages_with_tables: set = set()

    with pdfplumber.open(pdf_path) as plumber_doc:
        for page in plumber_doc.pages:
            table_blocks = _extract_tables(page, fiscal_year, source_file)
            if table_blocks:
                pages_with_tables.add(page.page_number)
                all_blocks.extend(table_blocks)

    # Now extract prose using PyMuPDF
    mupdf_doc = fitz.open(str(pdf_path))
    for page_num in range(len(mupdf_doc)):
        page_1indexed = page_num + 1
        if page_1indexed in pages_with_tables:
            # Still extract prose from pages with tables, but separately
            pass
        prose_blocks = _extract_prose(mupdf_doc, page_num, fiscal_year, source_file)
        all_blocks.extend(prose_blocks)

    mupdf_doc.close()

    logger.info(
        "parsing_complete",
        fiscal_year=fiscal_year,
        total_blocks=len(all_blocks),
        table_blocks=len(pages_with_tables),
    )
    return all_blocks
