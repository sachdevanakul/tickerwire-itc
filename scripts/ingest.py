"""
scripts/ingest.py
One-shot ingestion pipeline:
  1. Download ITC Annual Reports (FY22–FY25) from itcportal.com
  2. Parse each PDF (text + tables)
  3. Chunk into 512-token pieces
  4. Embed and index into ChromaDB + BM25

Run: python scripts/ingest.py
"""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.downloader import download_all_reports
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_blocks
from src.ingestion.indexer import ReportIndexer
from src.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("ingest")


async def main():
    logger.info("ingestion_start")

    # 1. Download PDFs
    print("\n📥 Step 1/4: Downloading ITC Annual Reports...")
    pdf_paths = await download_all_reports("./data/pdfs")
    print(f"   ✓ Downloaded {len(pdf_paths)} reports\n")

    # 2. Parse all PDFs
    print("📄 Step 2/4: Parsing PDFs (text + tables)...")
    all_blocks = []
    for fy, path in sorted(pdf_paths.items()):
        blocks = parse_pdf(path, fiscal_year=fy)
        all_blocks.extend(blocks)
        print(f"   ✓ {fy}: {len(blocks)} blocks extracted")
    print(f"   Total blocks: {len(all_blocks)}\n")

    # 3. Chunk
    print("✂️  Step 3/4: Chunking (512 tokens, 64 overlap)...")
    all_chunks = chunk_blocks(all_blocks)
    print(f"   ✓ {len(all_chunks)} chunks created\n")

    # 4. Index
    print("🔢 Step 4/4: Embedding + Indexing (this takes ~15 min)...")
    indexer = ReportIndexer(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        bm25_path="./data/bm25.pkl",
    )
    indexer.index_chunks(all_chunks)
    print(f"\n✅ Ingestion complete!")
    print(f"   ChromaDB: {os.getenv('CHROMA_PERSIST_DIR', './data/chroma')}")
    print(f"   BM25 index: ./data/bm25.pkl")
    print(f"   Total chunks indexed: {len(all_chunks)}")
    print("\n🚀 Ready! Run: uvicorn src.api.main:app --reload --port 8000\n")


if __name__ == "__main__":
    asyncio.run(main())


