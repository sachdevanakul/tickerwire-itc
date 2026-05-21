"""
src/ingestion/downloader.py
Downloads ITC Annual Reports from itcportal.com
"""

import os
import httpx
import asyncio
from pathlib import Path
from typing import Dict
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Public URLs from itcportal.com
ITC_ANNUAL_REPORT_URLS: Dict[str, str] = {
    "FY22": "https://www.itcportal.com/investor/annual-report/2022/itc-annual-report-2022.pdf",
    "FY23": "https://www.itcportal.com/investor/annual-report/2023/itc-annual-report-2023.pdf",
    "FY24": "https://www.itcportal.com/investor/annual-report/2024/itc-annual-report-2024.pdf",
    "FY25": "https://www.itcportal.com/investor/annual-report/2025/itc-annual-report-2025.pdf",
}


async def download_report(fiscal_year: str, url: str, data_dir: Path) -> Path:
    """Download a single annual report PDF."""
    output_path = data_dir / f"ITC_Annual_Report_{fiscal_year}.pdf"

    if output_path.exists():
        logger.info("report_already_exists", fiscal_year=fiscal_year, path=str(output_path))
        return output_path

    logger.info("downloading_report", fiscal_year=fiscal_year, url=url)

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        output_path.write_bytes(response.content)

    logger.info("download_complete", fiscal_year=fiscal_year, size_mb=round(len(response.content) / 1e6, 2))
    return output_path


async def download_all_reports(data_dir: str = "./data/pdfs") -> Dict[str, Path]:
    """Download all four annual reports."""
    pdf_dir = Path(data_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        download_report(fy, url, pdf_dir)
        for fy, url in ITC_ANNUAL_REPORT_URLS.items()
    ]

    paths = await asyncio.gather(*tasks)
    return {fy: path for fy, path in zip(ITC_ANNUAL_REPORT_URLS.keys(), paths)}
