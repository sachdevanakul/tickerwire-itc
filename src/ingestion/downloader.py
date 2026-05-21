"""
src/ingestion/downloader.py
Downloads ITC Annual Reports from itcportal.com
"""

import asyncio
from pathlib import Path
from typing import Dict

import httpx

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Working ITC Annual Report URLs
ITC_ANNUAL_REPORT_URLS: Dict[str, str] = {
    "FY22": "https://itcportal.com/content/dam/itc-corporate/pdfs/report-and-accounts/ITC-Report-and-Accounts-2022.pdf",
    "FY23": "https://itcportal.com/content/dam/itc-corporate/pdfs/report-and-accounts/ITC-Report-and-Accounts-2023.pdf",
    "FY24": "https://itcportal.com/content/dam/itc-corporate/pdfs/report-and-accounts/ITC-Report-and-Accounts-2024.pdf",
    "FY25": "https://itcportal.com/content/dam/itc-corporate/pdfs/report-and-accounts/ITC-Report-and-Accounts-2025.pdf",
}

# Browser-like headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.itcportal.com/",
    "Connection": "keep-alive",
}


async def download_report(
    fiscal_year: str,
    url: str,
    data_dir: Path,
) -> Path:
    """
    Download a single annual report PDF.
    """

    output_path = data_dir / f"ITC_Annual_Report_{fiscal_year}.pdf"

    # Skip existing files
    if output_path.exists():
        logger.info(
            "report_already_exists",
            fiscal_year=fiscal_year,
            path=str(output_path),
        )
        return output_path

    logger.info(
        "downloading_report",
        fiscal_year=fiscal_year,
        url=url,
    )

    try:
        async with httpx.AsyncClient(
            timeout=180.0,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:

            response = await client.get(url)

            logger.info(
                "http_response",
                fiscal_year=fiscal_year,
                status_code=response.status_code,
            )

            response.raise_for_status()

            # Validate PDF response
            content_type = response.headers.get("content-type", "")

            if "pdf" not in content_type.lower():
                raise ValueError(
                    f"Invalid content-type received: {content_type}"
                )

            # Save PDF
            output_path.write_bytes(response.content)

        logger.info(
            "download_complete",
            fiscal_year=fiscal_year,
            size_mb=round(len(response.content) / 1e6, 2),
        )

        return output_path

    except httpx.HTTPStatusError as e:
        logger.error(
            "download_failed_http",
            fiscal_year=fiscal_year,
            status_code=e.response.status_code,
            url=url,
        )
        raise

    except Exception as e:
        logger.error(
            "download_failed",
            fiscal_year=fiscal_year,
            error=str(e),
        )
        raise


async def download_all_reports(
    data_dir: str = "./data/pdfs",
) -> Dict[str, Path]:
    """
    Download all annual reports.
    """

    pdf_dir = Path(data_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        download_report(fy, url, pdf_dir)
        for fy, url in ITC_ANNUAL_REPORT_URLS.items()
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    downloaded_reports = {}

    for fy, result in zip(
        ITC_ANNUAL_REPORT_URLS.keys(),
        results,
    ):

        if isinstance(result, Exception):
            logger.error(
                "report_skipped",
                fiscal_year=fy,
                error=str(result),
            )
            continue

        downloaded_reports[fy] = result

    logger.info(
        "all_downloads_complete",
        total_downloaded=len(downloaded_reports),
    )

    return downloaded_reports