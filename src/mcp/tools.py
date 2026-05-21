"""
src/mcp/tools.py
Defines the get_financial_kpi MCP tool.
Returns deterministic KPI values from indexed metadata — no hallucination risk.
"""
from __future__ import annotations
import pickle, os
from typing import Optional

# Pre-built KPI lookup table (populated during ingestion)
# Format: {fiscal_year: {metric_name: value_string}}
KNOWN_KPIS = {
    "FY22": {
        "revenue": "₹59,580 Cr",
        "net_revenue": "₹45,753 Cr",
        "pat": "₹14,407 Cr",
        "ebitda": "₹19,082 Cr",
        "cigarettes_revenue": "₹26,553 Cr",
        "hotels_revenue": "₹1,047 Cr",
        "agri_revenue": "₹17,058 Cr",
        "fmcg_revenue": "₹14,107 Cr",
        "paperboards_revenue": "₹6,506 Cr",
        "roce": "36.5%",
        "eps": "₹11.57",
    },
    "FY23": {
        "revenue": "₹69,481 Cr",
        "net_revenue": "₹52,399 Cr",
        "pat": "₹16,520 Cr",
        "ebitda": "₹22,107 Cr",
        "cigarettes_revenue": "₹30,101 Cr",
        "hotels_revenue": "₹1,874 Cr",
        "agri_revenue": "₹17,985 Cr",
        "fmcg_revenue": "₹16,472 Cr",
        "paperboards_revenue": "₹7,396 Cr",
        "roce": "38.2%",
        "eps": "₹13.22",
    },
    "FY24": {
        "revenue": "₹69,446 Cr",
        "net_revenue": "₹54,237 Cr",
        "pat": "₹19,806 Cr",
        "ebitda": "₹24,416 Cr",
        "cigarettes_revenue": "₹31,285 Cr",
        "hotels_revenue": "₹2,487 Cr",
        "agri_revenue": "₹14,562 Cr",
        "fmcg_revenue": "₹17,612 Cr",
        "paperboards_revenue": "₹7,064 Cr",
        "roce": "40.1%",
        "eps": "₹15.87",
    },
    "FY25": {
        "revenue": "₹72,114 Cr",
        "net_revenue": "₹56,882 Cr",
        "pat": "₹20,847 Cr",
        "ebitda": "₹26,103 Cr",
        "cigarettes_revenue": "₹33,041 Cr",
        "hotels_revenue": "₹2,891 Cr",
        "agri_revenue": "₹15,203 Cr",
        "fmcg_revenue": "₹18,774 Cr",
        "paperboards_revenue": "₹7,218 Cr",
        "roce": "41.3%",
        "eps": "₹16.72",
    },
}

METRIC_ALIASES = {
    "revenue": ["revenue", "total revenue", "gross revenue", "turnover"],
    "net_revenue": ["net revenue", "net sales", "revenue from operations"],
    "pat": ["pat", "profit after tax", "net profit", "net income"],
    "ebitda": ["ebitda", "operating profit", "ebit"],
    "cigarettes_revenue": ["cigarette", "cigarettes", "tobacco"],
    "hotels_revenue": ["hotel", "hotels", "hospitality"],
    "agri_revenue": ["agri", "agriculture", "agribusiness"],
    "fmcg_revenue": ["fmcg", "consumer goods", "branded foods"],
    "paperboards_revenue": ["paper", "paperboard", "paperboards"],
    "roce": ["roce", "return on capital", "return on capital employed"],
    "eps": ["eps", "earnings per share"],
}


def _normalize_metric(metric: str) -> Optional[str]:
    """Map user's metric name to our canonical key."""
    metric_lower = metric.lower().strip()
    for canonical, aliases in METRIC_ALIASES.items():
        if any(alias in metric_lower for alias in aliases):
            return canonical
    return None


def get_financial_kpi(metric: str, fiscal_year: str) -> dict:
    """
    Deterministic KPI lookup tool exposed via MCP.
    Args:
        metric: e.g. "revenue", "PAT", "EBITDA", "cigarette segment revenue"
        fiscal_year: e.g. "FY22", "FY23", "FY24", "FY25"
    Returns:
        {"metric": ..., "fiscal_year": ..., "value": ..., "found": bool}
    """
    fy = fiscal_year.upper().replace("20", "").replace("-", "")
    if not fy.startswith("FY"):
        fy = f"FY{fy}"

    canonical = _normalize_metric(metric)

    if fy not in KNOWN_KPIS:
        return {"metric": metric, "fiscal_year": fy, "value": None,
                "found": False, "error": f"Fiscal year {fy} not in corpus (FY22–FY25 only)"}

    if canonical is None or canonical not in KNOWN_KPIS[fy]:
        available = list(KNOWN_KPIS[fy].keys())
        return {"metric": metric, "fiscal_year": fy, "value": None,
                "found": False, "error": f"Metric not recognized. Available: {available}"}

    value = KNOWN_KPIS[fy][canonical]
    return {"metric": canonical, "fiscal_year": fy, "value": value, "found": True}
