"""
src/agent/router.py
Classifies incoming query into one of 4 actions before any retrieval.
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Literal, Optional
from openai import AsyncOpenAI
from src.utils.logging import get_logger

logger = get_logger(__name__)

Action = Literal["direct_answer", "retrieve_then_answer", "clarify", "refuse"]

ROUTER_SYSTEM = """You are a query classifier for TickerWire's ITC research assistant.
The knowledge base contains ITC Limited's Annual Reports for FY22, FY23, FY24, and FY25 ONLY.

Classify the user's query into exactly one action:
- "direct_answer": Straightforward factual lookup (specific KPI, ratio, stated fact) answerable with single retrieval.
- "retrieve_then_answer": Requires synthesis across multiple passages or fiscal years (comparisons, trends, segment breakdowns).
- "clarify": Ambiguous — missing fiscal year, unclear metric, or could mean multiple things.
- "refuse": Outside knowledge base (non-ITC companies, post-FY25 projections, speculation, opinion).

Respond ONLY with valid JSON. No prose, no markdown.
Schema: {"action": "<action>", "reason": "<one sentence>", "clarification_question": "<question if clarify, else null>"}"""


@dataclass
class RouteDecision:
    action: Action
    reason: str
    clarification_question: Optional[str] = None


async def route_query(query: str, client: AsyncOpenAI) -> RouteDecision:
    """Classify query into an action. Fast, ~50 token output."""
    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": query},
        ],
        max_tokens=150,
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        decision = RouteDecision(
            action=data["action"],
            reason=data.get("reason", ""),
            clarification_question=data.get("clarification_question"),
        )
    except Exception as e:
        logger.warning("router_parse_error", error=str(e), raw=raw)
        decision = RouteDecision(action="retrieve_then_answer", reason="parse fallback")

    logger.info("routed", action=decision.action, reason=decision.reason)
    return decision
