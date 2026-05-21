"""
src/agent/cost_guard.py
Prevents runaway token spend. Aborts if estimated prompt tokens exceed limit.
"""
from __future__ import annotations
import os
import tiktoken
from src.utils.logging import get_logger

logger = get_logger(__name__)
enc = tiktoken.get_encoding("cl100k_base")
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "6000"))


def estimate_tokens(text: str) -> int:
    return len(enc.encode(text))


def check_cost_guard(prompt_parts: list[str]) -> tuple[bool, int]:
    """
    Returns (ok, token_count).
    If token_count > MAX_PROMPT_TOKENS, returns (False, count).
    """
    total = sum(estimate_tokens(p) for p in prompt_parts)
    if total > MAX_PROMPT_TOKENS:
        logger.warning("cost_guard_triggered", tokens=total, limit=MAX_PROMPT_TOKENS)
        return False, total
    return True, total


def truncate_chunks_to_limit(chunks: list[dict], query: str, system_prompt: str) -> list[dict]:
    """Truncate chunk list so total tokens stay under limit."""
    base_tokens = estimate_tokens(system_prompt) + estimate_tokens(query) + 200  # buffer
    budget = MAX_PROMPT_TOKENS - base_tokens
    selected, used = [], 0

    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk["text"])
        if used + chunk_tokens > budget:
            break
        selected.append(chunk)
        used += chunk_tokens

    logger.info("chunks_after_cost_guard", original=len(chunks), kept=len(selected), tokens_used=used)
    return selected
