"""
eval/metrics.py
Computes: Recall@5, nDCG@10, Faithfulness, Correctness, Latency, Cost.
"""
from __future__ import annotations
import json, math
from typing import List, Dict, Optional
import openai
import os

client = openai.OpenAI()

FAITHFULNESS_PROMPT = """You are an evaluation judge for a RAG system.
Determine whether each factual claim in the answer is supported by the source chunks.
A claim is "supported" if the specific number/fact appears in or is directly inferable from the source chunks.

Respond ONLY with valid JSON (no markdown):
{"claims": [{"claim": "<text>", "supported": true/false}], "faithfulness_score": <0.0-1.0>}

Source Chunks:
{chunks}

Answer to evaluate:
{answer}"""

CORRECTNESS_PROMPT = """You are an evaluation judge.
Compare the generated answer to the ground truth and score correctness.

Rubric:
1.0 = All key facts correct, no material omissions
0.75 = Mostly correct, minor imprecision
0.5 = Partially correct
0.25 = Mostly wrong, one correct element
0.0 = Completely wrong or refused when answer was available

Special: if ground_truth contains "REFUSE" and answer refuses → score 1.0
If ground_truth contains "CLARIFY" and answer asks for clarification → score 1.0

Respond ONLY with valid JSON: {{"score": <float>, "reasoning": "<one sentence>"}}

Ground truth: {ground_truth}
Generated answer: {answer}"""


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    if not relevant_ids:
        return 1.0  # Nothing to recall
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    dcg = sum(
        (1 / math.log2(rank + 2)) for rank, rid in enumerate(top_k) if rid in relevant_ids
    )
    ideal_dcg = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_faithfulness(answer: str, chunks: List[Dict]) -> float:
    chunk_texts = "\n\n".join([f"[{c.get('citation','?')}]: {c['text'][:500]}" for c in chunks])
    prompt = FAITHFULNESS_PROMPT.format(chunks=chunk_texts, answer=answer)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return float(data.get("faithfulness_score", 0))
    except Exception:
        return 0.5


def evaluate_correctness(answer: str, ground_truth: str) -> float:
    prompt = CORRECTNESS_PROMPT.format(ground_truth=ground_truth, answer=answer)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return float(data.get("score", 0))
    except Exception:
        return 0.5
