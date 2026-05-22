# Approach Document — TickerWire-ITC Assistant

---

# 1. Problem Decomposition

## The Core Pain Point

A 6-analyst research desk spending **25–40 minutes per lookup query** is not fundamentally a knowledge problem — it is a:

- Retrieval problem
- Verification problem
- Latency problem

Financial journalists require answers that are:

1. Fast
2. Verifiable
3. Precisely cited
4. Non-hallucinatory

A wrong fiscal year or incorrect KPI can result in:
- Incorrect reporting
- Editorial corrections
- Loss of trust

Therefore, the system must:

- Retrieve the correct evidence
- Cite it precisely
- Refuse unsupported claims
- Ask clarification questions when ambiguity exists

---

# 2. System Architecture Overview

The assistant follows a modular Retrieval-Augmented Generation (RAG) architecture:

```text
User Query
   ↓
Router Classification
   ↓
HyDE Query Expansion
   ↓
Dense Retrieval + BM25
   ↓
Reciprocal Rank Fusion (RRF)
   ↓
Cohere Re-ranking
   ↓
LLM Answer Generation
   ↓
Streaming JSON Response
```

---

# 3. Data Layer Design

## Challenge

ITC annual reports are complex financial PDFs containing:

- Narrative prose
- Multi-column tables
- KPI summaries
- Auditor statements
- Segment financials
- Notes to accounts

The hardest challenge is reliable table extraction.

Traditional PDF parsers frequently:
- Merge columns incorrectly
- Break rows
- Lose numeric alignment

---

## Extraction Strategy

### Libraries Used

| Purpose | Library |
|---|---|
| Structured table extraction | `pdfplumber` |
| Text extraction + layout | `PyMuPDF` |

### Why Two Libraries?

`pdfplumber`
- Preserves table structure
- Better for financial statements

`PyMuPDF`
- Better narrative text flow
- Faster text extraction

The outputs are merged during ingestion.

---

# 4. Chunking Strategy

## Why Chunking Matters

Naive fixed-size chunking creates severe retrieval issues:

- Tables split across chunks
- Headers detached from values
- Context fragmentation

---

## Final Chunking Strategy

### Prose Sections

| Parameter | Value |
|---|---|
| Chunk Size | 512 tokens |
| Overlap | 64 tokens |

Sliding-window chunking preserves continuity.

---

### Tables

Tables are treated as:
- Atomic chunks
- Never split across rows/pages

This preserves:
- Financial integrity
- Row-column relationships
- KPI correctness

---

# 5. Retrieval Layer

The retrieval system must handle two major failure modes.

---

## Failure Mode 1 — Semantic Miss

Example:

```text
Query:
"What was ITC's return on capital in FY24?"
```

Relevant document may contain:

```text
ROCE improved to 32.1%
```

Keyword retrieval alone fails.

### Solution

Dense semantic retrieval using embeddings.

---

## Failure Mode 2 — Exact Match Miss

Example:

```text
Query:
"FY2024 cigarette revenue"
```

Document may contain:

```text
2023-24 Cigarettes Segment Revenue
```

Semantic retrieval alone may miss exact numeric alignment.

### Solution

BM25 sparse retrieval.

---

# 6. Hybrid Retrieval Strategy

The system combines:

| Retrieval Type | Strength |
|---|---|
| Dense Retrieval | Semantic similarity |
| BM25 | Exact keyword matching |

These are merged using:

# Reciprocal Rank Fusion (RRF)

```text
RRF Score = Σ (1 / (k + rank))
```

Where:
- `k = 60`

This approach:
- Prevents dominance by one retriever
- Improves recall stability
- Works well on heterogeneous financial data

---

# 7. HyDE (Hypothetical Document Embeddings)

## Problem

Journalist queries are extremely short:

```text
"What was ITC's cigarette EBIT margin in FY24?"
```

But annual report passages are long and verbose.

This creates an embedding asymmetry problem.

---

## Solution — HyDE

The system first generates a hypothetical answer paragraph.

Example:

```text
"The cigarette segment reported EBIT margins of..."
```

The generated paragraph is then embedded and used for retrieval.

---

## Why HyDE Works

The hypothetical passage:
- Uses financial vocabulary
- Expands abbreviations
- Adds semantic structure
- Matches report writing style

This dramatically improves:
- Recall@5
- Passage relevance
- Retrieval stability

---

# 8. Agentic Routing Layer

A naive RAG system retrieves for every query.

That creates failure cases.

---

## Examples

### Clarification Case

```text
User:
"What was ITC's revenue?"
```

Problem:
- Four fiscal years exist

Correct behavior:
- Ask for year clarification

---

### Refusal Case

```text
User:
"What will ITC revenue be in FY26?"
```

Problem:
- Outside corpus
- Requires speculation

Correct behavior:
- Refuse deterministically

---

### Synthesis Case

```text
"Compare EBIT margins FY22 vs FY25"
```

Problem:
- Multi-year reasoning required

Correct behavior:
- Retrieve multiple chunks
- Generate comparative analysis

---

# 9. Router Design

The router performs lightweight classification before retrieval.

## Actions

| Action | Purpose |
|---|---|
| direct_answer | Simple factual lookup |
| retrieve_then_answer | Multi-hop synthesis |
| clarify | Ambiguous query |
| refuse | Out-of-corpus query |

---

## Why This Matters

Benefits:
- Lower retrieval cost
- Faster latency
- More predictable behavior
- Better auditability

---

# 10. MCP Tool Design

## Tool

```text
get_financial_kpi
```

---

## Purpose

Provides deterministic structured KPI retrieval.

Instead of:
- Free-text searching for exact numbers

The system:
- Queries indexed metadata directly

This improves:
- Numerical correctness
- KPI precision
- Deterministic outputs

---

# 11. Latency Optimization

Target Requirements:

| Metric | Target |
|---|---|
| p95 latency | ≤ 5 seconds |
| First token latency | < 1.5 seconds |

---

## Parallelization Strategy

The system parallelizes:

### Concurrent Tasks

- HyDE generation
- BM25 retrieval
- Dense retrieval

using:

```python
asyncio.gather()
```

---

## Sequential Steps

Only these remain sequential:

1. Reranking
2. Final generation

---

## Streaming

The answer generator streams tokens immediately via:

```text
Server-Sent Events (SSE)
```

This improves perceived responsiveness significantly.

---

# 12. Evaluation Framework

## Ground Truth Dataset

A manually curated evaluation set of:

```text
50 Q&A pairs
```

was created by reading the reports directly.

---

## Coverage Areas

The dataset includes:

- KPI lookups
- Segment financials
- YoY comparisons
- Narrative facts
- Out-of-corpus traps
- Ratio calculations

---

# 13. Evaluation Metrics

## Retrieval Metrics

| Metric | Purpose |
|---|---|
| Recall@5 | Relevant chunk retrieval |
| nDCG@10 | Ranking quality |

---

## Generation Metrics

| Metric | Purpose |
|---|---|
| Faithfulness | Answer grounded in context |
| Correctness | Matches ground truth |
| Citation Accuracy | Citation correctness |

---

# 14. Technologies Used

| Layer | Technology |
|---|---|
| API | FastAPI |
| LLM Provider | Groq |
| Model | Llama 3.1 8B Instant |
| Embeddings | SentenceTransformers |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25 |
| Reranker | Cohere |
| Streaming | SSE |
| Tracing | OpenTelemetry |

---

# 15. What Was Ruled Out

---

## LangChain / LlamaIndex

### Why Rejected

They abstract too much internal behavior.

For financial QA systems:
- Auditability matters
- Deterministic behavior matters
- Debuggability matters

Custom Python pipelines provide:
- Full control
- Easier tracing
- Better observability

---

## Pinecone / Weaviate / Qdrant Cloud

### Why Rejected

Corpus size:
- Only ~4 annual reports
- ~40K chunks

Managed vector DBs introduce:
- External dependencies
- API costs
- Network latency

ChromaDB is:
- Local
- Lightweight
- Sufficient at current scale

---

## Elasticsearch

### Why Rejected

BM25 requirements are minimal.

Elasticsearch adds:
- Docker infrastructure
- Operational overhead
- Memory consumption

`rank_bm25` provides equivalent value for this scale.

---

## Contextual Retrieval

Anthropic-style contextual retrieval was evaluated.

### Why Rejected

Requires:
- Expensive ingestion-time LLM calls
- Re-ingestion whenever prompts/models change

HyDE provides similar gains:
- At query time
- With lower operational cost

---

## Query Rewriting

### Experimental Result

| Metric | Change |
|---|---|
| Recall@5 | +0.02 |
| Latency | +400ms |

Improvement was too small relative to latency cost.

---

# 16. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Table extraction errors | Validate against known KPIs |
| HyDE hallucinations | Fallback to raw query retrieval |
| Long prompt cost | Token budget guardrails |
| Year attribution errors | Fiscal year tagging during ingestion |
| Out-of-corpus hallucinations | Router refusal behavior |
| Citation mismatch | Chunk-level citation metadata |
| Streaming failures | Deterministic JSON response fallback |

---

# 17. Hallucination Prevention Strategy

The system prevents hallucinations using multiple layers:

## Retrieval Constraints

- Retrieval-only answering
- No outside knowledge

---

## Prompt Constraints

Generator explicitly instructed:
- Use ONLY provided chunks
- Refuse unsupported answers

---

## Citation Enforcement

Every factual statement requires:

```text
[AR-FY24, p.87]
```

---

## Refusal Path

Out-of-corpus questions:
- Never sent to retrieval
- Never generated speculatively

---

# 18. Cost Optimization Strategy

The architecture minimizes unnecessary LLM usage.

| Step | Optimization |
|---|---|
| Clarify | Template-based |
| Refuse | Template-based |
| Router | Small low-cost model |
| Embeddings | Local model |
| Retrieval | Local vector DB |

---

# 19. Streaming Response Design

Responses are streamed as structured JSON events.

## Event Types

### Meta Event

```json
{
  "type": "meta",
  "query_id": "abc123"
}
```

---

### Token Event

```json
{
  "type": "token",
  "content": "ITC reported..."
}
```

---

### Done Event

```json
{
  "type": "done",
  "citations": [...],
  "latency_ms": 3211.5
}
```

---

# 20. Scalability Considerations

The architecture is designed for future scaling.

Potential upgrades:

| Current | Future |
|---|---|
| ChromaDB | Qdrant / Pinecone |
| Single corpus | Multi-company corpus |
| Local embeddings | Hosted embedding API |
| BM25 | Elasticsearch/OpenSearch |
| Single-agent | Multi-agent workflows |

---

# 21. Final Design Philosophy

The system prioritizes:

1. Correctness over creativity
2. Traceability over fluency
3. Determinism over abstraction
4. Low hallucination risk over broad coverage

The architecture is intentionally conservative because financial journalism is a high-trust domain where incorrect information carries significant reputational risk.

---