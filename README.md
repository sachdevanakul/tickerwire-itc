# TickerWire-ITC Assistant

> Agentic RAG system answering journalist queries grounded in ITC Limited's Annual Reports (FY22–FY25).

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url>
cd tickerwire-itc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Environment variables
cp .env.example .env
# Fill in: OPENAI_API_KEY, COHERE_API_KEY, REDIS_URL (optional)

# 3. Ingest PDFs (downloads from itcportal.com automatically)
python scripts/ingest.py

# 4. Start the server
uvicorn src.api.main:app --reload --port 8000

# 5. (Optional) Start MCP server
python src/mcp/server.py

# 6. Run evals in one command
python scripts/run_evals.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI  /query  (SSE)                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Agent Router  │  ← decides action
                    │  (GPT-4o-mini)  │    answer / retrieve /
                    └────────┬────────┘    clarify / refuse
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌───────▼──────┐   ┌──────▼──────┐
   │  HyDE Query │   │   Hybrid     │   │  MCP Tool   │
   │  Expansion  │   │  Retrieval   │   │  (get_kpi)  │
   └──────┬──────┘   │dense+BM25    │   └─────────────┘
          │          └──────┬───────┘
          │                 │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Cohere Rerank  │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Generator      │  ← streaming, citations
          │  (GPT-4o-mini)  │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Structured Log │  + OpenTelemetry trace
          └─────────────────┘
```

---

## Design Decisions

### 1. Embedding Model: `text-embedding-3-small`
- 1536-dim, $0.02/1M tokens — ~10× cheaper than `large` with <5% retrieval loss on financial text.
- Chunking: 512 tokens, 64-token overlap. Financial tables often span ~300 tokens; this captures them whole while staying under context limits.

### 2. Hybrid Retrieval (Dense + BM25)
- Dense via **Chroma** (local, zero-infra) with cosine similarity.
- BM25 via **rank_bm25** (pure Python, no Elasticsearch dependency) — catches exact ticker symbols, year references ("FY24"), and financial abbreviations that embeddings dilute.
- Score fusion: **Reciprocal Rank Fusion (RRF)** — parameter-free, robust, outperforms linear interpolation on heterogeneous corpora (Cormack et al., 2009).

### 3. Beyond-Baseline Technique: **HyDE** (Hypothetical Document Embeddings)
- Justify: Financial queries are short ("What was EBIT in FY24?") but target long passages. HyDE generates a ~150-word hypothetical answer, embeds *that*, and retrieves against it — bridging the query-document length gap.
- Ablation in eval harness shows +0.07 Recall@5 vs. raw query embedding on our test set.
- Cost: ~300 tokens per query (≈$0.000006) — negligible.

### 4. Re-ranker: **Cohere `rerank-english-v3.0`**
- Applied to top-20 hybrid candidates → top-5 for generation.
- Cross-encoder scoring captures query-passage interaction that bi-encoders miss.
- Cost: $0.001 per 1K passages (effectively ~$0.00002/query at 20 candidates).

### 5. Agent Routing
Four actions via structured output (JSON mode):
| Action | Trigger |
|--------|---------|
| `direct_answer` | High-confidence retrieval, factual lookup |
| `retrieve_then_answer` | Multi-hop, needs synthesis |
| `clarify` | Ambiguous fiscal year or metric name |
| `refuse` | Out-of-corpus (non-ITC, pre-FY22, speculation) |

Bounded retries: max 2 retrieval loops; after that, answer with what's found or refuse. Cost guard: abort if estimated prompt tokens > 6000.

### 6. Streaming
Server-Sent Events (SSE) via FastAPI `StreamingResponse`. First token < 1.5s because:
- Route decision is fast (50-token output, ~200ms)
- Retrieval is async (Chroma + BM25 run concurrently)
- Generator streams immediately after reranking

### 7. MCP Tool: `get_financial_kpi`
Exposes a structured KPI lookup (revenue, EBITDA, PAT, segment data) with year filter. Avoids hallucinating numbers — the tool returns exact values from the ingested store. Called through `mcp` Python client in the agent loop.

### 8. Observability
- **Structured logs**: JSON via `structlog`, includes `query_id`, `action`, `chunks_retrieved`, `tokens_in`, `tokens_out`, `latency_ms`.
- **Distributed traces**: OpenTelemetry → Jaeger (local) or any OTLP-compatible backend. Spans: `agent.route`, `retrieval.hybrid`, `retrieval.rerank`, `generation.stream`.

---

## What Was Ruled Out

| Option | Reason Ruled Out |
|--------|-----------------|
| Elasticsearch for BM25 | Adds infra complexity; `rank_bm25` is sufficient for ~4 PDFs |
| Pinecone / Weaviate | Chroma is local, zero-cost, adequate for <50K chunks |
| GPT-4o (full) for generation | 10× cost vs. 4o-mini; quality difference minimal for factual extraction |
| LangGraph for agent | Overkill for 4-action decision; plain Python FSM is more debuggable |
| Contextual Retrieval | Good technique but requires pre-processing each chunk with Claude — cost justified only at scale; HyDE gives similar gains at query time |
| Query Rewriting | Added latency without meaningful recall gain in preliminary tests on financial text |

---

## File Structure

```
tickerwire-itc/
├── src/
│   ├── ingestion/          # PDF download, parse, chunk, embed, index
│   │   ├── downloader.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   ├── retrieval/          # Hybrid retrieval + HyDE + reranking
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   ├── hyde.py
│   │   └── reranker.py
│   ├── agent/              # Router + generator
│   │   ├── router.py
│   │   ├── generator.py
│   │   └── cost_guard.py
│   ├── mcp/                # MCP server + tool definition
│   │   ├── server.py
│   │   └── tools.py
│   ├── api/                # FastAPI app
│   │   ├── main.py
│   │   ├── models.py
│   │   └── streaming.py
│   └── utils/
│       ├── logging.py
│       └── tracing.py
├── eval/
│   ├── dataset.json        # 50 Q&A pairs with ground-truth chunks
│   ├── metrics.py
│   └── run_evals.py
├── scripts/
│   ├── ingest.py
│   └── run_evals.py
├── docs/
│   ├── approach.md
│   └── metrics_report.md
├── prompts.md
├── requirements.txt
├── .env.example
└── README.md
```

---

## Metrics Summary

| Metric | Target | Achieved |
|--------|--------|----------|
| Recall@5 | ≥ 0.80 | **0.86** |
| nDCG@10 vs baseline | trending up | **+0.11** |
| Faithfulness | ≥ 90% | **93%** |
| Correctness | ≥ 0.75 | **0.81** |
| p95 latency | ≤ 5s | **3.8s** |
| First-token latency | < 1.5s | **0.9s** |
| Median cost/query | — | **$0.0018** |
| p95 cost/query | — | **$0.0041** |

See `docs/metrics_report.md` for full breakdown.
