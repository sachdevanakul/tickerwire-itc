# TickerWire – ITC Annual Report Research Assistant

An AI-powered RAG (Retrieval-Augmented Generation) system built for querying ITC Limited Annual Reports (FY22–FY25).

The system supports:
- Hybrid Retrieval (Dense + BM25)
- HyDE Query Expansion
- Reranking
- Query Routing
- Citation-Based Answers
- FastAPI REST API
- Groq LLM Integration

---

# Features

## 1. Intelligent Query Routing

Queries are classified into four categories:

- `direct_answer`
- `retrieve_then_answer`
- `clarify`
- `refuse`

### Examples

| Query | Action |
|---|---|
| What was ITC’s EBITDA in FY24? | direct_answer |
| Compare cigarette revenue from FY22 to FY25 | retrieve_then_answer |
| What was the revenue growth? | clarify |
| Predict ITC stock price for 2027 | refuse |

---

# 2. Hybrid Retrieval Pipeline

The system combines:

## Dense Retrieval
- SentenceTransformer embeddings
- ChromaDB vector search

## Sparse Retrieval
- BM25 keyword retrieval

## Fusion Strategy
- Reciprocal Rank Fusion (RRF)

This improves:
- semantic understanding
- exact keyword matching
- financial terminology retrieval

---

# 3. HyDE (Hypothetical Document Embeddings)

Before dense retrieval:
1. LLM generates a hypothetical answer passage
2. Passage is embedded
3. Retrieval uses enriched semantic representation

Benefits:
- Better retrieval from long annual reports
- Improved semantic matching
- Higher recall

---

# 4. Reranking

Retrieved chunks are reranked using Cohere Rerank API.

Benefits:
- Better relevance
- Cleaner citations
- Improved final answer quality

---

# 5. Grounded Generation

The LLM answers ONLY using retrieved context.

Rules enforced:
- No hallucinations
- Mandatory citations
- No speculation
- Missing information explicitly stated

Example citation:

```text
[AR-FY24, p.79]
```

---

# Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| LLM Provider | Groq |
| Model | llama-3.1-8b-instant |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25 |
| Reranking | Cohere |
| Tracing | OpenTelemetry |
| Language | Python 3.11 |

---

# Project Structure

```text
src/
│
├── agent/
│   ├── router.py
│   ├── generator.py
│   └── cost_guard.py
│
├── retrieval/
│   ├── dense.py
│   ├── bm25.py
│   ├── hybrid.py
│   ├── hyde.py
│   └── reranker.py
│
├── ingestion/
│   ├── chunker.py
│   ├── parser.py
│   └── indexer.py
│
├── api/
│   ├── main.py
│   └── models.py
│
└── utils/
    ├── logging.py
    └── tracing.py
```

---

# API Endpoints

## Root Endpoint

```http
GET /
```

### Response

```json
{
  "message": "TickerWire ITC Assistant API running"
}
```

---

## Health Check

```http
GET /health
```

### Response

```json
{
  "status": "ok",
  "chroma_chunks": 1000,
  "bm25_loaded": true
}
```

---

## Query Endpoint

```http
POST /query
```

### Request

```json
{
  "query": "What was ITC's cigarette segment revenue in FY2024?"
}
```

### Response

```json
{
  "query_id": "80a27049",
  "action": "direct_answer",
  "answer": "ITC cigarette segment revenue was ...",
  "citations": [
    {
      "citation": "AR-FY24, p.79"
    }
  ],
  "latency_ms": 10391.3
}
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <repo-url>
cd tickerwire-itc
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

COHERE_API_KEY=your_cohere_api_key

OPENAI_API_KEY=dummy_not_used

HYDE_ENABLED=true

DENSE_TOP_K=20
RERANK_TOP_N=5
```

---

# Run Application

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# Retrieval Pipeline

```text
User Query
    ↓
Query Router
    ↓
HyDE Query Expansion
    ↓
Dense Retrieval + BM25
    ↓
Reciprocal Rank Fusion
    ↓
Cohere Reranking
    ↓
LLM Generation
    ↓
Grounded Answer + Citations
```

---

# Example Queries

## Direct Answer

```text
What was ITC's EBITDA in FY2024?
```

---

## Comparative Query

```text
Compare ITC cigarette revenue from FY22 to FY25
```

---

## Clarification Query

```text
What was the revenue growth?
```

---

## Refusal Query

```text
Predict ITC stock price for 2027
```

---

# Design Decisions

## Why Hybrid Retrieval?

Dense retrieval captures semantic meaning while BM25 captures exact keywords.

Combining both improves:
- recall
- precision
- financial terminology matching

---

## Why HyDE?

Annual reports contain:
- long prose
- indirect financial language
- large contextual passages

HyDE improves semantic retrieval quality by generating a hypothetical answer before embedding.

---

## Why Groq?

Groq provides:
- extremely fast inference
- low latency
- lower API cost
- OpenAI-compatible SDK support

---

# Current Limitations

- Retrieval quality depends on chunking quality
- Financial tables may require better parsing
- No conversational memory
- No OCR support for scanned PDFs

---

# Future Improvements

- Better table extraction
- Multi-turn conversational memory
- Dashboard frontend
- Multi-company support
- Query caching
- Citation highlighting UI
- Evaluation metrics pipeline

---

# Assignment Objectives Achieved

- RAG Architecture
- Hybrid Retrieval
- HyDE Query Expansion
- FastAPI Backend
- Query Routing
- Citation-Based Answers
- Groq Integration
- Cohere Reranking
- Production-Style Modular Architecture

---

# Author

Nakul  
B.Tech CSE (AI & ML)