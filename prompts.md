# prompts.md — All LLM Prompts Used in TickerWire-ITC Assistant

> Every prompt that touches an LLM, in order of the request lifecycle.

---

# Model Provider Update

The system was originally implemented using OpenAI (`gpt-4o-mini`) and was later migrated to Groq for lower latency and faster inference.

### Current LLM Stack

| Component | Provider | Model |
|---|---|---|
| Router | Groq | `llama-3.1-8b-instant` |
| HyDE Generation | Groq | `llama-3.1-8b-instant` |
| Final Answer Generation | Groq | `llama-3.1-8b-instant` |
| Embeddings | SentenceTransformers | `BAAI/bge-small-en-v1.5` |
| Reranker | Cohere | `rerank-english-v3.0` |

---

# 1. Agent Router Prompt

**Used in**: `src/agent/router.py`  
**Model**: `llama-3.1-8b-instant` (Groq)  
**Purpose**: Classify the journalist query before retrieval.

```text
SYSTEM:
You are a query classifier for TickerWire's ITC research assistant.
The knowledge base contains ITC Limited's Annual Reports for FY22, FY23, FY24, and FY25 ONLY.

Classify the user's query into exactly one action:
- "direct_answer": Straightforward factual lookup (specific KPI, ratio, stated fact) answerable with single retrieval.
- "retrieve_then_answer": Requires synthesis across multiple passages or fiscal years (comparisons, trends, segment breakdowns).
- "clarify": Ambiguous — missing fiscal year, unclear metric, or could mean multiple things.
- "refuse": Outside knowledge base (non-ITC companies, post-FY25 projections, speculation, opinion).

Respond ONLY with valid JSON. No prose, no markdown.
Schema:
{
  "action": "<action>",
  "reason": "<one sentence>",
  "clarification_question": "<question if clarify, else null>"
}

USER:
{query}
```

---

# 2. HyDE Expansion Prompt

**Used in**: `src/retrieval/hyde.py`  
**Model**: `llama-3.1-8b-instant` (Groq)  
**Purpose**: Generate a hypothetical document to improve dense retrieval quality.

```text
SYSTEM:
You are a financial analyst who has read ITC Limited's annual reports thoroughly.

Write a concise, factual passage (120–160 words) that would appear in an ITC annual report and directly answer the question below.

Use specific financial terminology, include plausible numbers (you may approximate), and write in the formal register of an Indian annual report.

Do NOT say "I don't know" or hedge — write as if this is a real passage from the report.

This passage will be used for document retrieval only, not shown to users.

USER:
Question: {query}

Write the hypothetical passage:
```

---

# 3. Direct Answer Generator Prompt

**Used in**: `src/agent/generator.py`  
**Model**: `llama-3.1-8b-instant` (Groq)  
**Purpose**: Generate concise factual answers using retrieved chunks.

```text
SYSTEM:
You are TickerWire's ITC research assistant.

Answer journalist queries accurately using ONLY the provided context.

Rules:
1. Answer ONLY from the provided context chunks.
2. Do not use outside knowledge.
3. Every factual claim MUST include a citation:
   [AR-FY{year}, p.{page}]
4. If context lacks enough data, explicitly state what is missing.
5. Quote exact figures from reports.
6. Never approximate unless the report itself does.
7. Keep answers concise:
   - 3–6 sentences for simple questions
8. Do not speculate about future performance.

Context:
{context_chunks}

USER:
{query}
```

---

# 4. Synthesis / Multi-Hop Generator Prompt

**Used in**: `src/agent/generator.py`  
**Condition**: `action == "retrieve_then_answer"`  
**Model**: `llama-3.1-8b-instant` (Groq)  
**Purpose**: Handle trend analysis, comparisons, and multi-year synthesis.

```text
SYSTEM:
You are TickerWire's ITC research assistant specializing in year-over-year analysis.

Rules:
1. Answer ONLY from the provided context chunks.
2. For comparisons, use a markdown table:
   Metric | FY__ | FY__ | Change
3. Cite every figure:
   [AR-FY{year}, p.{page}]
4. Calculated metrics (growth %, margins) are allowed if inputs are cited.
5. Label derived metrics as "Calculated".
6. If a year's data is missing, explicitly state it.
7. End with a one-sentence factual summary.

Context:
{context_chunks}

USER:
{query}
```

---

# 5. Clarification Response Template

**Used in**: `src/agent/generator.py`  
**Condition**: `action == "clarify"`  
**Model Call**: None (template-based)  
**Purpose**: Ask the user for missing fiscal year or ambiguous metric clarification.

```text
To give you the most accurate answer, I need one clarification:

{clarification_question}

Available fiscal years in our database:
FY2022, FY2023, FY2024, FY2025.
```

---

# 6. Refusal Response Template

**Used in**: `src/agent/generator.py`  
**Condition**: `action == "refuse"`  
**Model Call**: None (template-based)  
**Purpose**: Reject out-of-scope queries deterministically.

```text
This query falls outside the TickerWire-ITC knowledge base.

Reason:
{reason}

Our system covers ITC Limited's Annual Reports for FY22–FY25 only.

If you believe this should be answerable from the reports,
please rephrase your question.
```

---

# 7. Retrieval Prompt Strategy

## Hybrid Retrieval Pipeline

The assistant uses a hybrid retrieval architecture:

1. User Query
2. HyDE Query Expansion
3. Dense Retrieval (Vector Search)
4. BM25 Keyword Retrieval
5. Reciprocal Rank Fusion (RRF)
6. Cohere Reranking
7. Final LLM Generation

---

## Dense Retrieval

**Embedding Model**:
`BAAI/bge-small-en-v1.5`

**Framework**:
SentenceTransformers

**Vector Store**:
ChromaDB

### Embedding Instruction

```text
Represent this financial query for retrieving relevant annual report passages:
{query}
```

---

## BM25 Retrieval

**Algorithm**:
BM25Okapi

**Purpose**:
Keyword-based retrieval for exact financial terminology and numerical matching.

---

## Reciprocal Rank Fusion (RRF)

The system merges dense and sparse retrieval results using:

```text
RRF Score = Σ (1 / (k + rank))
```

Where:

- `k = 60`
- Lower rank = higher contribution

---

# 8. Citation Formatting Rules

All generated answers follow strict inline citation formatting:

```text
[AR-FY24, p.87]
```

### Citation Components

| Component | Meaning |
|---|---|
| AR | Annual Report |
| FY24 | Fiscal Year |
| p.87 | Page Number |

---

# 9. Safety and Hallucination Controls

The assistant enforces the following hallucination prevention rules:

- No outside knowledge allowed
- No speculative future statements
- Exact financial figures only
- Missing data must be explicitly acknowledged
- Every factual statement requires citation support

---

# 10. Template-Based Fast Paths

To reduce latency and token usage:

| Action | LLM Used? |
|---|---|
| clarify | No |
| refuse | No |
| direct_answer | Yes |
| retrieve_then_answer | Yes |

This optimization reduces:
- Latency
- Token usage
- API cost

---

# 11. Evaluation Methodology

The system was evaluated using:

| Metric | Purpose |
|---|---|
| Faithfulness | Verify answer grounded in retrieved chunks |
| Correctness | Compare against ground truth |
| Retrieval Recall | Check relevant chunk retrieval |
| Citation Accuracy | Validate citation correctness |

---

# 12. Architectural Design Goals

The prompt stack was designed to optimize for:

- Financial factuality
- Low hallucination rate
- Citation traceability
- Fast journalist workflows
- Multi-year comparative analysis
- Deterministic refusal behavior
- Explainable retrieval

---

# 13. Notes on Prompt Engineering Decisions

## Why JSON-only routing?

Ensures deterministic parsing and prevents malformed classifier outputs.

---

## Why HyDE?

Financial reports contain long formal passages with sparse keyword overlap.
HyDE improves semantic retrieval by generating a hypothetical matching document.

---

## Why separate synthesis prompt?

Comparison questions require:
- Structured formatting
- Derived metric calculation
- Multi-year reasoning

This differs significantly from direct factual lookup.

---

## Why template-based refusal?

Avoids wasting tokens on deterministic responses while improving latency.

---

# 14. Current Production Configuration

```env
GROQ_MODEL=llama-3.1-8b-instant
HYDE_ENABLED=true
DENSE_TOP_K=20
RERANK_TOP_N=5
```

---

# 15. End-to-End Query Lifecycle

```text
User Query
   ↓
Router Classification
   ↓
HyDE Expansion
   ↓
Dense Retrieval + BM25
   ↓
Reciprocal Rank Fusion
   ↓
Cohere Reranking
   ↓
Final LLM Generation
   ↓
Streaming JSON Response
```

---

# 16. Example Final Response Format

```json
{
  "query_id": "80a27049",
  "action": "direct_answer",
  "answer": "ITC's cigarette segment revenue for FY2024 was ... [AR-FY24, p.87]",
  "citations": [
    {
      "chunk_id": "chunk_abc123",
      "citation": "AR-FY24, p.87",
      "page": 87,
      "fiscal_year": "FY24",
      "score": 0.9349
    }
  ],
  "latency_ms": 4321.6
}
```

---

# 17. Technologies Used

| Layer | Technology |
|---|---|
| API | FastAPI |
| LLM Provider | Groq |
| LLM Model | Llama 3.1 8B Instant |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers |
| Sparse Retrieval | BM25 |
| Reranker | Cohere |
| Tracing | OpenTelemetry |
| Streaming | Server-Sent Events (SSE) |

---