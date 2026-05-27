# Implementation Plan: Prometheus RAG Faithfulness Evaluator

**Branch**: `001-prometheus-faithfulness-evaluator` | **Date**: 2026-05-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-prometheus-faithfulness-evaluator/spec.md`

---

## Summary

Build a stateless REST API that accepts a query, a RAG-generated answer, and source document chunks, then returns a faithfulness score (0–1), hallucinated spans, evidence spans, and a natural-language explanation. Validated against 10 labeled evaluation cases (6 FAITHFUL, 4 HALLUCINATION) in `Inputs/`.

---

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: FastAPI, Uvicorn, Pydantic v2, OpenAI / Anthropic SDK (LLM call for claim extraction + comparison), pytest

**Storage**: None — fully stateless, no database

**Testing**: pytest + httpx (async test client)

**Target Platform**: Linux server / containerized (Docker)

**Project Type**: web-service (REST API)

**Performance Goals**: p95 latency < 5 seconds per evaluation request (LLM-bound)

**Constraints**: No persistent state; JSON in / JSON out; single `/evaluate` endpoint; deterministic on same input+model

**Scale/Scope**: Single-service deployment; 10-case offline benchmark dataset included

---

## Architecture

```
POST /evaluate
    │
    ▼
┌─────────────────────────┐
│  Input Validation        │  Pydantic schema check
│  (query, answer, docs)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Preprocessor            │  Light chunking, span extraction,
│                          │  normalization (dedup, clean text)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Evaluation Engine       │  LLM-based claim vs evidence matching
│  - Claim Extractor       │  Extract atomic claims from answer
│  - Evidence Matcher      │  Match each claim to source chunks
│  - Hallucination Detector│  Flag unsupported claims
│  - Score Calculator      │  faithfulness = supported / total claims
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Explanation Generator   │  Produce natural language reasoning
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Response Serializer     │  Return structured JSON
│  score, label,           │
│  hallucinated_spans,     │
│  evidence_spans,         │
│  explanation             │
└─────────────────────────┘
```

---

## Project Structure

```
prometheus/
├── Inputs/                          # 10 labeled evaluation cases (ground truth)
│   ├── prometheus_eval_01.json      # FAITHFUL — Banking/Fraud Detection
│   ├── prometheus_eval_02.json      # FAITHFUL — Healthcare/Scheduling
│   ├── prometheus_eval_03.json      # FAITHFUL — Manufacturing/Maintenance
│   ├── prometheus_eval_04.json      # FAITHFUL — Telecom/Retention
│   ├── prometheus_eval_05.json      # HALLUCINATION — Energy (31% vs 24%)
│   ├── prometheus_eval_06.json      # HALLUCINATION — Pharmacy (5h vs 3h)
│   ├── prometheus_eval_07.json      # HALLUCINATION — Rail (35% vs 29%)
│   ├── prometheus_eval_08.json      # HALLUCINATION — Education (510k vs 410k)
│   ├── prometheus_eval_09.json      # HALLUCINATION — Logistics (680k vs 780k)
│   └── prometheus_eval_10.json      # FAITHFUL — Supply Chain
│
├── specs/
│   └── 001-prometheus-faithfulness-evaluator/
│       ├── spec.md                  # Feature specification (this project)
│       ├── plan.md                  # This file
│       └── tasks.md                 # Implementation task list
│
├── src/
│   ├── main.py                      # FastAPI app entrypoint
│   ├── models.py                    # Pydantic request/response schemas
│   ├── preprocessor.py              # Chunking, normalization, span extraction
│   ├── evaluator.py                 # Core evaluation engine (LLM calls)
│   ├── scorer.py                    # Faithfulness score computation
│   └── explainer.py                 # Explanation generation
│
├── tests/
│   ├── test_api.py                  # FastAPI endpoint tests (US1–US4)
│   ├── test_evaluator.py            # Unit tests for evaluator logic
│   ├── test_batch.py                # Batch benchmark against all 10 inputs
│   └── conftest.py                  # Shared fixtures and test client
│
├── scripts/
│   └── batch_evaluate.py            # CLI: run all Inputs/ files and print summary
│
├── .env.example                     # Environment variable template (API keys)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container image definition
└── README.md                        # Project overview and usage
```

---

## Evaluation Ground Truth Mapping

Used for benchmark testing (`test_batch.py` and `scripts/batch_evaluate.py`):

| File | Domain | Label | Hallucinated Spans |
|------|--------|-------|--------------------|
| `prometheus_eval_01.json` | Banking | FAITHFUL | `[]` |
| `prometheus_eval_02.json` | Healthcare | FAITHFUL | `[]` |
| `prometheus_eval_03.json` | Manufacturing | FAITHFUL | `[]` |
| `prometheus_eval_04.json` | Telecom | FAITHFUL | `[]` |
| `prometheus_eval_05.json` | Energy | HALLUCINATION | `["31%"]` |
| `prometheus_eval_06.json` | Pharmacy | HALLUCINATION | `["five hours"]` |
| `prometheus_eval_07.json` | Rail Transit | HALLUCINATION | `["35%"]` |
| `prometheus_eval_08.json` | Education | HALLUCINATION | `["510,000 students"]` |
| `prometheus_eval_09.json` | Logistics | HALLUCINATION | `["680,000"]` |
| `prometheus_eval_10.json` | Supply Chain | FAITHFUL | `[]` |

---

## API Contract

### `POST /evaluate`

**Request body**:
```json
{
  "query": "string",
  "answer": "string",
  "documents": ["string", "..."]
}
```

**Response (200 OK)**:
```json
{
  "score": 0.95,
  "label": "FAITHFUL",
  "hallucinated_spans": [],
  "evidence_spans": ["...supported claim excerpts..."],
  "explanation": "The answer correctly reflects all figures stated in the source documents."
}
```

**Error (422 Unprocessable Entity)**: Missing or invalid fields.

---

## Implementation Phases

### Phase 0 — Project Setup
- Dependencies, env config, FastAPI skeleton, Pydantic schemas

### Phase 1 — Core Engine
- Preprocessor (chunking, normalization)
- Claim extractor (LLM prompt: extract atomic claims from answer)
- Evidence matcher (LLM prompt: match each claim to source)
- Hallucination detector (identify unsupported claims)
- Score calculator (faithfulness = supported_claims / total_claims)

### Phase 2 — API & Serialization
- `POST /evaluate` endpoint wired to evaluation engine
- Response serializer with all 5 output fields
- Input validation and error handling

### Phase 3 — Testing & Benchmark
- Unit tests for evaluator components
- API integration tests (all 5 user stories)
- Batch evaluation script against all 10 `Inputs/` files
- Target: 100% accuracy on benchmark set (10/10 correct labels)

### Phase 4 — Packaging
- Dockerfile, requirements.txt, .env.example, README
