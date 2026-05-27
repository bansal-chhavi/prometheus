---
description: "Task list for Prometheus RAG Faithfulness Evaluator"
---

# Tasks: Prometheus RAG Faithfulness Evaluator

**Input**: Design documents from `/specs/001-prometheus-faithfulness-evaluator/`

**Prerequisites**: `plan.md` (required), `spec.md` (required for user stories)

**Organization**: Tasks are grouped by phase and user story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (independent files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1–US5)

---

## Phase 0: Project Setup (Shared Infrastructure)

**Purpose**: Repository scaffold, dependencies, and environment configuration.

- [ ] T001 Create project directory structure: `src/`, `tests/`, `scripts/`, configs
- [ ] T002 Create `requirements.txt` with FastAPI, Uvicorn, Pydantic v2, httpx, pytest, python-dotenv, openai/anthropic SDK
- [ ] T003 [P] Create `.env.example` with `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` placeholders
- [ ] T004 [P] Create `Dockerfile` with Python 3.11 base image, dependency install, and `uvicorn src.main:app` entrypoint

---

## Phase 1: Pydantic Schemas (US3, US4)

**Purpose**: Define request/response data contracts.

- [ ] T005 Create `src/models.py` with `EvaluationRequest` schema: `query: str`, `answer: str`, `documents: list[str]`
- [ ] T006 [P] Add `EvaluationResponse` schema to `src/models.py`: `score: float`, `label: Literal["FAITHFUL","HALLUCINATION"]`, `hallucinated_spans: list[str]`, `evidence_spans: list[str]`, `explanation: str`

---

## Phase 2: Preprocessor (US1, US2)

**Purpose**: Light chunking, normalization, and span extraction before LLM calls.

- [ ] T007 Create `src/preprocessor.py` — implement `normalize_text(text: str) -> str`: lowercase, strip extra whitespace, remove duplicates across document list
- [ ] T008 [P] Add `chunk_document(text: str, max_chars: int = 1500) -> list[str]` to `src/preprocessor.py` — split large documents into overlapping chunks
- [ ] T009 [P] Add `extract_candidate_spans(text: str) -> list[str]` to `src/preprocessor.py` — identify numeric values, percentages, and named quantities (regex-based)

---

## Phase 3: Core Evaluation Engine (US1, US2)

**Purpose**: LLM-backed claim extraction, evidence matching, hallucination detection, and scoring.

- [ ] T010 Create `src/evaluator.py` — implement `extract_claims(answer: str) -> list[str]` using LLM prompt to decompose answer into atomic factual claims
- [ ] T011 Add `match_claim_to_evidence(claim: str, chunks: list[str]) -> tuple[bool, str]` to `src/evaluator.py` — LLM prompt checks if a claim is supported by any chunk; returns `(is_supported, evidence_excerpt)`
- [ ] T012 Add `detect_hallucinations(answer: str, claims: list[str], support_map: dict) -> list[str]` to `src/evaluator.py` — return list of answer spans corresponding to unsupported claims
- [ ] T013 Create `src/scorer.py` — implement `compute_faithfulness_score(claims: list[str], support_map: dict[str, bool]) -> float` returning `supported / total` ratio (0.0–1.0)
- [ ] T014 Add `compute_label(score: float, threshold: float = 0.8) -> str` to `src/scorer.py` — returns `"FAITHFUL"` if `score >= threshold`, else `"HALLUCINATION"`

---

## Phase 4: Explanation Generator (US3)

**Purpose**: Produce a human-readable explanation of the evaluation result.

- [ ] T015 Create `src/explainer.py` — implement `generate_explanation(query: str, answer: str, claims: list[str], support_map: dict, hallucinated_spans: list[str]) -> str` using LLM to produce a 2–4 sentence explanation identifying correct and incorrect claims

---

## Phase 5: FastAPI App & Endpoint (US4)

**Purpose**: Wire all components into a single stateless REST endpoint.

- [ ] T016 Create `src/main.py` — initialize FastAPI app, import router
- [ ] T017 Implement `POST /evaluate` route in `src/main.py` that: validates input via `EvaluationRequest`, calls preprocessor → evaluator → scorer → explainer, returns `EvaluationResponse`
- [ ] T018 [P] Add global exception handler for `422 Unprocessable Entity` with descriptive error messages for missing/invalid fields
- [ ] T019 [P] Add `/health` endpoint returning `{"status": "ok"}` for liveness checks

---

## Phase 6: Unit Tests (US1–US4)

**Purpose**: Test individual components in isolation.

- [ ] T020 Create `tests/conftest.py` — shared fixtures: FastAPI test client (`httpx.AsyncClient`), sample faithful payload from `prometheus_eval_01.json`, sample hallucination payload from `prometheus_eval_05.json`
- [ ] T021 Create `tests/test_evaluator.py` [US1, US2] — test `extract_claims`, `match_claim_to_evidence`, `detect_hallucinations` with fixtures
- [ ] T022 [P] Create `tests/test_scorer.py` [US1, US2] — test `compute_faithfulness_score` and `compute_label` with known claim/support maps
- [ ] T023 [P] Create `tests/test_preprocessor.py` [US1] — test `normalize_text`, `chunk_document`, `extract_candidate_spans`

---

## Phase 7: API Integration Tests (US1–US4)

**Purpose**: End-to-end tests against the running FastAPI app using the labeled evaluation cases.

- [ ] T024 Create `tests/test_api.py` [US4] — test `POST /evaluate` returns 200 with correct schema fields for a valid payload
- [ ] T025 [P] Add test [US4] — verify `POST /evaluate` returns 422 for a payload missing `answer` field
- [ ] T026 Add test [US1] — submit `prometheus_eval_01.json` payload, assert `label == "FAITHFUL"` and `hallucinated_spans == []`
- [ ] T027 [P] Add test [US2] — submit `prometheus_eval_05.json`, assert `label == "HALLUCINATION"` and `"31%"` in `hallucinated_spans`
- [ ] T028 [P] Add test [US2] — submit `prometheus_eval_06.json`, assert `"five hours"` in `hallucinated_spans`
- [ ] T029 [P] Add test [US2] — submit `prometheus_eval_07.json`, assert `"35%"` in `hallucinated_spans`
- [ ] T030 [P] Add test [US2] — submit `prometheus_eval_08.json`, assert `"510,000 students"` in `hallucinated_spans`
- [ ] T031 [P] Add test [US2] — submit `prometheus_eval_09.json`, assert `"680,000"` in `hallucinated_spans`
- [ ] T032 [P] Add test [US3] — verify all 5 response fields (`score`, `label`, `hallucinated_spans`, `evidence_spans`, `explanation`) present in every response

---

## Phase 8: Batch Evaluation Script & Benchmark (US5)

**Purpose**: Run all 10 labeled input files and report aggregate accuracy.

- [ ] T033 Create `scripts/batch_evaluate.py` [US5] — load all 10 `Inputs/prometheus_eval_*.json` files, call `POST /evaluate` for each, compare predicted label and spans to ground truth
- [ ] T034 [P] Add summary table output to `scripts/batch_evaluate.py` [US5] — print per-file results: `file`, `predicted_label`, `ground_truth_label`, `correct`, `detected_spans`
- [ ] T035 Create `tests/test_batch.py` [US5] — parametrized pytest test over all 10 input files asserting correct label prediction for each
- [ ] T036 [P] Add assertion to `tests/test_batch.py` [US5] — for each HALLUCINATION file, assert that all spans in `ground_truth hallucinated_spans` appear in the predicted `hallucinated_spans`

---

## Phase 9: Documentation & Packaging

- [ ] T037 [P] Update `README.md` with: project overview, local setup instructions, example `curl` command for `POST /evaluate`, batch evaluation usage
- [ ] T038 [P] Verify `Dockerfile` builds and `uvicorn` starts without errors
