# Feature Specification: Prometheus RAG Faithfulness Evaluator

**Feature Branch**: `001-prometheus-faithfulness-evaluator`

**Created**: 2026-05-27

**Status**: Draft

**Input**: Real-time evaluation of RAG answer faithfulness against source documents with hallucination detection.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Evaluate Faithful RAG Answers (Priority: P1)

A developer submits a query, a RAG-generated answer, and a set of source document chunks. The system evaluates the answer and returns a FAITHFUL result with a score close to 1.0, empty hallucinated spans, and an explanation confirming which evidence supports the answer.

**Why this priority**: This is the core use case — verifying that correctly grounded answers are recognized and scored accurately. It forms the foundation of the evaluation pipeline.

**Independent Test**: Submit `prometheus_eval_01.json` (BCG fraud detection case) to `POST /evaluate`. Expect `ground_truth_label: FAITHFUL`, empty `hallucinated_spans`, and `score >= 0.9`.

**Acceptance Scenarios**:

1. **Given** a query "How did the fraud detection modernization project improve banking operations?" and source document describing 28% false-positive reduction and 35% response time improvement, **When** the answer states those same figures, **Then** the API returns `{"label": "FAITHFUL", "score": >= 0.9, "hallucinated_spans": [], "explanation": <non-empty string>}`.

2. **Given** a query about healthcare scheduling outcomes and an answer matching all figures from the source document, **When** the payload is submitted to `/evaluate`, **Then** the response label is `FAITHFUL` and no spans are flagged.

3. **Given** a supply chain optimization answer matching the 17% inventory reduction, service levels above 96%, and 90-minute forecast time, **When** evaluated, **Then** score >= 0.9 and explanation references supporting evidence.

---

### User Story 2 - Detect Hallucinated Numeric Spans (Priority: P1)

A developer submits an answer that contains an incorrect statistic not supported by the source document. The system identifies the specific hallucinated span (e.g., a wrong percentage or figure), returns a `HALLUCINATION` label, and highlights which part of the answer is unsupported.

**Why this priority**: Hallucination detection is the core safety function of the system. Numeric span errors are the most common failure mode in RAG answers.

**Independent Test**: Submit `prometheus_eval_05.json` (forecasting accuracy 31% vs actual 24%) to `POST /evaluate`. Expect `ground_truth_label: HALLUCINATION`, `hallucinated_spans: ["31%"]`.

**Acceptance Scenarios**:

1. **Given** a source document stating forecasting accuracy improved by 24% and an answer claiming 31%, **When** evaluated, **Then** the response includes `{"label": "HALLUCINATION", "hallucinated_spans": ["31%"], "score": <= 0.5}`.

2. **Given** a source document stating replenishment cycle time reduced to ~3 hours and an answer claiming ~5 hours, **When** evaluated, **Then** `hallucinated_spans` includes `"five hours"` and label is `HALLUCINATION`.

3. **Given** an answer claiming inspection efficiency improved by 35% while the document states 29%, **When** evaluated, **Then** `hallucinated_spans` includes `"35%"`.

4. **Given** an answer with a wrong student count (510,000 vs actual 410,000), **When** evaluated, **Then** `hallucinated_spans` includes `"510,000 students"`.

5. **Given** an answer with wrong freight volume (680,000 vs actual 780,000), **When** evaluated, **Then** `hallucinated_spans` includes `"680,000"`.

---

### User Story 3 - Retrieve Score, Spans, and Explanation in Structured JSON (Priority: P2)

A developer needs the evaluation result in a machine-readable JSON format with clearly separated fields for downstream processing (dashboards, logging, alerting).

**Why this priority**: Structured output enables consumers of the API to build automated pipelines without parsing free-form text.

**Independent Test**: Call `POST /evaluate` with any valid payload and verify response JSON conforms to the schema `{score: float, label: string, hallucinated_spans: list[string], evidence_spans: list[string], explanation: string}`.

**Acceptance Scenarios**:

1. **Given** any valid evaluation request, **When** the API responds, **Then** the response body contains exactly the fields: `score`, `label`, `hallucinated_spans`, `evidence_spans`, `explanation`.

2. **Given** a FAITHFUL answer, **When** evaluated, **Then** `hallucinated_spans` is an empty list `[]` and `evidence_spans` lists the matched portions from the source document.

3. **Given** a HALLUCINATION answer, **When** evaluated, **Then** `hallucinated_spans` is a non-empty list and `explanation` specifically identifies the incorrect claim and its correct counterpart from the source.

---

### User Story 4 - Stateless REST API Endpoint (Priority: P2)

A developer integrates the evaluator into a CI/CD or observability pipeline via a REST call. There is no server-side session or storage — every request is fully self-contained.

**Why this priority**: Stateless design ensures horizontal scalability and no data persistence concerns, which is a stated architectural requirement.

**Independent Test**: Send two identical requests sequentially to `POST /evaluate`. Both return identical responses. Restart the server and the third identical request also returns the same response (no session state).

**Acceptance Scenarios**:

1. **Given** a valid evaluation payload, **When** `POST /evaluate` is called, **Then** a `200 OK` with the structured JSON response is returned within 5 seconds.

2. **Given** a malformed or missing field in the payload, **When** `POST /evaluate` is called, **Then** a `422 Unprocessable Entity` error with a descriptive message is returned.

3. **Given** a document and answer in the request, **When** the same request is repeated after server restart, **Then** the evaluation result is deterministic and unchanged.

---

### User Story 5 - Batch Evaluation via Preloaded Input Files (Priority: P3)

A QA engineer runs a batch evaluation against all 10 labeled input files to measure precision and recall of the hallucination detector. The system computes per-case and aggregate accuracy metrics.

**Why this priority**: Enables offline validation and benchmarking of the model/prompt against ground truth labels.

**Independent Test**: Run batch evaluation script against `Inputs/prometheus_eval_01.json` through `prometheus_eval_10.json`. Expect 6 FAITHFUL and 4 HALLUCINATION labels matching the ground truth in each file; all hallucinated spans in HALLUCINATION cases are detected.

**Acceptance Scenarios**:

1. **Given** 10 input evaluation files (6 FAITHFUL, 4 HALLUCINATION), **When** batch evaluated, **Then** all 10 ground truth labels are predicted correctly (100% accuracy on this benchmark set).

2. **Given** the 4 HALLUCINATION files, **When** evaluated, **Then** each predicted `hallucinated_spans` list contains at least the span listed in the file's `hallucinated_spans` field.

3. **Given** a batch run produces results, **When** a summary is requested, **Then** a table is returned with columns: `file`, `predicted_label`, `ground_truth_label`, `correct`, `detected_spans`.

---

## Evaluation Dataset Summary

The following 10 cases from `Inputs/` define the ground truth for system validation:

| File | Domain | Ground Truth | Hallucinated Span |
|------|--------|--------------|-------------------|
| `prometheus_eval_01.json` | Banking / Fraud Detection | FAITHFUL | — |
| `prometheus_eval_02.json` | Healthcare / Scheduling | FAITHFUL | — |
| `prometheus_eval_03.json` | Manufacturing / Predictive Maintenance | FAITHFUL | — |
| `prometheus_eval_04.json` | Telecom / Customer Retention | FAITHFUL | — |
| `prometheus_eval_05.json` | Energy / Wind Forecasting | HALLUCINATION | `"31%"` (actual: 24%) |
| `prometheus_eval_06.json` | Retail / Pharmacy Inventory | HALLUCINATION | `"five hours"` (actual: three hours) |
| `prometheus_eval_07.json` | Transit / Rail Maintenance | HALLUCINATION | `"35%"` (actual: 29%) |
| `prometheus_eval_08.json` | Education / Academic Planning | HALLUCINATION | `"510,000 students"` (actual: 410,000) |
| `prometheus_eval_09.json` | Logistics / Shipment Tracking | HALLUCINATION | `"680,000"` (actual: 780,000) |
| `prometheus_eval_10.json` | Supply Chain / Inventory | FAITHFUL | — |
