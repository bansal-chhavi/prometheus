# Prometheus Constitution

## Core Principles

### I. Stateless Evaluation
Every evaluation request is fully self-contained. No session state, no database writes, no cross-request memory. The system accepts a payload and returns a result — nothing more.

### II. Structured JSON Contract (NON-NEGOTIABLE)
All API responses conform to the fixed schema: `score`, `label`, `hallucinated_spans`, `evidence_spans`, `explanation`. No field may be omitted. Consumer pipelines must be able to rely on this contract without version negotiation.

### III. Ground-Truth-First Testing
All evaluation logic is validated against the 10 labeled cases in `Inputs/`. Any change to the evaluation engine must be benchmarked against this dataset before merging. Target: 100% label accuracy on the benchmark set.

### IV. Hallucination Span Precision
Detected hallucinated spans must reference actual substrings from the answer — not paraphrases, not document text, not claim summaries. Span detection is the core deliverable of the system and must be precise.

### V. Simplicity Over Abstraction
No caching layers, no plugin systems, no multi-model routing unless explicitly specified. One endpoint. One evaluation pipeline. Complexity must be justified by a concrete requirement from the spec.

---

## Security & Data Handling

- No user data is stored between requests.
- API keys for LLM providers are loaded from environment variables only — never hardcoded, never logged.
- Input documents are never forwarded to external services beyond the configured LLM endpoint.
- All LLM prompts must be structured to prevent prompt injection from document or answer content (use role separation and delimiters).

---

## Quality Gates

- All PRs must pass `pytest tests/` with 0 failures before merge.
- `tests/test_batch.py` must achieve 10/10 correct label predictions.
- No new dependencies may be added to `requirements.txt` without PR justification.
- API response time p95 must remain below 5 seconds for single-document inputs.

---

## Governance

This constitution supersedes all other guidelines. Amendments require updating this file with a version bump and a description of the change.

**Version**: 1.0.0 | **Ratified**: 2026-05-27 | **Last Amended**: 2026-05-27
