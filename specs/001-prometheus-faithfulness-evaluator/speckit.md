**Prometheus: NLI + DeBERTa + Groq — Speckit**

Short description
- **What**: Concise operational spec for the TrustSignal faithfulness pipeline visualized in the diagram.
- **Source diagram**: [Prometheus_using_NLI_Deberta_Groq.svg](Prometheus_using_NLI_Deberta_Groq.svg#L1)

Overview
- Purpose: Evaluate an AI-generated answer for faithfulness to provided sources by performing per-claim NLI, mapping supporting spans, detecting semantic drift, and producing a structured `EvalResult` consumed by servers or a pip package.

Expanded pipeline stages
- **1. Claim decomposition (answer → claims)**
  - Input: `rag_answer` (text)
  - Output: `claims[]` — each claim: `{id, text, provenance}`
  - Algorithm: lightweight rule-based + small LLM prompt to split multi-sentence answers into atomic factual assertions. Keep deterministic prompts / caching.

- **2. NLI scoring (claim × source → label + score)**
  - Primary provider: HuggingFace DeBERTa-v3 (local/remote HF pipeline).
  - Fallback: Groq LLM-as-NLI when HF rate-limited (429) or unavailable.
  - Output: `claim_verdicts[]` entries `{claim_id, source_index, label: [ENTAILS|CONTRADICTS|NEUTRAL], confidence}`.
  - Behavior: run NLI for each (claim, candidate source passage); short-circuit if high-confidence entailment found.

- **3. Span extraction (support localization)**
  - Input: source document text + claim text
  - Output: `spans[]` — `{claim_id, source_index, start, end, snippet}` (character offsets)
  - Algorithm: exact match heuristics followed by fuzzy substring search; optionally use model to propose offsets when text is paraphrased.

- **4. Drift detection (query ↔ answer alignment)**
  - Input: `query`, `rag_answer`
  - Output: `drift_score` (0–1) and `drift_flags` (e.g., `TOPIC_SHIFT`, `OVERSTATED`)
  - Algorithm: embed-based similarity (semantic) + rule checks for hallucinated entities present in answer but absent from query or sources.

- **5. Aggregation & flagging**
  - Inputs: claim verdicts, spans, drift signals
  - Outputs: `score` (0–1), `flags[]`, `explanation` (text), `details` object
  - Algorithm: weighted aggregation (configurable weights). Example rule: if >50% claims CONTRADICT or no supporting span found for entailment → flag `HALLU`.

Inputs (formal)
- `query` (string)
- `rag_answer` (string)
- `sources[]` (array of {id, text, metadata}) — RAG-provided passages or full docs
- Optional: `config` overrides (thresholds, provider keys, model names)

Outputs (EvalResult schema)
- `score`: number (0.0–1.0)
- `flags`: string[] (e.g., `HALLU`, `DRIFT`, `LOW_CONFIDENCE`)
- `hallu.spans`: [{text, start, end, source_index}]
- `explanation`: string (human-readable rationale)
- `details`: {
  faithfulness_score: number,
  drift_score: number,
  claim_verdicts: [{claim_id, source_index, label, confidence}],
  test_cases_passed: number,
  test_cases_total: number
}

Example EvalResult (JSON)
```json
{
  "score": 0.72,
  "flags": ["LOW_CONFIDENCE"],
  "hallu": {"spans": [{"text":"...","start":234,"end":286,"source_index":0}]},
  "explanation": "Most claims are supported but two have low-confidence matches.",
  "details": {"faithfulness_score":0.75,"drift_score":0.6,"claim_verdicts":[]}
}
```

Config & operational notes
- Configure via env vars or programmatically with `TrustSignalConfig()`:
  - `TRUSTSIGNAL_HF_MODEL` (default: DeBERTa-v3)
  - `TRUSTSIGNAL_HF_ENDPOINT` / credentials
  - `TRUSTSIGNAL_GROQ_KEY`
  - `TRUSTSIGNAL_NLI_THRESHOLD` (float)
  - `TRUSTSIGNAL_FALLBACK_ON_429` (bool)
- Notification hook: `on_nli_notification(callback)` — callback receives `{reason, details}` when HF → Groq fallback occurs.

Performance & resilience
- NLI is parallelizable across claims and sources; pipeline is async/stateless.
- Cache recent NLI results and decomposition outputs to avoid repeated costs.

Delivery surfaces & usage
- All three delivery surfaces call the same `pipeline.evaluate()` API:
  - MCP server: `mcp_server.py` — integrates into model evaluation pipelines
  - FastAPI: `api.py` — exposes an HTTP endpoint
  - Library: `pip install trustsignal` — importable Python package

Minimal Python usage example
```py
from trustsignal import pipeline, TrustSignalConfig

cfg = TrustSignalConfig(hf_model="microsoft/deberta-v3-large")
res = pipeline.evaluate(query=my_query, rag_answer=ai_text, sources=sources, config=cfg)
print(res.score, res.flags)
```

CLI / curl example (FastAPI)
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query":"...","rag_answer":"...","sources":[{"id":"1","text":"..."}]}'
```

Testing guidance
- Add unit tests that: (1) run decomposition on sample answers, (2) simulate HF 429 and verify fallback to Groq via `on_nli_notification`, (3) assert EvalResult schema and threshold behavior.

Notes & assumptions
- Decomposition uses deterministic prompts to maintain reproducibility.
- Fallback behavior should be logged and notified through `on_nli_notification` so downstream systems can adapt.

Where to look next
- Implementation details and task breakdown live in this spec folder: `tasks.md`, `spec.md`, and `plan.md`.
