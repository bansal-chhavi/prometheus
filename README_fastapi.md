# Running the TrustSignal FastAPI endpoint (short)

This short README shows how to run a local FastAPI server that exposes the TrustSignal `pipeline.evaluate()` endpoint.

Prerequisites
- Python 3.10+ recommended
- Install runtime deps:

```bash
python -m venv .venv
source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install fastapi uvicorn
```

Start the server
- If you have an app module at `src.trustsignal.api` exposing a FastAPI `app` object, run:

```bash
# from repository root
uvicorn src.trustsignal.api:app --reload --host 127.0.0.1 --port 8000
```

Example request
- POST to `/evaluate` with JSON body containing `query`, `rag_answer`, and `sources` (an array of `{id, text}`):

```bash
curl -sS -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query":"Who won 2020 election?","rag_answer":"Candidate A won the 2020 election.","sources":[{"id":"s1","text":"Candidate A won the 2020 election."}] }'
```

Notes
- The repository includes a minimal prototype of `pipeline.evaluate()` at `src/trustsignal/pipeline.py` for local testing.
- The FastAPI app should call `from src.trustsignal.pipeline import evaluate, TrustSignalConfig` and return the `EvalResult` as JSON.
- Configure model/provider settings via environment variables or programmatic `TrustSignalConfig()` when wiring into production.

Next steps
- I can create a minimal `src/trustsignal/api.py` FastAPI app that wraps `pipeline.evaluate()` (would you like that?).
