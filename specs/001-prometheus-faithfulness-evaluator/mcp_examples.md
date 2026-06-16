# Calling the TrustSignal MCP handler — Examples

This file shows simple ways other services can call the MCP-style handler provided in `src/trustsignal/mcp_server.py`.

1) Programmatic import (Python)

```py
from src.trustsignal.mcp_server import handle_mcp_request

payload = {
  "query": "Who won 2020 election?",
  "rag_answer": "Candidate A won the 2020 election.",
  "sources": [{"id": "s1", "text": "Candidate A won the 2020 election."}]
}

result = handle_mcp_request(payload)
print(result)
```

2) HTTP (MCP FastAPI surface)

POST JSON to `/mcp/evaluate` on the running server (defaults to `http://127.0.0.1:8000/mcp/evaluate`).

```bash
curl -sS -X POST http://127.0.0.1:8000/mcp/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query":"Who won 2020 election?","rag_answer":"Candidate A won the 2020 election.","sources":[{"id":"s1","text":"Candidate A won the 2020 election."}] }'
```

Expected response: JSON `EvalResult` shape with `score`, `flags`, `hallu_spans`, `explanation`, and `details`.

3) From another service (conceptual)
- Import `handle_mcp_request` and call it from a worker, lambda, or microservice.
- Alternatively, call the HTTP endpoint; wrap retries and timeouts when calling over the network.

Payload shape
- `query`: string
- `rag_answer`: string
- `sources`: array of `{id: str, text: str, metadata?: object}`
- optional `config`: object to override `TrustSignalConfig` fields

Notes
- `handle_mcp_request` is synchronous and returns a JSON-serializable dict.
- The FastAPI route `/mcp/evaluate` simply proxies to `handle_mcp_request`.
