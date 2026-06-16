# Async & Retry Guidance for MCP calls

This note contains concise guidance and examples for calling the MCP handler safely from other services.

1) Prefer async endpoint when possible
- Use `/mcp/evaluate-async` from async callers to avoid blocking the event loop. The endpoint runs the synchronous evaluator in a threadpool.

2) HTTP client best-practices (timeouts + retries)
- Use a small connect/read timeout and an exponential backoff retry strategy for transient errors (502/503/429, timeouts).
- Example using `httpx` with simple retries:

```py
import httpx
import asyncio

async def post_with_retries(url, json_payload, retries=3):
    backoff = 0.5
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=10.0)) as client:
        for attempt in range(retries):
            try:
                r = await client.post(url, json=json_payload)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                # retry on 429, 502, 503 or network errors
                status = getattr(exc, 'response', None)
                code = status.status_code if status is not None else None
                if code in (429, 502, 503) or isinstance(exc, httpx.RequestError):
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise
        raise RuntimeError('Max retries exceeded')
```

3) Programmatic / in-process calls
- Prefer `handle_mcp_request(payload)` for in-process calls (fast, no HTTP overhead). If you need non-blocking behavior from an async app, call `async_handle_mcp_request(payload)` instead.

4) Timeouts and worker sizing
- The pipeline can be CPU-bound when running many NLI checks; if you expose the endpoint, configure worker counts accordingly (uvicorn/gunicorn settings) and set per-request timeouts.

5) Observability
- Log fallback events (HF → Groq) and surface them via `on_nli_notification` so callers can adapt retries or degrade gracefully.
