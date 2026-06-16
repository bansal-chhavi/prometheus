from typing import Dict, Any, List
from fastapi import FastAPI

from src.trustsignal.pipeline import evaluate, TrustSignalConfig


def handle_mcp_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Programmatic handler for MCP-like requests.

    Expects payload keys: `query`, `rag_answer`, `sources` (list of {id, text}), optional `config`.
    Returns a JSON-serializable EvalResult dict.
    """
    query = payload.get("query", "")
    rag_answer = payload.get("rag_answer", "")
    sources = payload.get("sources", [])
    cfg_kwargs = payload.get("config") or {}
    cfg = TrustSignalConfig(**cfg_kwargs)
    res = evaluate(query, rag_answer, sources, cfg)
    from dataclasses import asdict

    return asdict(res)


async def async_handle_mcp_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Async wrapper that runs the sync handler in a threadpool.

    Useful for async frameworks or when you want non-blocking behavior.
    """
    import asyncio

    return await asyncio.to_thread(handle_mcp_request, payload)


# Optional FastAPI surface for MCP-compatible endpoint
app = FastAPI(title="TrustSignal MCP (minimal)")


@app.post("/mcp/evaluate")
def mcp_evaluate(payload: Dict[str, Any]):
    return handle_mcp_request(payload)


if __name__ == "__main__":
    # Local quick test
    sample = {
        "query": "Who won 2020 election?",
        "rag_answer": "Candidate A won the 2020 election.",
        "sources": [{"id": "s1", "text": "Candidate A won the 2020 election."}],
    }
    print(handle_mcp_request(sample))
