from typing import List, Dict, Any, Optional
import logging
from fastapi import FastAPI
from pydantic import BaseModel

from src.trustsignal.pipeline import evaluate, TrustSignalConfig
# reuse the MCP handler to expose /mcp/evaluate on the same app
from src.trustsignal.mcp_server import handle_mcp_request
from src.trustsignal.mcp_server import async_handle_mcp_request


class Source(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class EvalRequest(BaseModel):
    query: str
    rag_answer: str
    sources: List[Source]
    config: Optional[Dict[str, Any]] = None


app = FastAPI(title="TrustSignal (minimal) API")


@app.on_event("startup")
async def on_startup():
    logging.getLogger("uvicorn").info("TrustSignal API starting up")


@app.on_event("shutdown")
async def on_shutdown():
    logging.getLogger("uvicorn").info("TrustSignal API shutting down")


@app.get("/")
def root():
    return {"status": "ok", "info": "TrustSignal minimal API"}


@app.post("/evaluate")
def evaluate_endpoint(req: EvalRequest):
    cfg = TrustSignalConfig(**(req.config or {}))
    # convert pydantic sources to plain dicts expected by pipeline
    sources = [s.dict() for s in req.sources]
    res = evaluate(req.query, req.rag_answer, sources, cfg)
    # EvalResult is a dataclass; convert to dict for JSON response
    from dataclasses import asdict

    return asdict(res)


@app.post("/mcp/evaluate")
def mcp_evaluate(payload: Dict[str, Any]):
    """MCP-compatible endpoint forwarding to the programmatic handler.

    Accepts a JSON payload with `query`, `rag_answer`, `sources`, and optional `config`.
    """
    return handle_mcp_request(payload)


@app.post("/mcp/evaluate-async")
async def mcp_evaluate_async(payload: Dict[str, Any]):
    """Async MCP endpoint that forwards to the async handler.

    Use this route when calling from async services to avoid thread-blocking.
    """
    return await async_handle_mcp_request(payload)
