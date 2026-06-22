from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from prometheus.config import PrometheusConfig
from prometheus.pipeline import PrometheusPipeline
from prometheus.models import EvalRequest, EvalResult
from prometheus.logging_config import setup_logging
from prometheus.observability import DD_AVAILABLE

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    setup_logging()
    config = PrometheusConfig()
    pipeline = PrometheusPipeline(config)
    yield
    # Cleanup if needed
    pipeline = None

app = FastAPI(title="Prometheus API", version="0.1.0", lifespan=lifespan)

if DD_AVAILABLE:
    try:
        from ddtrace.contrib.asgi import TraceMiddleware
        app.add_middleware(TraceMiddleware)
    except ImportError:
        pass

@app.post("/evaluate", response_model=EvalResult)
async def evaluate(req: EvalRequest):
    return await pipeline.evaluate(req.query, req.rag_answer, req.sources)

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
