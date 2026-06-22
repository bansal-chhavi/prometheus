import asyncio
import os
from mcp.server.fastmcp import FastMCP
from prometheus.config import PrometheusConfig, LLMProvider
from prometheus.pipeline import PrometheusPipeline
from prometheus.logging_config import setup_logging

mcp = FastMCP("Prometheus")

@mcp.tool()
async def evaluate_faithfulness(query: str, rag_answer: str, sources: list[str]) -> str:
    """Evaluates whether a RAG-generated answer faithfully represents its source documents."""
    provider_str = os.environ.get("PROMETHEUS_LLM_PROVIDER", "groq")
    model_str = os.environ.get("PROMETHEUS_LLM_MODEL", "llama-3.3-70b-versatile")
    
    config = PrometheusConfig(
        llm_provider=LLMProvider(provider_str),
        llm_model=model_str
    )
    
    pipeline = PrometheusPipeline(config)
    result = await pipeline.evaluate(query, rag_answer, sources)
    return result.model_dump_json()

def main():
    setup_logging()
    mcp.run()

if __name__ == "__main__":
    main()
