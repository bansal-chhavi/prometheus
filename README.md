# Prometheus

Prometheus is a zero-cost, open-source pipeline for detecting hallucinations in Retrieval-Augmented Generation (RAG) applications. It evaluates whether an AI-generated answer faithfully represents its source documents, detects hallucinated spans, answer drift, and unsupported generalizations, and returns a faithfulness score with pinpointed problem spans.

## Features

- **Claim Extraction**: Breaks down AI answers into verifiable atomic factual claims.
- **NLI Scoring**: Uses a HuggingFace DeBERTa Natural Language Local Inference model to verify claims against source documents.
- **Graceful Fallback**: Automatically falls back to an LLM provider (e.g., Groq) if HuggingFace rate limits are hit.
- **Span Extraction**: Pinpoints the exact substrings in the answer that are hallucinated.
- **Drift Detection**: Detects if the answer goes off-topic from the original user query.
- **Zero Cost Infrastructure**: Designed to run using Groq's free tier and HuggingFace's free inference API without local GPU requirements.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/prometheus.git
cd prometheus

# Install all dependencies including dev, api, mcp, and observability tools
pip install -e ".[all]"
```

## Setup & Configuration

You will need an API key from an LLM provider (Groq is the default) and optionally a HuggingFace token.

```bash
export GROQ_API_KEY="your_groq_api_key"
export HF_API_TOKEN="your_huggingface_token" # Optional, but recommended for NLI
```

You can change the provider via environment variables:
```bash
export PROMETHEUS_LLM_PROVIDER="openai" # or "anthropic"
export OPENAI_API_KEY="your_openai_key"
```

## Usage

### 1. Python API
You can run evaluations programmatically in your own scripts:

```bash
python example.py
```

### 2. FastAPI Server
Start a FastAPI server to run evaluations via HTTP:

```bash
uvicorn server.api:app --port 8100
```
Then send a POST request:
```bash
curl -X POST http://localhost:8100/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the revenue?",
    "rag_answer": "The revenue is $10 million.",
    "sources": ["The company made 10 million dollars in revenue."]
  }'
```

### 3. MCP Server
Run the MCP server to integrate with AI assistants:

```bash
python -m server.mcp_server
```

## Evaluation Suite

To run the built-in test evaluation suite of 10 FAITHFUL and HALLUCINATION cases:

```bash
python -m eval.run_eval
```

## Testing

Run the full test suite using `pytest`:

```bash
pytest tests/
```

## Observability

Prometheus includes built-in support for Datadog tracing and metrics. If you have the `ddtrace` and `datadog` packages installed, metrics and spans are automatically emitted for pipeline steps. If the packages are not installed, it gracefully degrades with zero errors.
