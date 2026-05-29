# Getting Started with Prometheus RAG Faithfulness Evaluator

## Quick Start

### 1. Configure Environment Variables

Copy `.env.example` to `.env` and update with your LLM credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
LLM_PROVIDER=openai           # or 'anthropic'
LLM_MODEL=gpt-4              # adjust as needed
OPENAI_API_KEY=sk-...        # your OpenAI key
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the Server

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 4. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Using the API

### Evaluate an Answer

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the improvement?",
    "answer": "The system improved by 24%.",
    "documents": ["Our system achieved a 24% improvement in efficiency."]
  }'
```

### Response Example

```json
{
  "score": 0.95,
  "label": "FAITHFUL",
  "hallucinated_spans": [],
  "evidence_spans": ["24% improvement in efficiency"],
  "explanation": "The answer correctly summarizes the improvements stated in the document."
}
```

## Running Tests

### Unit Tests

```bash
pytest tests/test_preprocessor.py -v
pytest tests/test_scorer.py -v
```

### API Integration Tests

```bash
pytest tests/test_api.py -v
```

### All Tests

```bash
pytest tests/ -v --cov=src
```

## Batch Evaluation

Evaluate against all labeled test cases:

```bash
python scripts/batch_evaluate.py
```

This will:
- Load all 10 evaluation cases from `Inputs/`
- Run each through the evaluation pipeline
- Generate accuracy metrics
- Save detailed report to `batch_evaluation_report.json`

## Docker Deployment

### Build Image

```bash
docker build -t prometheus-evaluator:1.0 .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e LLM_PROVIDER=openai \
  prometheus-evaluator:1.0
```

### Or Use Docker Compose

```bash
docker-compose up -d
```

## Project Structure

```
prometheus/
├── src/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic request/response schemas
│   ├── evaluator.py         # Core evaluation engine (LLM-based)
│   ├── scorer.py            # Faithfulness scoring
│   ├── explainer.py         # Explanation generation
│   ├── preprocessor.py      # Text preprocessing
│   └── config.py            # Configuration management
│
├── tests/
│   ├── test_api.py          # API integration tests
│   ├── test_scorer.py       # Scorer unit tests
│   ├── test_preprocessor.py # Preprocessor unit tests
│   ├── conftest.py          # Pytest configuration
│   └── __init__.py
│
├── scripts/
│   └── batch_evaluate.py    # Batch evaluation runner
│
├── Inputs/                  # 10 labeled evaluation cases
│   ├── prometheus_eval_01.json (FAITHFUL - Banking)
│   ├── prometheus_eval_02.json (FAITHFUL - Healthcare)
│   ├── prometheus_eval_03.json (FAITHFUL - Manufacturing)
│   ├── prometheus_eval_04.json (FAITHFUL - Telecom)
│   ├── prometheus_eval_05.json (HALLUCINATION - Energy)
│   ├── prometheus_eval_06.json (HALLUCINATION - Pharmacy)
│   ├── prometheus_eval_07.json (HALLUCINATION - Rail)
│   ├── prometheus_eval_08.json (HALLUCINATION - Education)
│   ├── prometheus_eval_09.json (HALLUCINATION - Logistics)
│   └── prometheus_eval_10.json (FAITHFUL - Supply Chain)
│
├── specs/                   # Design specifications
│   └── 001-prometheus-faithfulness-evaluator/
│       ├── spec.md
│       ├── plan.md
│       ├── tasks.md
│       └── constitution.md
│
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container config
└── README.md
```

## API Endpoints

### POST /evaluate
Evaluate RAG answer faithfulness

**Request**:
```json
{
  "query": "string",
  "answer": "string",
  "documents": ["string", "..."]
}
```

**Response**:
```json
{
  "score": 0.0-1.0,
  "label": "FAITHFUL|HALLUCINATION",
  "hallucinated_spans": ["string", "..."],
  "evidence_spans": ["string", "..."],
  "explanation": "string"
}
```

### GET /health
Health check for liveness probes

**Response**:
```json
{
  "status": "ok"
}
```

## Configuration

Environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider: `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4` | Model name |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `FAITHFULNESS_THRESHOLD` | `0.8` | Score threshold for FAITHFUL label |
| `MAX_CHUNK_SIZE` | `1500` | Max characters per document chunk |
| `DEBUG` | `false` | Enable debug mode |

## Troubleshooting

### LLM Configuration Error
```
ValueError: OPENAI_API_KEY not set in environment
```
**Solution**: Set `OPENAI_API_KEY` in `.env` or environment

### API not responding
```
ConnectionError: Cannot connect to localhost:8000
```
**Solution**: Ensure server is running: `python -m uvicorn src.main:app --reload`

### Import errors
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

## Architecture

The evaluator uses a multi-stage pipeline:

1. **Input Validation** - Pydantic schema validation
2. **Preprocessing** - Text normalization, deduplication
3. **Claim Extraction** - LLM decomposes answer into atomic claims
4. **Evidence Matching** - LLM checks claim support against documents
5. **Hallucination Detection** - Identifies unsupported spans
6. **Scoring** - Computes faithfulness score (0-1)
7. **Explanation** - Generates natural language reasoning

All processing is **stateless** and **deterministic** for the same inputs.

## Performance

- **Latency**: p95 < 5 seconds per evaluation (LLM-bound)
- **Throughput**: ~12-20 evals/minute per instance
- **Scalability**: Horizontal - no shared state, can load-balance across instances

## License

See LICENSE file

## Support

For issues or questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Review API docs at http://localhost:8000/docs
3. Check test files for usage examples
