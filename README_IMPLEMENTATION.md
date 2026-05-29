# 🔍 Prometheus RAG Faithfulness Evaluator

A stateless REST API for real-time evaluation of RAG (Retrieval-Augmented Generation) answer faithfulness against source documents with automatic hallucination detection.

## 📋 Overview

**Purpose**: Evaluate whether LLM-generated answers are faithfully grounded in source documents, identifying and flagging unsupported claims (hallucinations).

**Key Features**:
- ✅ Real-time evaluation with immediate response
- ✅ Hallucination detection with span identification
- ✅ Faithfulness score (0-1) with classification
- ✅ Natural language explanations
- ✅ Fully stateless - no persistence required
- ✅ LLM-powered claim extraction and evidence matching
- ✅ REST API with async support
- ✅ Docker-ready deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (or Anthropic)
- pip/virtual environment

### Installation

```bash
# 1. Clone/navigate to project
cd prometheus

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### Run the Server

```bash
python -m uvicorn src.main:app --reload
```

Visit: http://localhost:8000/docs (Swagger UI)

## 📝 Example Usage

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How did the fraud detection project improve banking operations?",
    "answer": "It reduced false-positive alerts by 28% and improved response times by 35%.",
    "documents": [
      "The fraud detection modernization reduced false-positive fraud alerts by 28% and improved investigation response times by 35%."
    ]
  }'
```

**Response**:
```json
{
  "score": 0.95,
  "label": "FAITHFUL",
  "hallucinated_spans": [],
  "evidence_spans": ["reduced false-positive fraud alerts by 28%", "improved response times by 35%"],
  "explanation": "The answer correctly summarizes the key improvements with strong evidence from the source documents."
}
```

## 🏗️ Architecture

### Pipeline Stages

```
Input Request
    ↓
[1] Validation (Pydantic schema)
    ↓
[2] Preprocessing (normalize, deduplicate documents)
    ↓
[3] Claim Extraction (LLM decomposes answer into atomic claims)
    ↓
[4] Evidence Matching (LLM checks each claim against docs)
    ↓
[5] Hallucination Detection (identifies unsupported spans)
    ↓
[6] Scoring (computed ratio: supported_claims / total_claims)
    ↓
[7] Classification (FAITHFUL if score ≥ threshold, else HALLUCINATION)
    ↓
[8] Explanation (generate natural language reasoning)
    ↓
JSON Response
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Models** | `src/models.py` | Pydantic request/response schemas |
| **Evaluator** | `src/evaluator.py` | LLM-powered claim extraction and evidence matching |
| **Scorer** | `src/scorer.py` | Faithfulness score computation and labeling |
| **Explainer** | `src/explainer.py` | Natural language explanation generation |
| **Preprocessor** | `src/preprocessor.py` | Text normalization, chunking, span extraction |
| **API** | `src/main.py` | FastAPI application and endpoints |
| **Config** | `src/config.py` | Environment configuration management |

## 📊 API Reference

### POST /evaluate
Evaluate RAG answer faithfulness against source documents.

**Request Schema**:
```json
{
  "query": "string (required)",
  "answer": "string (required)",
  "documents": ["string", "... (required, min 1 item)"]
}
```

**Response Schema**:
```json
{
  "score": "number (0.0-1.0)",
  "label": "string (FAITHFUL | HALLUCINATION)",
  "hallucinated_spans": ["string"],
  "evidence_spans": ["string"],
  "explanation": "string"
}
```

**Status Codes**:
- `200 OK` - Successful evaluation
- `422 Unprocessable Entity` - Invalid input
- `500 Internal Server Error` - LLM or processing error

### GET /health
Health check endpoint for liveness probes.

**Response**:
```json
{
  "status": "ok"
}
```

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/test_preprocessor.py -v
pytest tests/test_scorer.py -v
```

### Run API Integration Tests
```bash
pytest tests/test_api.py -v
```

### Run All Tests
```bash
pytest tests/ -v --cov=src
```

### Batch Evaluation Against Test Cases
```bash
python scripts/batch_evaluate.py
```

This evaluates all 10 labeled cases in `Inputs/` and generates:
- Per-case predictions vs ground truth
- Accuracy metrics
- Detailed JSON report

## 📦 Project Structure

```
prometheus/
├── src/                          # Source code
│   ├── main.py                  # FastAPI app & endpoints
│   ├── models.py                # Pydantic schemas
│   ├── evaluator.py             # Core evaluation engine
│   ├── scorer.py                # Scoring logic
│   ├── explainer.py             # Explanation generation
│   ├── preprocessor.py          # Text preprocessing
│   ├── config.py                # Configuration
│   └── __init__.py
│
├── tests/                        # Test suite
│   ├── test_api.py              # API endpoint tests
│   ├── test_scorer.py           # Scorer unit tests
│   ├── test_preprocessor.py     # Preprocessor unit tests
│   ├── conftest.py              # Pytest fixtures
│   └── __init__.py
│
├── scripts/
│   └── batch_evaluate.py        # Batch evaluation runner
│
├── Inputs/                       # Test data (10 labeled cases)
│   ├── prometheus_eval_01.json  (FAITHFUL - Banking/Fraud Detection)
│   ├── prometheus_eval_02.json  (FAITHFUL - Healthcare/Scheduling)
│   ├── prometheus_eval_03.json  (FAITHFUL - Manufacturing/Maintenance)
│   ├── prometheus_eval_04.json  (FAITHFUL - Telecom/Retention)
│   ├── prometheus_eval_05.json  (HALLUCINATION - Energy Forecasting)
│   ├── prometheus_eval_06.json  (HALLUCINATION - Pharmacy Operations)
│   ├── prometheus_eval_07.json  (HALLUCINATION - Rail Inspection)
│   ├── prometheus_eval_08.json  (HALLUCINATION - Education Enrollment)
│   ├── prometheus_eval_09.json  (HALLUCINATION - Logistics Volume)
│   └── prometheus_eval_10.json  (FAITHFUL - Supply Chain)
│
├── specs/                        # Specification documents
│   └── 001-prometheus-faithfulness-evaluator/
│       ├── spec.md              # Feature specification
│       ├── plan.md              # Implementation plan
│       ├── tasks.md             # Task breakdown
│       └── constitution.md      # Design principles
│
├── Dockerfile                    # Container definition
├── docker-compose.yml           # Multi-container config
├── requirements.txt             # Python dependencies
├── .env.example                # Environment template
├── pytest.ini                   # Pytest configuration
├── SETUP.md                     # Setup instructions
└── README.md                    # This file
```

## 🔧 Configuration

Create `.env` from `.env.example` and configure:

```env
# LLM Configuration (required)
LLM_PROVIDER=openai                # 'openai' or 'anthropic'
LLM_MODEL=gpt-4                    # Model name
OPENAI_API_KEY=sk-...             # OpenAI API key
# ANTHROPIC_API_KEY=...           # Anthropic API key (if using)

# API Configuration
API_HOST=0.0.0.0                  # Server host
API_PORT=8000                     # Server port

# Evaluation Configuration
FAITHFULNESS_THRESHOLD=0.8        # Score threshold for FAITHFUL label
MAX_CHUNK_SIZE=1500               # Max doc chunk size
DEBUG=false                       # Enable debug mode
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t prometheus-evaluator:1.0 .
```

### Run Container
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  prometheus-evaluator:1.0
```

### Or Use Docker Compose
```bash
docker-compose up -d
```

## 📈 Performance

- **Latency**: p95 < 5 seconds per request (LLM-bound)
- **Throughput**: ~12-20 evaluations/minute per instance
- **Scalability**: Fully stateless - horizontal scaling via load balancing

## 🎯 Key User Stories

1. **US1 - Faithful Answer Detection** ✅
   - Recognize and score faithful answers correctly

2. **US2 - Hallucination Detection** ✅
   - Identify specific hallucinated spans (numeric, text)

3. **US3 - Structured JSON Output** ✅
   - Return machine-readable evaluation results

4. **US4 - Stateless REST API** ✅
   - No persistence, deterministic results

5. **US5 - Batch Evaluation** ✅
   - Benchmark against 10 labeled test cases

## 🔍 Evaluation Criteria

- **Score**: Ratio of supported claims to total claims (0.0-1.0)
- **Label**: FAITHFUL (score ≥ 0.8) or HALLUCINATION (score < 0.8)
- **Hallucinated Spans**: Text fragments not supported by source documents
- **Evidence Spans**: Text from documents supporting the answer
- **Explanation**: 2-4 sentence reasoning for the evaluation result

## ⚡ Dependencies

- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **Pydantic v2** - Data validation
- **OpenAI SDK** - LLM integration (optional)
- **Anthropic SDK** - LLM integration (optional)
- **pytest** - Testing framework
- **httpx** - Async HTTP client

## 🛠️ Development

### Add a New Feature
1. Create module in `src/`
2. Add tests in `tests/`
3. Update `src/main.py` if adding endpoint
4. Run tests: `pytest tests/ -v`

### Customize Threshold
Edit evaluation config in code or environment:
```python
FAITHFULNESS_THRESHOLD=0.75  # More lenient
FAITHFULNESS_THRESHOLD=0.90  # More strict
```

### Switching LLM Provider
```env
# For OpenAI (default)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# For Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-opus-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

## ⚠️ Limitations

- **LLM Dependent**: Quality depends on LLM performance
- **Latency**: LLM calls add 2-5 second latency per request
- **Cost**: Charged per LLM API call
- **No Long-term Storage**: Fully stateless, no audit trail
- **No Caching**: Each request fully evaluated

## 🤝 Contributing

1. Follow the task list in `specs/001-prometheus-faithfulness-evaluator/tasks.md`
2. Write tests for new features
3. Ensure all tests pass: `pytest tests/ -v`
4. Update documentation as needed

## 📄 License

Check LICENSE file

## 🆘 Support

### Common Issues

**Q: Getting "OPENAI_API_KEY not set"**
A: Set `OPENAI_API_KEY` in `.env` and restart the server

**Q: API returning 500 errors**
A: Check logs: `docker logs prometheus-evaluator` (if running in Docker)

**Q: Tests failing**
A: Ensure dependencies installed: `pip install -r requirements.txt`

**Q: Slow response times**
A: LLM calls typically take 2-5 seconds; this is expected

---

**Built for the Prometheus Team** | **Assignment: RAG Faithfulness Evaluation**
