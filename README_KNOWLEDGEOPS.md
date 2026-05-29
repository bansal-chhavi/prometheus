# 🔍 Prometheus Inspector - RAG Faithfulness Evaluator
## KnowledgeOps Championship Module | Team Prometheus

**Module Name**: Inspector  
**Team**: Prometheus (Heena, Chhavi, Aakash, Vinay, Srividya, Aparana, Manik)  
**Task**: Build a faithfulness evaluator that detects hallucinations and verifies RAG answer correctness  
**Status**: ✅ Week 1-3 Build Complete

---

## 📋 Assignment Overview

As part of the **KnowledgeOps Championship**, the Prometheus team is building the **Inspector module** - a trust verification system that evaluates whether RAG-generated answers are factually grounded in source documents.

### The Vision
> "Before an AI answer reaches a BCG consultant, KnowledgeOps ensures it is factually verified, source-traced, graph-validated, expert-aligned, and compliant — or it is flagged."

### What This Module Does

The Prometheus Inspector:
1. ✅ **Verifies** factual claims in RAG answers
2. ✅ **Detects** hallucinated or contradicted statements  
3. ✅ **Scores** faithfulness (0-1)
4. ✅ **Flags** unsupported claims
5. ✅ **Traces** evidence back to source documents

---

## 🏗️ Architecture

### Input: KnowledgeTask
```json
{
  "query": "string",
  "rag_answer": "string",
  "sources": [
    {"text": "string", "doc_id": "string", "chunk_id": 0}
  ],
  "task_type": "factual|synthesis|expert|regulatory|exploratory"
}
```

### Output: TrustSignal
```json
{
  "score": 0.95,
  "evidence": ["supporting text"],
  "flags": ["UNSUPPORTED"],
  "applies_to": ["factual"],
  "module": "prometheus-inspector",
  "details": {
    "hallucinated_spans": ["wrong claim"],
    "evidence_spans": ["correct claim"],
    "claim_scores": [{"claim": "...", "score": 1}],
    "explanation": "..."
  }
}
```

### Pipeline
```
KnowledgeTask
    ↓
[1] Input Validation
    ↓
[2] Document Preprocessing (normalize, deduplicate)
    ↓
[3] Claim Extraction (LLM decomposes answer)
    ↓
[4] Evidence Matching (LLM verifies against sources)
    ↓
[5] Hallucination Detection (identifies unsupported spans)
    ↓
[6] Faithfulness Scoring (supported/total claims)
    ↓
[7] Explanation Generation (natural language reasoning)
    ↓
TrustSignal
```

---

## 📊 Module Specification

### What It Scores
- **Score**: Faithfulness ratio (supported claims / total claims)
- **Flags**: UNSUPPORTED, CONTRADICTED
- **Evidence**: Source passages backing each claim
- **Hallucinated Spans**: Specific text that is wrong

### What It Returns
- Atomic faithfulness score (0-1)
- Structured evidence references  
- Explicit hallucination identification
- Confidence explanation

### Task Types Handled
- ✅ **factual** - Direct factual verification (primary)
- 🔲 synthesis - Multi-fact synthesis (future)
- 🔲 expert - Expert alignment (via Scout module)
- 🔲 regulatory - Compliance checking (via Themis module)
- 🔲 exploratory - Open-ended reasoning (future)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI/Anthropic API key

### Installation

```bash
cd c:\github\prometheus

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key
```

### Run the Server

```bash
# Development (with hot reload)
python -m uvicorn src.main:app --reload

# Production
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Test the API

**Swagger UI**: http://localhost:8000/docs

**Example Request to KnowledgeTask endpoint**:
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How did Project X improve performance?",
    "rag_answer": "Project X improved performance by 30%",
    "sources": [
      {
        "text": "Project X delivered a 30% improvement in efficiency",
        "doc_id": "proj_001",
        "chunk_id": 1
      }
    ],
    "task_type": "factual"
  }'
```

**Response**:
```json
{
  "score": 0.95,
  "evidence": ["30% improvement"],
  "flags": [],
  "applies_to": ["factual"],
  "module": "prometheus-inspector",
  "details": {
    "hallucinated_spans": [],
    "evidence_spans": ["30% improvement"],
    "claim_scores": [{"claim": "Project X improved performance by 30%", "score": 1}],
    "explanation": "The answer is fully supported by source documents..."
  }
}
```

---

## 📦 Project Structure

```
prometheus/
├── src/                              # Source code
│   ├── main.py                      # FastAPI app + KnowledgeTask/TrustSignal endpoints
│   ├── models.py                    # Pydantic schemas (KnowledgeTask, TrustSignal)
│   ├── evaluator.py                 # Core evaluation engine (LLM-based)
│   ├── scorer.py                    # Faithfulness scoring logic
│   ├── explainer.py                 # Explanation generation
│   ├── preprocessor.py              # Text preprocessing
│   ├── config.py                    # Configuration management
│   └── __init__.py
│
├── tests/                            # Test suite
│   ├── test_api.py                  # API integration tests
│   ├── test_scorer.py               # Scoring unit tests
│   ├── test_preprocessor.py         # Preprocessing unit tests
│   ├── conftest.py                  # Pytest fixtures
│   └── __init__.py
│
├── scripts/
│   └── batch_evaluate.py            # Batch evaluation runner
│
├── Inputs/                           # Evaluation test cases (10 files, 5 FAITHFUL + 5 HALLUCINATION)
│   ├── prometheus_eval_01.json
│   ├── prometheus_eval_02.json
│   ├── ...
│   └── prometheus_eval_10.json
│
├── Dockerfile                        # Container definition
├── docker-compose.yml               # Docker Compose setup
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment template
├── pytest.ini                       # Pytest config
├── SETUP.md                         # Setup guide
├── README_IMPLEMENTATION.md         # Implementation details
└── README_KNOWLEDGEOPS.md          # This file
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Unit Tests Only
```bash
pytest tests/test_preprocessor.py tests/test_scorer.py -v
```

### Run API Integration Tests
```bash
pytest tests/test_api.py -v
```

### Batch Evaluate Against Test Cases
```bash
python scripts/batch_evaluate.py
```

This evaluates all 10 JSON files in Inputs/ against ground truth labels and generates accuracy metrics.

---

## 🐳 Docker Deployment

### Build
```bash
docker build -t prometheus-inspector:1.0 .
```

### Run
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  prometheus-inspector:1.0
```

### Docker Compose
```bash
docker-compose up -d
```

---

## ⚙️ Configuration

Create `.env` file:

```env
# LLM Configuration (required)
LLM_PROVIDER=openai                # or 'anthropic'
LLM_MODEL=gpt-4                    # Model name
OPENAI_API_KEY=sk-...             # Your OpenAI key

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8000

# Evaluation Settings
FAITHFULNESS_THRESHOLD=0.8        # Score >= 0.8 = only unsupported claims
MAX_CHUNK_SIZE=1500               # Max doc chunk size
DEBUG=false
```

---

## 🎯 Integration Points

### Trust Router Integration
The module connects to the Trust Router (central coordinator) via:
- **Input**: REST POST with KnowledgeTask JSON
- **Output**: TrustSignal JSON response
- **Contract**: Shared across all 5 modules (ATHENA, APOLLO, PROMETHEUS, HERMES, THEMIS)

### MCP Server (Stretch Goal)
Wrap the `/evaluate` endpoint as an MCP tool to enable:
- Direct Claude access
- LLM agent integration
- **Bonus points** in KnowledgeOps championship

---

##  📈 Key Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Latency (p95) | < 5 seconds | LLM-bound |
| Faithfulness Accuracy | > 90% | Depends on corpus |
| Label F1-Score | > 0.85 | Depends on corpus |
| Test Coverage | > 80% | ✅ Achieved |

---

## 🔄 Week 1-3 Deliverables ✅

### Week 1: Spec & Corpus
- ✅ 10 evaluation test cases created (prometheus_eval_01-10.json)
- ✅ 5 FAITHFUL + 5 HALLUCINATION ground truth labels
- ✅ Module specification locked
- ✅ Project scaffold + dependencies

### Week 2: Build
- ✅ Claim extraction engine (LLM-powered)
- ✅ Evidence matching logic
- ✅ Hallucination detection
- ✅ Faithfulness scoring
- ✅ REST /evaluate endpoint
- ✅ Explanation generation

### Week 3: Polish & Demo
- ✅ Cross-corpus testing readiness
- ✅ Standalone demo script
- ✅ Comprehensive error handling
- ✅ Integration handshake ready
- ✅ Pitch prepared

---

## 🚀 Next: Integration Phase (Weeks 4-6)

After Week 3 demo, the module integrates with:
- **Trust Router**: Receives KnowledgeTasks from central coordinator
- **Other Modules**: Data flows to/from ATHENA, APOLLO, HERMES, THEMIS
- **MCP Server**: Optional enhanced integration
- **Demo Day**: Live launch with all 5 modules

---

## 🆘 Support

### Common Issues

**Q: Getting "LLM not configured" error**  
A: Set `OPENAI_API_KEY` in `.env` file and restart server

**Q: API returns 500 errors**  
A: Check logs: `docker logs prometheus-inspector` or review stderr

**Q: Tests failing**  
A: Ensure dependencies: `pip install -r requirements.txt`

### Documentation
- [SETUP.md](SETUP.md) - Setup and troubleshooting
- [README_IMPLEMENTATION.md](README_IMPLEMENTATION.md) - Technical implementation details
- [specs/001-prometheus-faithfulness-evaluator/](specs/001-prometheus-faithfulness-evaluator/) - Original specifications

---

## 📝 License & Attribution

Built for **KnowledgeOps Championship 2026**  
Team: **Prometheus (Inspector module)**  
Date: **May 2026**

---

## 🏆 Key Achievements

✅ **Complete REST API** - KnowledgeTask/TrustSignal contract  
✅ **LLM-Powered Evaluation** - Sophisticated claim/evidence matching  
✅ **Hallucination Detection** - Identifies specific unsupported spans  
✅ **Comprehensive Testing** - 30+ unit and integration tests  
✅ **Production Ready** - Docker, logging, error handling  
✅ **Fully Documented** - Specs, setup guides, inline comments  

---

**Questions?** Check [SETUP.md](SETUP.md) or contact the Prometheus team.
