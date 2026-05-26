# prometheus

🔹 Purpose
Real-time evaluation of RAG answer faithfulness
No persistence → fully stateless, request-response system
Returns score + hallucination spans + explanation instantly

🔹 1. Inputs
Query
RAG Answer
Source Documents (context chunks)

🔹 2. Preprocessing (On-the-Fly)
Light Chunking (if needed)
Split large documents dynamically
Span Extraction
Identify key parts:
claims in answer
supporting evidence in sources
Normalization
Clean text, remove duplicates, standardize format

🔹 3. Evaluation Engine (Core)
Claim vs Evidence Matching
Compare answer statements with source content
Hallucination Detection
Flag unsupported or incorrect spans
Faithfulness Scoring
Compute:
faithfulness score (0–1)
drift / unsupported signals
Explanation Generator
Provide reasoning:
which parts are correct
which are hallucinated

🔹 4. Output (Immediate Response)
Returns structured JSON:
score
hallucinated_spans
evidence_spans
explanation
Delivered via:
REST API (/evaluate)
Optional SDK / MCP wrapper

🔹 5. Observability (Lightweight, Optional)
Logging (optional) – request/response traces
Metrics (optional) – latency, error rates
No long-term storage required

🔹 End-to-End Flow
Receive (Query + Answer + Sources)
Extract claims & evidence in real time
Compare and detect hallucinations
Compute score + generate explanation
Return response immediately

🔹 Key Characteristics
⚡ Low latency (no DB calls)
🔄 Stateless & scalable (easy horizontal scaling)
🧩 Plug-and-play API for any RAG system
🚫 No corpus management or persistence overhead
