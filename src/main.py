"""
Main FastAPI application for the Prometheus RAG Faithfulness Evaluator.

Stateless REST API for evaluating the faithfulness of RAG-generated answers
against source documents with hallucination detection.

Implements two contracts:
1. KnowledgeTask/TrustSignal (official KnowledgeOps format)
2. EvaluationRequest/EvaluationResponse (legacy format for evaluation files)
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

from src.models import (
    KnowledgeTask, TrustSignal, TrustSignalDetails,
    EvaluationRequest, EvaluationResponse, HealthResponse
)
from src.preprocessor import normalize_text, deduplicate_documents, chunk_document
from src.evaluator import Evaluator
from src.scorer import Scorer
from src.explainer import Explainer
from src.config import get_evaluation_config

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Prometheus RAG Faithfulness Evaluator (Inspector)",
    description="KnowledgeOps module for detecting hallucinations and verifying RAG answer faithfulness",
    version="2.0.0"
)

# Initialize components
evaluator = None
scorer = Scorer()
explainer = None


def init_components():
    """Initialize evaluator and explainer (lazy initialization for testability)."""
    global evaluator, explainer
    try:
        evaluator = Evaluator()
        explainer = Explainer()
    except ValueError as e:
        print(f"Configuration error: {e}")
        raise


@app.on_event("startup")
async def startup():
    """Initialize components on startup."""
    try:
        init_components()
    except ValueError as e:
        print(f"Warning: LLM not configured - {e}. Configure environment variables to use the service.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for liveness probes.
    
    Returns:
        HealthResponse with status
    """
    return HealthResponse(status="ok")


@app.post("/evaluate", response_model=TrustSignal, status_code=200)
async def evaluate_knowledge_task(request: KnowledgeTask) -> TrustSignal:
    """
    Evaluate faithfulness of a RAG answer against source documents.
    
    Official KnowledgeOps API endpoint. Accepts KnowledgeTask and returns TrustSignal.
    
    This endpoint:
    1. Validates input via KnowledgeTask schema
    2. Extracts source texts from documents
    3. Preprocesses text (normalization, deduplication)
    4. Extracts atomic claims from the answer using LLM
    5. Matches each claim against source documents
    6. Detects hallucinated spans
    7. Computes faithfulness score (0-1)
    8. Returns TrustSignal with score, evidence, flags, and details
    
    Args:
        request: KnowledgeTask with query, rag_answer, sources, and task_type
        
    Returns:
        TrustSignal with score, evidence, flags, and module-specific details
        
    Raises:
        HTTPException: 422 if input validation fails, 500 if LLM call fails
    """
    try:
        if not evaluator or not explainer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM not configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )
        
        # Extract source texts and build document list
        source_texts = [source.text for source in request.sources]
        preprocessed_docs = deduplicate_documents(source_texts)
        
        if not preprocessed_docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid documents provided after preprocessing"
            )
        
        config = get_evaluation_config()
        
        # Extract claims from answer
        claims = evaluator.extract_claims(request.rag_answer)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract claims from answer"
            )
        
        # Match each claim to evidence
        support_map = {}
        evidence_spans = []
        
        for claim in claims:
            is_supported, evidence = evaluator.match_claim_to_evidence(
                claim, preprocessed_docs
            )
            support_map[claim] = is_supported
            if is_supported and evidence:
                evidence_spans.append(evidence)
        
        # Detect hallucinated spans
        hallucinated_spans = evaluator.detect_hallucinations(
            request.rag_answer, claims, support_map
        )
        
        # Compute faithfulness score
        faithfulness_score = scorer.compute_faithfulness_score(claims, support_map)
        
        # Determine flags
        flags = []
        if len(hallucinated_spans) > 0:
            flags.append("UNSUPPORTED")
        
        # Generate explanation
        explanation = explainer.generate_explanation(
            request.query,
            request.rag_answer,
            claims,
            support_map,
            hallucinated_spans
        )
        
        # Build TrustSignal response
        details = TrustSignalDetails(
            hallucinated_spans=hallucinated_spans,
            evidence_spans=evidence_spans,
            claim_scores=[
                {"claim": claim, "score": int(support_map.get(claim, False))}
                for claim in claims
            ],
            explanation=explanation
        )
        
        return TrustSignal(
            score=faithfulness_score,
            evidence=evidence_spans,
            flags=flags,
            applies_to=["factual"],  # Prometheus Inspector handles factual verification
            module="prometheus-inspector",
            details=details
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.post("/evaluate-legacy", response_model=EvaluationResponse, status_code=200)
async def evaluate_legacy(request: EvaluationRequest) -> EvaluationResponse:
    """
    Legacy evaluation endpoint (backward compatibility).
    
    Accepts EvaluationRequest with query, answer, and documents.
    Returns EvaluationResponse with score, label, and spans.
    
    Args:
        request: EvaluationRequest with query, answer, and documents
        
    Returns:
        EvaluationResponse with score, label, hallucinated_spans, evidence_spans, explanation
    """
    try:
        if not evaluator or not explainer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM not configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )
        
        # Preprocess inputs
        preprocessed_docs = deduplicate_documents(request.documents)
        if not preprocessed_docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid documents provided after preprocessing"
            )
        
        config = get_evaluation_config()
        
        # Extract claims from answer
        claims = evaluator.extract_claims(request.answer)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract claims from answer"
            )
        
        # Match each claim to evidence
        support_map = {}
        evidence_spans = []
        
        for claim in claims:
            is_supported, evidence = evaluator.match_claim_to_evidence(
                claim, preprocessed_docs
            )
            support_map[claim] = is_supported
            if is_supported and evidence:
                evidence_spans.append(evidence)
        
        # Detect hallucinated spans
        hallucinated_spans = evaluator.detect_hallucinations(
            request.answer, claims, support_map
        )
        
        # Compute faithfulness score
        faithfulness_score = scorer.compute_faithfulness_score(claims, support_map)
        
        # Determine label
        label = scorer.compute_label(
            faithfulness_score,
            threshold=config["faithfulness_threshold"]
        )
        
        # Generate explanation
        explanation = explainer.generate_explanation(
            request.query,
            request.answer,
            claims,
            support_map,
            hallucinated_spans
        )
        
        return EvaluationResponse(
            score=faithfulness_score,
            label=label,
            hallucinated_spans=hallucinated_spans,
            evidence_spans=evidence_spans,
            explanation=explanation
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.exception_handler(ValueError)
async def value_error_exception_handler(request, exc):
    """Handle Pydantic validation errors with 422 status."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    from src.config import get_api_config
    
    api_config = get_api_config()
    uvicorn.run(
        "src.main:app",
        host=api_config["host"],
        port=api_config["port"],
        reload=api_config["debug"]
    )
