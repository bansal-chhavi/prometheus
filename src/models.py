"""
Pydantic models for KnowledgeOps request/response validation.
Implements TrustSignal contract for Prometheus Inspector module.
"""
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Source document chunk from KnowledgeTask."""
    text: str = Field(..., description="Document chunk text")
    doc_id: str = Field(..., description="Document identifier")
    chunk_id: int = Field(..., description="Chunk sequence number")


class KnowledgeTask(BaseModel):
    """
    Shared input contract for all KnowledgeOps modules.
    
    This is the standard request format received from Trust Router.
    """
    query: str = Field(..., min_length=1, description="User query")
    rag_answer: str = Field(..., min_length=1, description="RAG-generated answer")
    sources: list[SourceDocument] = Field(..., min_items=1, description="Source documents")
    task_type: Literal["factual", "synthesis", "expert", "regulatory", "exploratory"] = Field(
        ..., description="Task type classification"
    )


class TrustSignalDetails(BaseModel):
    """
    Prometheus-specific details within TrustSignal.
    """
    hallucinated_spans: list[str] = Field(default_factory=list, description="Unsupported text spans")
    evidence_spans: list[str] = Field(default_factory=list, description="Supporting text spans")
    claim_scores: list[dict] = Field(default_factory=list, description="Per-claim scoring details")
    explanation: str = Field(default="", description="Detailed explanation")


class TrustSignal(BaseModel):
    """
    Shared output contract for all KnowledgeOps modules.
    
    Core fields are locked. details{} is module-specific.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Trust score 0.0-1.0")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence")
    flags: list[str] = Field(default_factory=list, description="Risk flags (UNSUPPORTED/CONTRADICTED)")
    applies_to: list[str] = Field(
        default_factory=lambda: ["factual"],
        description="Task types this signal applies to"
    )
    module: str = Field(default="prometheus-inspector", description="Module identifier")
    details: TrustSignalDetails = Field(default_factory=TrustSignalDetails, description="Module-specific details")


# Legacy format support (for backward compatibility with evaluation files)
class EvaluationRequest(BaseModel):
    """
    Backward-compatible request schema.
    
    Attributes:
        query: The original user question/query
        answer: The RAG-generated answer to evaluate
        documents: List of source document chunks providing context
    """
    query: str = Field(..., min_length=1, description="The original query")
    answer: str = Field(..., min_length=1, description="The RAG-generated answer")
    documents: list[str] = Field(..., min_items=1, description="Source document chunks")


class EvaluationResponse(BaseModel):
    """
    Backward-compatible response schema for evaluation endpoint.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Faithfulness score 0.0-1.0")
    label: Literal["FAITHFUL", "HALLUCINATION"] = Field(..., description="Classification")
    hallucinated_spans: list[str] = Field(default_factory=list, description="Unsupported spans")
    evidence_spans: list[str] = Field(default_factory=list, description="Supporting spans")
    explanation: str = Field(..., description="Natural language explanation")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(default="ok", description="Service status")
