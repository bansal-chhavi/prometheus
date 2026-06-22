from __future__ import annotations
import time
import logging
from .config import PrometheusConfig
from .models import EvalResult, EvalDetails, Flag, NLILabel, ClaimVerdict, HallucinatedSpan
from .llm_providers import create_provider
from .detectors import extract_claims, HuggingFaceNLI, score_claims, extract_hallucinated_spans, detect_drift
from .observability import trace_span, emit_counter, emit_histogram, emit_gauge
from .logging_config import new_trace_id

logger = logging.getLogger("prometheus")

class PrometheusPipeline:
    def __init__(self, config: PrometheusConfig):
        self.config = config
        self._llm = None
        self._hf = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_provider(self.config)
        return self._llm

    @property
    def hf(self):
        if self._hf is None:
            token = self.config.get_hf_token()
            if token:
                self._hf = HuggingFaceNLI(api_token=token, model_name=self.config.hf_nli_model)
        return self._hf

    async def evaluate(self, query: str, rag_answer: str, sources: list[str]) -> EvalResult:
        trace_id = new_trace_id()
        logger.info("pipeline.evaluate called", extra={"query_len": len(query), "source_count": len(sources)})
        
        async with trace_span("prometheus.evaluate", query_length=len(query), source_count=len(sources)):
            start = time.perf_counter()

            # Step 1
            async with trace_span("prometheus.claim_extraction", llm_provider=self.config.llm_provider.value):
                try:
                    claims = await extract_claims(self.llm, query, rag_answer)
                except Exception as e:
                    logger.error(f"LLM call failed: provider={self.config.llm_provider.value}, error={str(e)}")
                    claims = []
            
            logger.info(f"claims extracted: {len(claims)}")
            
            if not claims:
                result = EvalResult(score=1.0, flags=[Flag.FAITHFUL], explanation="No factual claims found.")
                return result

            # Step 2
            async with trace_span("prometheus.nli_scoring"):
                try:
                    verdicts, nli_backend = await score_claims(claims, sources, self.llm, self.hf)
                except Exception as e:
                    logger.error(f"LLM NLI scoring failed: provider={self.config.llm_provider.value}, error={str(e)}")
                    verdicts = []
                    nli_backend = "error"
            
            supported = sum(1 for v in verdicts if v.label == NLILabel.ENTAILMENT)
            contradicted = sum(1 for v in verdicts if v.label == NLILabel.CONTRADICTION)
            logger.info(f"NLI scoring complete: backend={nli_backend}, supported={supported}, contradicted={contradicted}")

            # Step 3
            async with trace_span("prometheus.span_extraction"):
                hallucinated_spans = extract_hallucinated_spans(rag_answer, verdicts)

            # Step 4
            async with trace_span("prometheus.drift_detection"):
                try:
                    drift_score, drift_explanation = await detect_drift(self.llm, query, rag_answer)
                except Exception as e:
                    logger.error(f"LLM call failed: provider={self.config.llm_provider.value}, error={str(e)}")
                    drift_score = 0.5
                    drift_explanation = "Error detecting drift."

            # Step 5
            async with trace_span("prometheus.aggregation"):
                result = self._aggregate(verdicts, hallucinated_spans, drift_score, drift_explanation, nli_backend)
            
            logger.info(f"evaluation complete: score={result.score}, flags={[f.value for f in result.flags]}")

            # Metrics
            elapsed_ms = (time.perf_counter() - start) * 1000
            emit_counter("evaluation.count")
            emit_histogram("evaluation.latency", elapsed_ms)
            emit_gauge("faithfulness.score", result.details.faithfulness_score)
            emit_gauge("drift.score", result.details.drift_score)
            if Flag.HALLUCINATION in result.flags:
                emit_counter("hallucination.detected")
            emit_counter("nli.backend", tags=[f"backend:{nli_backend}"])
            emit_histogram("claims.count", len(claims))

            return result

    def _aggregate(
        self, 
        verdicts: list[ClaimVerdict], 
        hallucinated_spans: list[HallucinatedSpan], 
        drift_score: float, 
        drift_explanation: str,
        nli_backend: str
    ) -> EvalResult:
        total = len(verdicts)
        supported = sum(1 for v in verdicts if v.label == NLILabel.ENTAILMENT)
        contradicted = sum(1 for v in verdicts if v.label == NLILabel.CONTRADICTION)
        unsupported = sum(1 for v in verdicts if v.label == NLILabel.NEUTRAL)

        faithfulness_score = round(supported / total, 3) if total > 0 else 1.0
        overall_score = round(faithfulness_score * 0.8 + (1.0 - drift_score) * 0.2, 3)

        # Flags
        flags = []
        if contradicted > 0: flags.append(Flag.HALLUCINATION)
        if unsupported > 0 and unsupported / total > self.config.unsupported_threshold: flags.append(Flag.UNSUPPORTED)
        if drift_score > self.config.drift_threshold: flags.append(Flag.DRIFT)
        if not flags: flags.append(Flag.FAITHFUL)

        # Explanation always starts with [NLI: backend_name]
        explanation = f"[NLI: {nli_backend}] Faithfulness: {faithfulness_score}. Drift: {drift_score}. {drift_explanation}"

        details = EvalDetails(
            faithfulness_score=faithfulness_score,
            hallucinated_spans=hallucinated_spans,
            drift_score=drift_score,
            claim_verdicts=verdicts
        )

        return EvalResult(
            score=overall_score,
            flags=flags,
            explanation=explanation,
            details=details
        )
