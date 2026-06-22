from __future__ import annotations
import logging
from ..llm_providers.base import BaseLLMProvider

logger = logging.getLogger("prometheus")

SYSTEM_PROMPT = """You are a relevance evaluator. Given a user QUERY and an AI ANSWER,
rate how well the answer addresses the query.

Score from 0.0 to 1.0:
- 1.0 = Answer directly and completely addresses the query
- 0.7 = Answer mostly addresses the query, minor tangents
- 0.4 = Answer partially relevant but has significant drift
- 0.1 = Answer is mostly off-topic
- 0.0 = Answer has nothing to do with the query

Respond ONLY with JSON:
{
  "relevance_score": 0.0 to 1.0,
  "explanation": "brief reason"
}"""

USER_TEMPLATE = "QUERY: {query}\n\nANSWER: {answer}"

async def detect_drift(llm: BaseLLMProvider, query: str, answer: str) -> tuple[float, str]:
    user_prompt = USER_TEMPLATE.format(query=query, answer=answer)
    try:
        data = await llm.complete_json(SYSTEM_PROMPT, user_prompt)
        rel_score = float(data.get("relevance_score", 0.5))
        explanation = data.get("explanation", "")
        drift_score = round(1.0 - rel_score, 3)
        return drift_score, explanation
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return 0.5, "Failed to detect drift due to error"
