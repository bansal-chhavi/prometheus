from __future__ import annotations
from ..llm_providers.base import BaseLLMProvider
from ..models import Claim

SYSTEM_PROMPT = """You are a claim extraction engine. Given a question and an AI-generated answer,
break the answer into individual atomic factual claims.

Rules:
- Each claim must be a single, independently verifiable statement.
- Preserve the EXACT wording from the answer — do not paraphrase.
- Include numbers, dates, names, percentages exactly as they appear.
- Skip filler phrases like "Sure, here is..." or "Based on the context...".
- Return between 1-20 claims.

Respond ONLY with a JSON object in this format:
{
  "claims": [
    {"text": "exact claim text from answer", "source_span": "the phrase in the answer this came from"}
  ]
}"""

USER_TEMPLATE = "Question: {query}\n\nAnswer to decompose:\n{answer}"

async def extract_claims(llm: BaseLLMProvider, query: str, answer: str) -> list[Claim]:
    user_prompt = USER_TEMPLATE.format(query=query, answer=answer)
    try:
        data = await llm.complete_json(SYSTEM_PROMPT, user_prompt)
    except Exception:
        return []
    
    claims_data = data.get("claims", [])
    if not isinstance(claims_data, list):
        return []

    claims = []
    answer_lower = answer.lower()
    for c_dict in claims_data:
        text = c_dict.get("text", "")
        source_span = c_dict.get("source_span", "")
        if not text or not source_span:
            continue
            
        start_idx = answer_lower.find(source_span.lower())
        end_idx = start_idx + len(source_span) if start_idx != -1 else -1
        
        claims.append(Claim(
            text=text,
            source_span=source_span,
            start=start_idx,
            end=end_idx
        ))
        
    return claims
