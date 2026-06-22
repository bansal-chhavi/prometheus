from __future__ import annotations
import re
from ..models import ClaimVerdict, NLILabel, HallucinatedSpan

def extract_hallucinated_spans(answer: str, verdicts: list[ClaimVerdict]) -> list[HallucinatedSpan]:
    spans = []
    
    for v in verdicts:
        if v.label == NLILabel.ENTAILMENT:
            continue
            
        span_text = v.claim.source_span
        if not span_text:
            continue
            
        # 1. Exact match
        idx = answer.find(span_text)
        if idx != -1:
            spans.append((idx, idx + len(span_text), span_text, v.explanation))
            continue
            
        # 2. Case-insensitive
        match = re.search(re.escape(span_text), answer, re.IGNORECASE)
        if match:
            spans.append((match.start(), match.end(), match.group(), v.explanation))
            continue
            
        # 3. Key token extraction
        patterns = [r'\d+%', r'\d{4}', r'\$[\d,.]+', r'\d[\d,.]+']
        for p in patterns:
            tokens = re.findall(p, span_text)
            for t in tokens:
                t_match = re.search(re.escape(t), answer)
                if t_match:
                    spans.append((t_match.start(), t_match.end(), t, v.explanation))

    return _merge_spans(spans)

def _merge_spans(spans: list[tuple[int, int, str, str]]) -> list[HallucinatedSpan]:
    if not spans:
        return []
        
    spans.sort(key=lambda x: x[0])
    merged = []
    
    current_start, current_end, current_text, current_reason = spans[0]
    
    for start, end, text, reason in spans[1:]:
        if start <= current_end + 1:
            # Overlapping or adjacent
            current_end = max(current_end, end)
            # Combine reasons if different
            if reason and reason not in current_reason:
                current_reason += f" | {reason}"
        else:
            merged.append(HallucinatedSpan(
                text=current_text, # text might not be perfectly accurate after merge, but we just use start/end in practice for highlighting usually
                start=current_start,
                end=current_end,
                reason=current_reason
            ))
            current_start, current_end, current_text, current_reason = start, end, text, reason
            
    merged.append(HallucinatedSpan(
        text=current_text,
        start=current_start,
        end=current_end,
        reason=current_reason
    ))
    
    return merged
