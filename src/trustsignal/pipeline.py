from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class TrustSignalConfig:
    hf_model: str = "microsoft/deberta-v3-large"
    fallback_on_429: bool = True
    nli_threshold: float = 0.6
    simulate_hf_failure: bool = False


@dataclass
class ClaimVerdict:
    claim_id: int
    source_index: int
    label: str
    confidence: float


@dataclass
class Span:
    claim_id: int
    source_index: int
    start: int
    end: int
    text: str


@dataclass
class EvalResult:
    score: float
    flags: List[str]
    hallu_spans: List[Dict[str, Any]]
    explanation: str
    details: Dict[str, Any]


def _decompose_claims(answer: str) -> List[Dict[str, Any]]:
    # Very small deterministic decomposition: split on sentences
    parts = [p.strip() for p in answer.split(".") if p.strip()]
    return [{"id": i, "text": t} for i, t in enumerate(parts)]


def _nli_with_hf(claim: str, source: str, cfg: TrustSignalConfig) -> ClaimVerdict:
    # Mock HF NLI: deterministic confidence based on length
    if cfg.simulate_hf_failure:
        raise RuntimeError("HF_RATE_LIMIT")
    confidence = min(0.99, max(0.01, len(claim) / (len(source) + 1)))
    label = "ENTAILS" if confidence >= cfg.nli_threshold else "NEUTRAL"
    return ClaimVerdict(claim_id=-1, source_index=-1, label=label, confidence=confidence)


def _nli_with_groq(claim: str, source: str, cfg: TrustSignalConfig) -> ClaimVerdict:
    # Simpler fallback scoring
    confidence = min(0.9, max(0.05, (len(claim.split()) / (len(source.split()) + 1)) * 0.8))
    label = "ENTAILS" if confidence >= (cfg.nli_threshold * 0.9) else "NEUTRAL"
    return ClaimVerdict(claim_id=-1, source_index=-1, label=label, confidence=confidence)


def _extract_span(claim: str, source: str, claim_id: int, source_index: int) -> Optional[Span]:
    idx = source.find(claim)
    if idx >= 0:
        return Span(claim_id=claim_id, source_index=source_index, start=idx, end=idx + len(claim), text=claim)
    # not found — no span
    return None


def _detect_drift(query: str, answer: str) -> Dict[str, Any]:
    qtokens = set(query.lower().split())
    atokens = set(answer.lower().split())
    overlap = qtokens.intersection(atokens)
    score = len(overlap) / max(1, len(qtokens))
    flags = []
    if score < 0.3:
        flags.append("DRIFT")
    return {"drift_score": score, "drift_flags": flags}


def _aggregate(claim_verdicts: List[ClaimVerdict], drift: Dict[str, Any]) -> EvalResult:
    if not claim_verdicts:
        return EvalResult(score=0.0, flags=["NO_CLAIMS"], hallu_spans=[], explanation="No claims extracted", details={})
    # compute average confidence where label == ENTAILS
    ent_conf = [v.confidence for v in claim_verdicts if v.label == "ENTAILS"]
    overall = float(sum(ent_conf) / len(ent_conf)) if ent_conf else 0.0
    flags = []
    if overall < 0.5:
        flags.append("LOW_CONFIDENCE")
    flags.extend(drift.get("drift_flags", []))
    explanation = f"Aggregate entailment confidence={overall:.2f}; drift={drift.get('drift_score'):.2f}"
    details = {
        "faithfulness_score": overall,
        "drift_score": drift.get("drift_score", 0.0),
        "claim_verdicts": [asdict(v) for v in claim_verdicts],
    }
    # hallu_spans intentionally empty here — populated by caller if spans found
    return EvalResult(score=overall, flags=flags, hallu_spans=[], explanation=explanation, details=details)


def evaluate(query: str, rag_answer: str, sources: List[Dict[str, Any]], config: Optional[TrustSignalConfig] = None) -> EvalResult:
    """Simple, deterministic skeleton of pipeline.evaluate().

    This is a minimal, local implementation for prototyping and tests.
    It does NOT call external APIs.
    """
    cfg = config or TrustSignalConfig()
    claims = _decompose_claims(rag_answer)

    claim_verdicts: List[ClaimVerdict] = []
    spans: List[Dict[str, Any]] = []

    for c in claims:
        claim_id = c["id"]
        text = c["text"]
        for i, src in enumerate(sources):
            src_text = src.get("text", "")
            # try HF NLI
            try:
                v = _nli_with_hf(text, src_text, cfg)
            except Exception:
                if cfg.fallback_on_429:
                    v = _nli_with_groq(text, src_text, cfg)
                else:
                    raise
            v.claim_id = claim_id
            v.source_index = i
            claim_verdicts.append(v)

            sp = _extract_span(text, src_text, claim_id, i)
            if sp:
                spans.append(asdict(sp))

    drift = _detect_drift(query, rag_answer)

    res = _aggregate(claim_verdicts, drift)
    res.hallu_spans = spans
    return res


if __name__ == "__main__":
    # quick manual smoke run
    cfg = TrustSignalConfig()
    r = evaluate("Who won 2020 election?", "Candidate A won the 2020 election.", [{"id": "s1", "text": "Candidate A won the 2020 election."}], cfg)
    print(r)
