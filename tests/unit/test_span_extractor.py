import pytest
from prometheus.detectors.span_extractor import extract_hallucinated_spans, _merge_spans
from prometheus.models import ClaimVerdict, Claim, NLILabel

def test_extract_spans_exact():
    claim = Claim(text="c", source_span="in 2021", start=0, end=0)
    verdict = ClaimVerdict(claim=claim, label=NLILabel.CONTRADICTION, confidence=1.0)
    spans = extract_hallucinated_spans("launched in 2021 yes", [verdict])
    assert len(spans) == 1
    assert spans[0].start == 9
    assert spans[0].end == 16

def test_extract_spans_case_insensitive():
    claim = Claim(text="c", source_span="IN 2021", start=0, end=0)
    verdict = ClaimVerdict(claim=claim, label=NLILabel.CONTRADICTION, confidence=1.0)
    spans = extract_hallucinated_spans("launched in 2021 yes", [verdict])
    assert len(spans) == 1

def test_extract_spans_key_tokens():
    claim = Claim(text="c", source_span="reducing time by 50%", start=0, end=0)
    verdict = ClaimVerdict(claim=claim, label=NLILabel.CONTRADICTION, confidence=1.0)
    spans = extract_hallucinated_spans("cut by 50% roughly", [verdict])
    assert len(spans) == 1
    assert spans[0].text == "50%"

def test_merge_spans():
    spans = [
        (0, 10, "text1", "r1"),
        (5, 15, "text2", "r2")
    ]
    merged = _merge_spans(spans)
    assert len(merged) == 1
    assert merged[0].start == 0
    assert merged[0].end == 15
