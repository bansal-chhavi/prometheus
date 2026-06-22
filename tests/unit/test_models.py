from prometheus.models import EvalRequest, Flag, NLILabel, EvalResult, HallucinatedSpan, ClaimVerdict, Claim

def test_eval_request():
    req = EvalRequest(query="Q", rag_answer="A", sources=["S1"])
    assert req.query == "Q"

def test_enums():
    assert Flag.FAITHFUL == "FAITHFUL"
    assert NLILabel.ENTAILMENT == "entailment"

def test_eval_result_json():
    res = EvalResult(score=1.0)
    assert "score" in res.model_dump_json()

def test_hallucinated_span_defaults():
    span = HallucinatedSpan(text="test")
    assert span.start == -1
    assert span.end == -1
    assert span.reason == ""

def test_claim_verdict():
    claim = Claim(text="t", source_span="s")
    v = ClaimVerdict(claim=claim, label=NLILabel.ENTAILMENT, confidence=1.0)
    assert v.confidence == 1.0
