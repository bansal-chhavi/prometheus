import pytest
from prometheus.pipeline import PrometheusPipeline
from prometheus.config import PrometheusConfig
from prometheus.models import ClaimVerdict, NLILabel, Claim, Flag

def test_aggregator_all_entail():
    pipeline = PrometheusPipeline(PrometheusConfig())
    c = Claim(text="t")
    verdicts = [ClaimVerdict(claim=c, label=NLILabel.ENTAILMENT, confidence=1.0)] * 3
    res = pipeline._aggregate(verdicts, [], 0.0, "exp", "test")
    assert Flag.FAITHFUL in res.flags
    assert res.score == 1.0

def test_aggregator_contradiction():
    pipeline = PrometheusPipeline(PrometheusConfig())
    c = Claim(text="t")
    verdicts = [
        ClaimVerdict(claim=c, label=NLILabel.CONTRADICTION, confidence=1.0),
        ClaimVerdict(claim=c, label=NLILabel.CONTRADICTION, confidence=1.0),
        ClaimVerdict(claim=c, label=NLILabel.ENTAILMENT, confidence=1.0),
    ]
    res = pipeline._aggregate(verdicts, [], 0.0, "exp", "test")
    assert Flag.HALLUCINATION in res.flags
    assert res.details.faithfulness_score == 0.333

def test_aggregator_drift():
    pipeline = PrometheusPipeline(PrometheusConfig())
    c = Claim(text="t")
    verdicts = [ClaimVerdict(claim=c, label=NLILabel.ENTAILMENT, confidence=1.0)]
    res = pipeline._aggregate(verdicts, [], 0.8, "exp", "test")
    assert Flag.DRIFT in res.flags
