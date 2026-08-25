import pytest

from delay_intelligence.evaluation.cross_domain import (
    CrossDomainEvaluator,
    ExternalValidationNotPerformed,
)


def test_cross_domain_does_not_fabricate_zero_shot_metrics():
    evaluator = CrossDomainEvaluator()
    with pytest.raises(ExternalValidationNotPerformed):
        evaluator.evaluate_zero_shot({}, {})


def test_cross_domain_protocol_is_explicitly_not_validated():
    evaluator = CrossDomainEvaluator()
    protocol = evaluator.protocol()
    assert protocol["evidence_label"] == "NOT VALIDATED"
    assert protocol["status"] == "protocol_only"
