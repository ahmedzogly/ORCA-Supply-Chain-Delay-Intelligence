import pytest
from delay_intelligence.decision.engine import DecisionEngine

def test_decision_traceability_schema():
    engine = DecisionEngine()
    d = engine.evaluate('test_id', 0.8, 5.0, (2.0, 8.0), 500, 'Air', ['Vendor'], [])
    
    assert 'risk_probability' in d
    assert 'severity_p50' in d
    assert 'decision_reason' in d
    assert 'expected_impact' in d
    assert 'recommended_action' in d
