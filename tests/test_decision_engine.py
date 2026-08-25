import pytest
from delay_intelligence.decision.engine import DecisionEngine

@pytest.fixture
def engine():
    return DecisionEngine()

def test_risk_tiers_applied(engine):
    assert engine._determine_risk_tier(0.1) == "LOW_RISK"
    assert engine._determine_risk_tier(0.9) == "CRITICAL"

def test_high_uncertainty_downgrades_to_human_review(engine):
    # CRITICAL probability but high uncertainty (width 20 > 14)
    d = engine.evaluate('test', 0.95, 10.0, (1.0, 21.0), 1000, 'Ocean', [], [])
    assert d['recommended_action'] == "HUMAN_REVIEW"
    assert d['high_uncertainty'] is True
