import pytest
from delay_intelligence.decision.engine import DecisionEngine

def test_human_review_flag_safety():
    engine = DecisionEngine()
    d = engine.evaluate('test_id', 0.9, 10.0, (8.0, 12.0), 5000, 'Ocean', [], [])
    
    # Action might be EXPEDITE or HUMAN_REVIEW. Both require human approval.
    assert d['human_approval_required'] is True
    
    # LOW RISK -> NO_ACTION -> auto eligible
    d_low = engine.evaluate('test_id', 0.1, 0.0, (0.0, 1.0), 100, 'Ocean', [], [])
    assert d_low['recommended_action'] == 'NO_ACTION'
    assert d_low['human_approval_required'] is False
