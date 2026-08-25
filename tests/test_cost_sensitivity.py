import pytest
from delay_intelligence.decision.engine import DecisionEngine

def test_cost_sensitivity_classifications():
    engine = DecisionEngine()
    d = engine.evaluate_sensitivity('test', 0.8, 5.0, (4.0, 6.0), 10000, 'Air', ['Shipment Mode'], ['Shipment Mode -> Delay_Days'])
    assert 'robustness_class' in d
    assert d['robustness_class'] in ['ROBUST', 'SENSITIVE', 'UNSUPPORTED']
