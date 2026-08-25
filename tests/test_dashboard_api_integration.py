def test_dashboard_api_integration():
    from delay_intelligence.dashboard.api_client import api_predict, api_explain, api_recommend
    
    features = {"Line Item Value": 1500, "Shipment Mode": "Air"}
    
    pred = api_predict(features)
    assert pred is not None
    assert 'probability_late' in pred
    
    expl = api_explain(features)
    assert expl is not None
    assert 'top_predictive_drivers' in expl
    
    rec = api_recommend(features)
    assert rec is not None
    assert 'recommendation' in rec
    assert 'human_approval_required' in rec
