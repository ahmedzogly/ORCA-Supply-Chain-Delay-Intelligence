def test_dashboard_data_contract_schemas():
    # Ensure all required display keys are returned by the API client
    from delay_intelligence.dashboard.api_client import api_predict, api_recommend
    features = {"Line Item Value": 1500}
    
    pred = api_predict(features)
    assert set(['probability_late', 'risk_tier', 'severity_p50', 'severity_interval_90', 'model_version', 'prediction_contract_version']).issubset(pred.keys())
    
    rec = api_recommend(features)
    assert set(['recommendation', 'decision_reason', 'expected_impact_type', 'robustness', 'human_approval_required']).issubset(rec.keys())
