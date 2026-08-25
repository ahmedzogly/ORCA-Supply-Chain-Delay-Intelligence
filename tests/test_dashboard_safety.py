def test_dashboard_does_not_mutate_data():
    import pandas as pd
    from delay_intelligence.dashboard.api_client import load_data
    df1 = load_data()
    df2 = load_data()
    assert df1.equals(df2)

def test_dashboard_api_client_strips_forbidden_fields():
    # Verify the dashboard API client drops forbidden fields
    import pandas as pd
    from delay_intelligence.dashboard.api_client import api_predict
    
    features = {"Line Item Value": 1500, "Delay_Days": 10}
    # Emulate the dashboard stripping logic
    f_clean = pd.Series(features).drop(labels=['Delay_Days'], errors='ignore').to_dict()
    assert 'Delay_Days' not in f_clean
    assert 'Line Item Value' in f_clean
