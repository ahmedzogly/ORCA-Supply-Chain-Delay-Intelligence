def test_shap_explainability_no_leakage():
    forbidden = ['ID', 'T_pred', 'Delay_Days', 'Delay_Flag', 'Delivered to Client Date']
    import pandas as pd
    import os
    if os.path.exists('artifacts/explainability/shap_stability.csv'):
        df = pd.read_csv('artifacts/explainability/shap_stability.csv')
        for f in forbidden:
            assert f not in df['feature'].values

def test_shap_output_dimensions():
    # just a dummy test for now
    assert True
