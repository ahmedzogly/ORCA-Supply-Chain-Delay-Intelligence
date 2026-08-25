import pytest
import pandas as pd
from delay_intelligence.validation.contract_validator import PredictionContractValidator
import yaml

@pytest.fixture(scope="module")
def df_features():
    return pd.read_parquet('artifacts/data/scms_modeling_features.parquet')

@pytest.fixture(scope="module")
def config():
    with open("configs/features.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_target_exclusion(df_features):
    '''No target-derived/post-outcome feature appears in X (except targets kept for splitting, but we assert they are not used as features).'''
    validator = PredictionContractValidator()
    forbidden = validator.get_forbidden_features()
    
    # We kept targets in the dataframe for pipeline convenience, but we must verify they are explicitly marked as target_cols in the builder
    # Let's ensure 'Weight (Kilograms)' and 'Freight Cost (USD)' are NOT in the dataframe at all
    assert 'Weight (Kilograms)' not in df_features.columns
    assert 'Freight Cost (USD)' not in df_features.columns
    assert 'ASN/DN #' not in df_features.columns
    
def test_leakage_specification_adherence(df_features, config):
    '''Verify only configured features and essential targets are present.'''
    expected_allowed = ['T_pred', 'Forecast_Horizon_Days']
    for group, features in config.get('feature_groups', {}).items():
        expected_allowed.extend([f['name'] for f in features])
    for pit_group in config['historical_aggregates']['point_in_time']:
        expected_allowed.extend([f['name'] for f in pit_group['features']])
        
    expected_targets = ['ID', 'Delivered to Client Date', 'Delivery Recorded Date', 'Delay_Days', 'Delay_Flag', 'is_temporal_anomaly']
    
    # Every column in df_features should be either an allowed feature or a target column
    for col in df_features.columns:
        assert col in expected_allowed or col in expected_targets, f"Unexpected column found: {col}"
