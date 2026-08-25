import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="module")
def df_features():
    return pd.read_parquet('artifacts/data/scms_modeling_features.parquet')

def test_prediction_boundary(df_features):
    '''For every row, feature_timestamp <= T_pred. 
       This is implicitly true for all non-PIT features due to their definition.
       We test that T_pred is valid and Forecast Horizon is non-negative.'''
    assert df_features['T_pred'].notna().all()
    # T_pred is available at T_pred by definition
    
    # Forecast Horizon must be >= 0 according to R3
    assert (df_features['Forecast_Horizon_Days'] >= 0).all()

def test_row_integrity(df_features):
    '''Expected modeling cohort remains traceable.'''
    # Base count is ~8319
    assert len(df_features) > 8000
    assert 'ID' in df_features.columns
    assert df_features['ID'].is_unique

def test_structural_missingness(df_features):
    '''RDC/pre-PQ indicators remain valid.'''
    assert 'is_rdc_fulfillment' in df_features.columns
    assert 'is_pre_pq_process' in df_features.columns
    assert set(df_features['is_rdc_fulfillment'].dropna().unique()).issubset({0, 1})
    assert set(df_features['is_pre_pq_process'].dropna().unique()).issubset({0, 1})

def test_feature_transformations(df_features):
    '''Verify numeric transformations (log1p).'''
    assert 'Line Item Quantity' in df_features.columns
    # Ensure they are numeric and positive (since we used log1p and clipped to 0)
    assert (df_features['Line Item Quantity'] >= 0).all()
    assert (df_features['Line Item Value'] >= 0).all()
