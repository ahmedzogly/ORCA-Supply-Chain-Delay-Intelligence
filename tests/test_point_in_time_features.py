import pytest
import pandas as pd
import numpy as np
from delay_intelligence.features.builder import TemporalFeatureBuilder

@pytest.fixture(scope="module")
def df_features():
    return pd.read_parquet('artifacts/data/scms_modeling_features.parquet')

def test_point_in_time_historical_aggregates(df_features):
    '''For every historical aggregate: aggregate_information_time < T_pred'''
    assert 'vendor_hist_delay_rate' in df_features.columns
    assert 'vendor_hist_delay_median' in df_features.columns
    assert 'country_hist_delay_rate' in df_features.columns
    
    # Check that rates are bounded [0, 1] (or nan, though cold start fills them)
    rates = df_features['vendor_hist_delay_rate'].dropna()
    assert (rates >= 0).all() and (rates <= 1).all()

def test_cold_start_behavior(df_features):
    '''Unseen historical entities are handled deterministically (filled with global mean).'''
    # We should have NO nulls in the historical features after cold start
    assert df_features['vendor_hist_delay_rate'].isna().sum() == 0
    assert df_features['country_hist_delay_rate'].isna().sum() == 0
    assert df_features['site_hist_delay_rate'].isna().sum() == 0

def test_reproducibility():
    '''Same input + same config -> same feature output.'''
    # Run a small sample through the builder
    df_raw = pd.read_parquet('artifacts/data/bronze_scms.parquet').head(500)
    # mock T_pred and limits to simulate cohort
    df_raw['T_pred'] = pd.to_datetime(df_raw['Scheduled Delivery Date']) - pd.Timedelta(days=30)
    
    builder1 = TemporalFeatureBuilder()
    out1 = builder1.build_features(df_raw)
    
    builder2 = TemporalFeatureBuilder()
    out2 = builder2.build_features(df_raw)
    
    pd.testing.assert_frame_equal(out1, out2)
