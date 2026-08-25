import pytest
import pandas as pd
import numpy as np
import os
import yaml
from delay_intelligence.evaluation.splitter import RollingOriginSplitter

@pytest.fixture(scope="module")
def df_features():
    df = pd.read_parquet('artifacts/data/scms_modeling_features.parquet')
    return df

@pytest.fixture(scope="module")
def split_results(df_features):
    splitter = RollingOriginSplitter()
    folds, holdout_idx, manifest = splitter.split(df_features)
    return folds, holdout_idx, manifest, df_features, splitter

def test_train_strictly_precedes_eval(split_results):
    folds, holdout_idx, manifest, df, _ = split_results
    for fold in folds:
        train_df = df.loc[fold['train']]
        val_df = df.loc[fold['val']]
        if len(train_df) > 0 and len(val_df) > 0:
            assert train_df['T_pred'].max() < val_df['T_pred'].min()

def test_configured_gap(split_results):
    folds, holdout_idx, manifest, df, splitter = split_results
    for fold in folds:
        train_df = df.loc[fold['train']]
        val_df = df.loc[fold['val']]
        if len(train_df) > 0 and len(val_df) > 0:
            gap = (val_df['T_pred'].min() - train_df['T_pred'].max()).days
            assert gap >= splitter.gap_days - 1

def test_no_overlap(split_results):
    folds, holdout_idx, manifest, df, _ = split_results
    for fold in folds:
        train_idx = set(fold['train'])
        val_idx = set(fold['val'])
        assert len(train_idx.intersection(val_idx)) == 0

def test_final_holdout_isolation(split_results):
    folds, holdout_idx, manifest, df, _ = split_results
    holdout_set = set(holdout_idx)
    for fold in folds:
        assert len(set(fold['train']).intersection(holdout_set)) == 0
        assert len(set(fold['val']).intersection(holdout_set)) == 0
        
def test_deterministic_folds(df_features):
    splitter = RollingOriginSplitter()
    f1, h1, m1 = splitter.split(df_features)
    f2, h2, m2 = splitter.split(df_features)
    
    assert h1.equals(h2)
    pd.testing.assert_frame_equal(m1, m2)
    for fold1, fold2 in zip(f1, f2):
        assert fold1['train'].equals(fold2['train'])
        assert fold1['val'].equals(fold2['val'])

def test_insufficient_folds_raises_error(df_features):
    config = {
        'temporal_validation': {
            'n_folds': 100, 
            'gap_days': 90,
            'holdout_duration_days': 365,
            'val_duration_days': 180,
            'min_train_days': 730
        }
    }
    cfg_path = "scratch/bad_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)
        
    splitter = RollingOriginSplitter(cfg_path)
    with pytest.raises(ValueError, match="insufficient training days"):
        splitter.split(df_features)
        
    if os.path.exists(cfg_path):
        os.remove(cfg_path)

def test_temporal_ordering_ties():
    df = pd.DataFrame({
        'T_pred': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-01', '2021-01-01']),
        'Delay_Flag': [1, 0, 1, 0]
    })
    
    cfg_path = "scratch/tie_config.yaml"
    with open(cfg_path, 'w') as f:
        yaml.dump({
            'temporal_validation': {
                'n_folds': 1, 'gap_days': 0, 'holdout_duration_days': 10, 'val_duration_days': 10, 'min_train_days': 1
            }
        }, f)
    
    splitter = RollingOriginSplitter(cfg_path)
    folds, holdout, _ = splitter.split(df)
    train_idx = folds[0]['train']
    assert len(train_idx) == 3
    
    if os.path.exists(cfg_path):
        os.remove(cfg_path)
