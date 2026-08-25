"""
Test Suite: Chronological Drift Detection Temporal Safety & Isolation.
Verifies:
1. Strict isolation of the 365-day Final Holdout (2014-08-24 to 2015-08-24).
2. Chronological ordering of reference and detection windows (W_ref <= W_det).
3. Configured 90-day embargo gap adherence between training and validation.
4. Immobility and immutability of frozen development thresholds.
"""

import os
import yaml
import pytest
import numpy as np
import pandas as pd

from delay_intelligence.evaluation.splitter import RollingOriginSplitter
from delay_intelligence.drift.detector import ChronologicalDriftDetector
from delay_intelligence.drift.policy import DriftTriggerPolicy
from delay_intelligence.drift.schemas import DriftStatus


@pytest.fixture
def modeling_df():
    path = "artifacts/data/scms_modeling_features.parquet"
    if not os.path.exists(path):
        pytest.skip(f"Artifact {path} not found")
    df = pd.read_parquet(path)
    df['T_pred'] = pd.to_datetime(df['T_pred'])
    return df


def test_holdout_isolation_in_threshold_calibration(modeling_df):
    """
    Asserts that the 365-day Final Holdout (2014-08-24 to 2015-08-24)
    is strictly isolated from all 5 Development CV folds.
    """
    splitter = RollingOriginSplitter()
    folds, holdout_idx, manifest_df = splitter.split(modeling_df)
    
    assert len(holdout_idx) == 1013, f"Expected 1013 holdout rows, got {len(holdout_idx)}"
    
    holdout_data = modeling_df.loc[holdout_idx]
    holdout_min_date = holdout_data['T_pred'].min()
    assert holdout_min_date >= pd.to_datetime('2014-08-24'), f"Holdout starts before 2014-08-24: {holdout_min_date}"
    
    # Check all CV folds
    for fold in folds:
        train_idx = set(fold['train'])
        val_idx = set(fold['val'])
        holdout_set = set(holdout_idx)
        
        # Zero intersection with holdout
        assert len(train_idx.intersection(holdout_set)) == 0, f"Fold {fold['fold_id']} train contains holdout data!"
        assert len(val_idx.intersection(holdout_set)) == 0, f"Fold {fold['fold_id']} val contains holdout data!"
        
        train_max_date = modeling_df.loc[list(train_idx), 'T_pred'].max()
        val_max_date = modeling_df.loc[list(val_idx), 'T_pred'].max()
        
        assert train_max_date < pd.to_datetime('2014-08-24'), f"Fold {fold['fold_id']} train touches holdout period"
        assert val_max_date <= pd.to_datetime('2014-08-24'), f"Fold {fold['fold_id']} val touches holdout period"


def test_strict_chronological_ordering_of_windows(modeling_df):
    """
    Asserts that for every CV fold, W_train strictly precedes W_val with at least gap_days.
    """
    splitter = RollingOriginSplitter()
    folds, _, _ = splitter.split(modeling_df)
    
    for fold in folds:
        train_df = modeling_df.loc[fold['train']]
        val_df = modeling_df.loc[fold['val']]
        
        train_end = train_df['T_pred'].max()
        val_start = val_df['T_pred'].min()
        
        # Strict inequality
        assert train_end < val_start, f"Fold {fold['fold_id']} train_end ({train_end}) >= val_start ({val_start})"
        
        # Gap verification (>= 90 calendar days)
        gap_days = (val_start - train_end).days
        assert gap_days >= 90, f"Fold {fold['fold_id']} gap is {gap_days} days (< 90 days)"


def test_label_lag_embargo_compliance(modeling_df):
    """
    Verifies that target delivery outcomes occur on or after T_pred
    and that prediction timestamp precedes delivery date.
    """
    if 'Delivered to Client Date' in modeling_df.columns:
        deliv = pd.to_datetime(modeling_df['Delivered to Client Date'])
        t_pred = modeling_df['T_pred']
        
        # Delivery must be on or after prediction timestamp
        assert (deliv >= t_pred).all(), "Found shipments delivered before prediction timestamp!"


def test_dev_frozen_threshold_immutability():
    """
    Asserts that configs/drift.yaml contains immutable development thresholds.
    """
    with open("configs/drift.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    thresh = cfg['thresholds']
    assert thresh['feature_drift']['psi_warning'] == 0.10
    assert thresh['feature_drift']['psi_critical'] == 0.25
    assert thresh['feature_drift']['wasserstein_norm_warning'] == 0.15
    assert thresh['feature_drift']['wasserstein_norm_critical'] == 0.30
    
    assert thresh['prediction_drift']['prob_psi_warning'] == 0.10
    assert thresh['prediction_drift']['prob_psi_critical'] == 0.20
    
    assert thresh['uncertainty_drift']['nominal_coverage'] == 0.90
    assert thresh['uncertainty_drift']['coverage_error_critical'] == 0.08
    assert thresh['uncertainty_drift']['nonconformity_wasserstein_critical_days'] == 3.0
    
    policy_cfg = cfg['policy']
    assert policy_cfg['n_min'] == 50
    assert policy_cfg['t_max_days'] == 180
    assert policy_cfg['t_cooldown_days'] == 30
    assert policy_cfg['k_persistence'] == 2


def test_drift_runner_artifacts_temporal_isolation():
    """
    Verifies that all generated development drift artifacts in artifacts/drift/
    only cover detection periods strictly ending on or before 2014-08-24.
    """
    metrics_path = "artifacts/drift/drift_metrics.csv"
    if not os.path.exists(metrics_path):
        pytest.skip(f"Artifact {metrics_path} not found")
        
    df_metrics = pd.read_csv(metrics_path)
    assert len(df_metrics) == 5, f"Expected 5 CV folds in development metrics, got {len(df_metrics)}"
    
    for idx, row in df_metrics.iterrows():
        det_end = pd.to_datetime(row['det_end'])
        assert det_end <= pd.to_datetime('2014-08-24'), f"Fold {row['fold_id']} det_end ({det_end}) exceeds Development cutoff 2014-08-24"
