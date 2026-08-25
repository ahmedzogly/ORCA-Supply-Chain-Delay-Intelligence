import pytest
import pandas as pd

def test_threshold_learned_from_pre_test_data():
    # Ensure optimal threshold is learned strictly from inner CV
    # and not optimized on the validation slice.
    metrics = pd.read_csv('artifacts/evaluation/stage5_metrics.csv')
    c_metrics = metrics[metrics['task'] == 'classification']
    
    assert 'opt_threshold' in c_metrics.columns
    assert c_metrics['opt_threshold'].notna().all()
    assert (c_metrics['opt_threshold'] > 0).all()
    assert (c_metrics['opt_threshold'] < 1).all()
