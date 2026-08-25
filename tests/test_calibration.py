import pytest
import pandas as pd
import numpy as np

def test_calibration_bounds():
    # Probabilities must be within [0, 1]
    # And brier score must be calculated
    metrics = pd.read_csv('artifacts/evaluation/stage5_metrics.csv')
    c_metrics = metrics[metrics['task'] == 'classification']
    
    assert 'brier_score' in c_metrics.columns
    assert c_metrics['brier_score'].notna().all()
    assert (c_metrics['brier_score'] >= 0).all()
    assert (c_metrics['brier_score'] <= 1).all()
