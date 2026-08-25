import pytest
import pandas as pd
from delay_intelligence.evaluation.splitter import RollingOriginSplitter
from sklearn.model_selection import TimeSeriesSplit

@pytest.fixture
def dummy_data():
    dates = pd.date_range(start='2010-01-01', periods=3000, freq='D')
    df = pd.DataFrame({
        'T_pred': dates,
        'Delay_Days': [0] * 3000,
        'Delay_Flag': [0] * 3000
    })
    return df

def test_conformal_temporal_safety(dummy_data):
    # Ensure calibration strictly precedes validation
    splitter = RollingOriginSplitter()
    
    folds, holdout_idx, _ = splitter.split(dummy_data)
    
    tscv = TimeSeriesSplit(n_splits=2)
    
    for fold in folds:
        train_idx = fold['train']
        val_idx = fold['val']
        
        inner_train_idx, inner_calib_idx = list(tscv.split(train_idx))[-1]
        
        # Calibration must be strictly inside the original training set
        assert set(inner_calib_idx).issubset(set(train_idx))
        
        # Calibration must strictly precede validation
        max_calib_time = dummy_data.iloc[inner_calib_idx]['T_pred'].max()
        min_val_time = dummy_data.iloc[val_idx]['T_pred'].min()
        
        assert max_calib_time < min_val_time
        
        # Calibration must not touch final holdout
        assert len(set(inner_calib_idx).intersection(set(holdout_idx))) == 0

def test_no_final_holdout_access_in_uncertainty(dummy_data):
    splitter = RollingOriginSplitter()
    folds, holdout_idx, _ = splitter.split(dummy_data)
    
    for fold in folds:
        assert len(set(fold['train']).intersection(set(holdout_idx))) == 0
        assert len(set(fold['val']).intersection(set(holdout_idx))) == 0
