import pytest
import numpy as np
from delay_intelligence.uncertainty.conformal import ConformalQuantileCalibrator

def test_conformal_calibrator_adjustment():
    calibrator = ConformalQuantileCalibrator(alpha=0.1)
    
    q_low_calib = np.array([10, 20, 30, 40, 50])
    q_high_calib = np.array([15, 25, 35, 45, 55])
    # Truth is outside the interval by 5 on both sides for some points
    y_calib = np.array([4, 28, 30, 42, 51])
    
    # scores: 
    # max(10-4, 4-15) = 6
    # max(20-28, 28-25) = 3
    # max(30-30, 30-35) = 0
    # max(40-42, 42-45) = 0
    # max(50-51, 51-55) = 0
    # Scores: [6, 3, 0, 0, 0]
    
    calibrator.fit(q_low_calib, q_high_calib, y_calib)
    
    assert calibrator.q_adjustment_ >= 0
    assert calibrator.q_adjustment_ == 6.0 # The 90% quantile of this small set
    
    q_low_test = np.array([10])
    q_high_test = np.array([20])
    
    adj_low, adj_high = calibrator.predict(q_low_test, q_high_test)
    assert adj_low[0] == 4.0
    assert adj_high[0] == 26.0

def test_calibrator_predict_before_fit_raises():
    calibrator = ConformalQuantileCalibrator(alpha=0.1)
    with pytest.raises(ValueError, match="Calibrator is not fitted"):
        calibrator.predict(np.array([10]), np.array([20]))

def test_quantile_monotonicity():
    # Simulated check that Q10 <= Q50 <= Q90 after adjustment
    q10 = np.array([1.0, 2.0])
    q50 = np.array([3.0, 4.0])
    q90 = np.array([5.0, 6.0])
    
    # Valid ordering
    assert np.all(q10 <= q50)
    assert np.all(q50 <= q90)
    
    calibrator = ConformalQuantileCalibrator(alpha=0.2)
    calibrator.q_adjustment_ = 1.5
    
    q10_adj, q90_adj = calibrator.predict(q10, q90)
    
    # After widening, the bounds should be even further apart
    assert np.all(q10_adj <= q10)
    assert np.all(q90_adj >= q90)
    assert np.all(q10_adj <= q50)
    assert np.all(q90_adj >= q50)
