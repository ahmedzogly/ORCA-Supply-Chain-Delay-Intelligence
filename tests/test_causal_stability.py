def test_causal_stability():
    import pandas as pd
    import os
    if os.path.exists('artifacts/causal/causal_edge_stability.csv'):
        df = pd.read_csv('artifacts/causal/causal_edge_stability.csv')
        assert 'fold_count' in df.columns
        assert 'stability_class' in df.columns
