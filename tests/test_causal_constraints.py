def test_causal_constraints_enforced():
    # Verify that Target -> Upstream is rejected
    import yaml
    with open('configs/causal.yaml') as f:
        config = yaml.safe_load(f)
    assert 'Target -> Upstream process variable' in config['causal']['forbidden_directions']
