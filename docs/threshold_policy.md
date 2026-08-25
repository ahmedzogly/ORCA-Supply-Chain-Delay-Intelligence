# Threshold Governance Policy

## 1. Concept
The decision threshold is not assumed to be 0.50. In imbalanced supply chain data, the default threshold often yields poor operational tradeoffs (e.g., maximizing precision but ignoring 95% of delays).

## 2. Policy Constraints
1. **No Target Leakage**: The threshold MUST be learned strictly from training data (pre-test information).
2. **Optimization Target**: Maximize F1 Score on the inner CV/training probabilities.
3. **Application**: The learned threshold is then applied directly to the validation/test fold without modification.

## 3. Reproducibility
The exact optimal threshold per temporal fold is logged in the evaluation manifest to ensure full deterministic reproducibility.
