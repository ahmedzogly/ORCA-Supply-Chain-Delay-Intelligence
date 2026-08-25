# Serving Architecture

## Strict Separation
The serving layer translates live requests into pre-compiled structures. It performs NO training, tuning, or threshold optimizations.

## Feature Contract
The raw JSON payload is processed by the same feature schema built during Stage 3. Unknown categoricals are handled natively by CatBoost missing handling, preventing runtime inference errors.
