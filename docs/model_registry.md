# Model Registry

The local filesystem registry is located at rtifacts/model_registry/v1/.

It contains:
- catboost_champion.cbm
- eature_schema.json
- decision.yaml
- metadata.json

Every prediction can trace its model_version, prediction_contract_version, and configuration_version via the API response.
