from fastapi.testclient import TestClient
from delay_intelligence.api.main import app
client = TestClient(app)

def test_api_schema_forbidden_fields():
    response = client.post("/predict", json={"features": {"Delay_Days": 10}})
    assert response.status_code == 422
