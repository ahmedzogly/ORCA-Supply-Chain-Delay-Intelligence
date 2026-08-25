from fastapi.testclient import TestClient
from delay_intelligence.api.main import app
client = TestClient(app)

def test_api_safety_no_execution():
    response = client.post("/execute", json={})
    assert response.status_code == 404
