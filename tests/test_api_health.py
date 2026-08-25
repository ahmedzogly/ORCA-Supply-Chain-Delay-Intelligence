from fastapi.testclient import TestClient
from delay_intelligence.api.main import app
client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
