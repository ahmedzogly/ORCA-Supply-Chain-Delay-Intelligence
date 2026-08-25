from fastapi.testclient import TestClient
from delay_intelligence.api.main import app
client = TestClient(app)

def test_api_recommend():
    response = client.post("/recommend", json={"features": {"Line Item Value": 1500}})
    assert response.status_code == 200
