from fastapi.testclient import TestClient
from delay_intelligence.api.main import app
client = TestClient(app)

def test_api_explain():
    response = client.post("/explain", json={"features": {"Line Item Value": 1500}})
    assert response.status_code == 200
