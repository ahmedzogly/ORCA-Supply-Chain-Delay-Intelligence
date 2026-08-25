from fastapi.testclient import TestClient
from delay_intelligence.api.main import app
client = TestClient(app)

def test_end_to_end():
    payload = {"features": {"Line Item Value": 500000, "Shipment Mode": "Ocean", "Fulfill Via": "From RDC"}}
    pred_res = client.post("/predict", json=payload).json()
    expl_res = client.post("/explain", json=payload).json()
    rec_res = client.post("/recommend", json=payload).json()
    assert pred_res['probability_late'] == expl_res['probability_late']
