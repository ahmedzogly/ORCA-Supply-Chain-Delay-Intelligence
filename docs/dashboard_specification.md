# Dashboard Specification

The Control Tower dashboard acts as the primary presentation and orchestration layer for the Delay Intelligence system.

## Principles
- **No Duplicated Logic**: Consumes the Stage 9 local API.
- **Actionability**: Highlights recommendations, requiring human approval.
- **Safety**: No write operations or modifications to underlying models/data.

## Architecture
- Framework: Streamlit
- Pages: App layout mapped to 5 distinct views supporting operational execution, analytics, and academic review.
- Data Integration: pi_client.py utilizes TestClient to natively pass requests through the FastAPI router asynchronously without overhead.
