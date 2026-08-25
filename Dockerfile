FROM python:3.11-slim

WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY pyproject.toml .
COPY src/ ./src/
COPY artifacts/model_registry/v1/ ./artifacts/model_registry/v1/

# Expose API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Start FastAPI app
CMD ["uvicorn", "delay_intelligence.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
