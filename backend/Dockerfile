FROM python:3.13-slim

WORKDIR /app

# gcc needed for XGBoost source builds on ARM64 if no wheel available
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

# 2 workers — safe for 24 GB Oracle VM; model (~150 MB) loaded once per worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
