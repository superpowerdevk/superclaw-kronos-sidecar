FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Vendor the Kronos model code (provides the `model` package: Kronos, KronosTokenizer, KronosPredictor)
RUN git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /app/kronos

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY kronos_sidecar.py .

ENV PYTHONUNBUFFERED=1 \
    KRONOS_DIR=/app/kronos \
    KRONOS_MODEL=NeoQuasar/Kronos-small \
    KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base \
    DEVICE=cpu \
    INTERVAL=1h \
    HORIZON_HRS=4 \
    LOOKBACK=512 \
    SAMPLE_COUNT=30 \
    REFRESH_TTL=300

# Render provides $PORT; default 8080 locally.
CMD ["sh", "-c", "uvicorn kronos_sidecar:app --host 0.0.0.0 --port ${PORT:-8080}"]
