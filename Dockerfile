# One image, two roles. The compose file and Render both override CMD:
#   API : uvicorn src.api:app --host 0.0.0.0 --port 8000
#   UI  : streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY locustfile.py .

# The CIFAR pickles are 186 MB and cannot live in git, so pull them at build time.
RUN python -m scripts.download_data

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

