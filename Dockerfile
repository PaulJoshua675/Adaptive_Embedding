FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_USE_TRANSFORMERS=0 \
    EMBEDDING_LOCAL_ONLY=1 \
    RERANKER_LOCAL_ONLY=1 \
    CLASSIFIER_LOCAL_ONLY=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-hf.txt ./
RUN pip install --no-cache-dir -r requirements-hf.txt \
    && pip install --no-cache-dir "fastapi==0.115.12" "uvicorn[standard]==0.34.2"

COPY . ./
RUN mkdir -p dataset vector_store data/runtime_uploads

EXPOSE 8000 7860 8501
CMD ["sh", "-c", "uvicorn main_api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.address 0.0.0.0 --server.port \"${PORT:-7860}\" --server.headless true; wait"]