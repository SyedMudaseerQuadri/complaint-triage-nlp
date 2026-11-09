# Serves the API with the TF-IDF baseline (small, CPU-only). Build after training the baseline:
#   python -m src.train_baseline   ->   models/baseline.joblib
#   docker build -t complaint-triage .   &&   docker run -p 8000:8000 complaint-triage
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_BACKEND=baseline
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py src/config.py src/preprocess.py ./src/
COPY api ./api
COPY models/baseline.joblib ./models/baseline.joblib

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
