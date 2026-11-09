# NeuralBlitz v50 Flask API runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY neuralblitz-v50/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=10001:10001 neuralblitz-v50/applications/ /app/applications/
COPY --chown=10001:10001 neuralblitz-v50/multi_reality_nn.py /app/
COPY --chown=10001:10001 neuralblitz-v50/quantum_spiking_neuron.py /app/

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /app/logs /app/data \
    && chown -R app:app /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:5000/api/v1/health || exit 1

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${FLASK_PORT:-5000} --workers ${NB_WORKERS:-4} --threads ${NB_THREADS:-4} --timeout ${NB_TIMEOUT:-120} --max-requests ${NB_MAX_REQUESTS:-1000} --access-logfile - --error-logfile - applications.unified_api:app"]
