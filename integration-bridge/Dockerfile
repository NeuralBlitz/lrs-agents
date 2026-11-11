# Integration Bridge Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by cryptography and native wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and source before installation so setuptools can
# discover the src-layout package during the editable install.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Copy the runtime entrypoint to the exact path used by ENTRYPOINT.
COPY --chmod=755 docker/entrypoint.sh /app/entrypoint.sh

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

EXPOSE 9000 9001 9090

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
