# Multi-stage build for LRS-Agents

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY pyproject.toml setup.py /app/
WORKDIR /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 lrsuser && \
    mkdir -p /app /app/logs /app/data && \
    chown -R lrsuser:lrsuser /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=lrsuser:lrsuser . /app/

# Set working directory
WORKDIR /app

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LRS_LOG_DIR=/app/logs \
    LRS_DATA_DIR=/app/data

# Copy entrypoint
COPY --chown=lrsuser:lrsuser docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to non-root user
USER lrsuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["server"]

