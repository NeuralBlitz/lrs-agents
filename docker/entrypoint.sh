#!/bin/bash
set -e

# Entrypoint script for LRS-Agents Docker container

echo "Starting LRS-Agents container..."

# Wait for dependencies (if needed)
if [ -n "$WAIT_FOR_DB" ]; then
    echo "Waiting for database..."
    while ! nc -z ${DB_HOST:-postgres} ${DB_PORT:-5432}; do
        sleep 1
    done
    echo "Database is ready!"
fi

# Run database migrations (if applicable)
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running migrations..."
    # Add migration commands here
fi

# Set up logging directory
mkdir -p /app/logs
chmod 755 /app/logs

# Export environment variables for LRS
export LRS_LOG_DIR=/app/logs
export LRS_DATA_DIR=/app/data

# Execute the main command
if [ "$1" = "server" ]; then
    echo "Starting LRS-Agents API server..."
    exec uvicorn lrs.api.server:app --host 0.0.0.0 --port ${PORT:-8000}

elif [ "$1" = "worker" ]; then
    echo "Starting LRS-Agents worker..."
    exec celery -A lrs.worker worker --loglevel=info

elif [ "$1" = "dashboard" ]; then
    echo "Starting LRS-Agents dashboard..."
    exec streamlit run lrs/monitoring/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0

elif [ "$1" = "benchmark" ]; then
    echo "Running benchmark: ${2:-chaos}"
    exec python examples/chaos_benchmark.py

elif [ "$1" = "shell" ]; then
    echo "Starting interactive shell..."
    exec python

else
    # Execute custom command
    echo "Executing: $@"
    exec "$@"
fi

