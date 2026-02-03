#!/bin/bash
set -e

# Docker entrypoint script for opencode-LRS integration bridge

echo "Starting opencode ↔ LRS-Agents Integration Bridge..."
echo "Environment: ${ENVIRONMENT:-development}"
echo "Debug mode: ${DEBUG:-false}"

# Wait for dependencies to be ready
echo "Waiting for dependencies..."

# Wait for Redis
if [ -n "$REDIS_URL" ]; then
    echo "Checking Redis connection..."
    redis-cli -u "$REDIS_URL" ping || exit 1
    echo "Redis is ready"
fi

# Wait for database
if [ -n "$DB_URL" ]; then
    echo "Checking database connection..."
    python -c "
import asyncio
import asyncpg
async def check_db():
    conn = await asyncpg.connect('$DB_URL')
    await conn.close()
    print('Database is ready')
asyncio.run(check_db())
" || exit 1
fi

# Run database migrations if needed
if [ "$ENVIRONMENT" != "development" ]; then
    echo "Running database migrations..."
    # alembic upgrade head
fi

# Initialize SSL certificates if mTLS is enabled
if [ "$SECURITY_ENABLE_MTLS" = "true" ]; then
    echo "mTLS enabled, checking certificates..."
    if [ ! -f "$SECURITY_CERT_FILE" ]; then
        echo "Error: Certificate file not found: $SECURITY_CERT_FILE"
        exit 1
    fi
    if [ ! -f "$SECURITY_KEY_FILE" ]; then
        echo "Error: Private key file not found: $SECURITY_KEY_FILE"
        exit 1
    fi
fi

# Set up monitoring endpoints
if [ "$MONITORING_ENABLE_METRICS" = "true" ]; then
    echo "Metrics enabled on port ${MONITORING_METRICS_PORT:-9090}"
fi

# Start the application with the specified number of workers
WORKERS=${WORKERS:-4}
echo "Starting with $WORKERS workers..."

# Use uvicorn for production
if [ "$ENVIRONMENT" = "production" ]; then
    exec uvicorn opencode_lrs_bridge.main:app \
        --host ${API_HOST:-0.0.0.0} \
        --port ${API_PORT:-9000} \
        --workers $WORKERS \
        --access-log \
        --log-level info \
        --ssl-keyfile "$SECURITY_KEY_FILE" \
        --ssl-certfile "$SECURITY_CERT_FILE" \
        --ssl-ca-file "$SECURITY_CA_FILE"
else
    # Development mode with auto-reload
    exec uvicorn opencode_lrs_bridge.main:app \
        --host ${API_HOST:-0.0.0.0} \
        --port ${API_PORT:-9000} \
        --reload \
        --access-log \
        --log-level debug
fi