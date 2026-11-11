#!/bin/sh
set -eu

echo "Starting opencode ↔ LRS-Agents Integration Bridge..."
echo "Environment: ${ENVIRONMENT:-development}"
echo "Debug mode: ${DEBUG:-false}"

# Fail early when explicitly configured dependencies are unavailable.
if [ -n "${REDIS_URL:-}" ]; then
    echo "Checking Redis connection..."
    python - "${REDIS_URL}" <<'PY'
import asyncio
import sys
from redis.asyncio import Redis

async def main():
    client = Redis.from_url(sys.argv[1])
    try:
        await client.ping()
    finally:
        await client.aclose()

asyncio.run(main())
PY
    echo "Redis is ready"
fi

if [ -n "${DB_URL:-}" ]; then
    echo "Checking database connection..."
    python - "${DB_URL}" <<'PY'
import asyncio
import sys
import asyncpg

async def main():
    connection = await asyncpg.connect(sys.argv[1])
    await connection.close()

asyncio.run(main())
PY
    echo "Database is ready"
fi

if [ "${ENVIRONMENT:-development}" != "development" ]; then
    echo "Production mode: database migration hook is not configured."
fi

# Build arguments for the installed console command. The package script and
# Docker image now use the same, tested application factory.
set -- --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-9000}" \
    --workers "${WORKERS:-4}" \
    --log-level "${LOG_LEVEL:-info}"

if [ -n "${CONFIG_FILE:-}" ]; then
    set -- "$@" --config "${CONFIG_FILE}"
fi

if [ "${ENVIRONMENT:-development}" = "development" ] && [ "${DEBUG:-false}" = "true" ]; then
    set -- "$@" --reload
fi

# Enable TLS only when explicitly configured. When a CA is supplied, require
# client certificates rather than silently accepting unauthenticated clients.
if [ "${SECURITY_ENABLE_MTLS:-false}" = "true" ]; then
    : "${SECURITY_KEY_FILE:?SECURITY_KEY_FILE is required when mTLS is enabled}"
    : "${SECURITY_CERT_FILE:?SECURITY_CERT_FILE is required when mTLS is enabled}"
    : "${SECURITY_CA_FILE:?SECURITY_CA_FILE is required when mTLS is enabled}"

    [ -f "${SECURITY_KEY_FILE}" ] || {
        echo "Private key file not found: ${SECURITY_KEY_FILE}" >&2
        exit 1
    }
    [ -f "${SECURITY_CERT_FILE}" ] || {
        echo "Certificate file not found: ${SECURITY_CERT_FILE}" >&2
        exit 1
    }
    [ -f "${SECURITY_CA_FILE}" ] || {
        echo "CA certificate file not found: ${SECURITY_CA_FILE}" >&2
        exit 1
    }

    set -- "$@" \
        --ssl-keyfile "${SECURITY_KEY_FILE}" \
        --ssl-certfile "${SECURITY_CERT_FILE}" \
        --ssl-ca-certs "${SECURITY_CA_FILE}"
fi

exec opencode-lrs-bridge "$@"
