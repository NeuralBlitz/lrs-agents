#!/usr/bin/env bash
# NeuralBlitz v50 authentication smoke test.
# Credentials are supplied by the operator; no defaults are embedded here.

set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:5000}"
: "${ADMIN_USERNAME:?Set ADMIN_USERNAME to a configured demo/user identity}"
: "${ADMIN_PASSWORD:?Set ADMIN_PASSWORD to its password}"
: "${VIEWER_USERNAME:?Set VIEWER_USERNAME to a read-only identity}"
: "${VIEWER_PASSWORD:?Set VIEWER_PASSWORD to its password}"

admin_token="$(
  curl --fail --silent --show-error -X POST "$API_URL/api/v1/auth/token" \
    --data-urlencode "username=$ADMIN_USERNAME" \
    --data-urlencode "password=$ADMIN_PASSWORD" \
    --data-urlencode "grant_type=password" |
    jq --exit-status --raw-output '.access_token'
)"

viewer_token="$(
  curl --fail --silent --show-error -X POST "$API_URL/api/v1/auth/token" \
    --data-urlencode "username=$VIEWER_USERNAME" \
    --data-urlencode "password=$VIEWER_PASSWORD" \
    --data-urlencode "grant_type=password" |
    jq --exit-status --raw-output '.access_token'
)"

admin_status="$(
  curl --silent --output /dev/null --write-out '%{http_code}' \
    -H "Authorization: Bearer $admin_token" "$API_URL/api/v1/metrics"
)"
viewer_status="$(
  curl --silent --output /dev/null --write-out '%{http_code}' \
    -H "Authorization: Bearer $viewer_token" "$API_URL/api/v1/metrics"
)"
anonymous_status="$(
  curl --silent --output /dev/null --write-out '%{http_code}' \
    "$API_URL/api/v1/metrics"
)"

printf 'Admin metrics response: %s\n' "$admin_status"
printf 'Viewer metrics response: %s\n' "$viewer_status"
printf 'Anonymous metrics response: %s\n' "$anonymous_status"

[[ "$admin_status" == "200" ]]
[[ "$viewer_status" == "403" ]]
[[ "$anonymous_status" == "401" ]]

echo "NeuralBlitz v50 authentication smoke tests passed."
