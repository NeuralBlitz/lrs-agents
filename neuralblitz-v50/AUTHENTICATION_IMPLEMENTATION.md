# NeuralBlitz v50 Authentication and Deployment Security

**Audit update:** 9 November 2026

## Security posture

The API no longer embeds a reusable JWT signing key or default demo passwords in source. Production startup now requires a deployment-provided `JWT_SECRET`. Demo identities are optional, configured using password hashes, and disabled in production. The repository must not contain TLS private keys.

The prior TLS private key and source-embedded JWT signing key were both present in the public repository. The remediation branch removes their current source copies and adds ignore rules, but this does **not** remove historical copies or invalidate credentials already issued with the old JWT key. Treat both values as compromised: replace the TLS certificate/key pair at the TLS termination point, set a new deployment-managed `JWT_SECRET`, and invalidate existing JWTs. Historical secret removal requires a separately planned history rewrite.

## 1. JWT configuration

Required production environment variable:

- `JWT_SECRET`: a randomly generated, high-entropy signing secret stored in the deployment's secret manager.

Generate a value locally with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Do not paste the generated value into source control, documentation, issue comments, or chat. Do not reuse the test-only signing key.

When `FLASK_ENV=production` and `JWT_SECRET` is absent, the application fails during import rather than starting with a known default. In non-production mode only, an ephemeral random signing key is generated if none is supplied; tokens signed with that key become invalid after process restart.

## 2. Optional demo identities

Demo identities are not built into the application. Configure them only in an isolated non-production environment using `NEURALBLITZ_DEMO_USERS_JSON`.

Each user entry must contain:
- `password_hash`: a Werkzeug password hash, never a plaintext password.
- `scopes`: an array of allowed API scopes.

Generate a hash without storing the plaintext in source control:

```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash(input('Demo password: ')))"
```

Example JSON structure (replace the placeholder hash with a generated value):

```json
{
  "developer": {
    "password_hash": "<generated-password-hash>",
    "scopes": ["read", "metrics"]
  }
}
```

Set `ENABLE_DEMO_AUTH=true` only in a non-production environment. The `/api/v1/auth/demo` endpoint returns configured usernames and scopes only; it never returns passwords or password hashes. It returns 404 unless demo authentication is enabled, and demo authentication is always disabled when `FLASK_ENV=production`.

For production, connect a proper identity provider or a managed user store rather than enabling demo identities.

## 3. API authentication

- `POST /api/v1/auth/token`: accepts configured username/password and issues a short-lived HS256 JWT.
- `GET /api/v1/health`: public liveness endpoint.
- `/api/v1/status`, `/api/v1/metrics`, state, and execution endpoints: require bearer authentication and the appropriate scope.
- `POST /api/v1/auth/introspect`: requires a valid bearer token.
- `GET /api/v1/auth/demo`: non-production only, and only when explicitly enabled.

The Prometheus exposition endpoint `/metrics` is intended for the private Docker network. Do not publish the API container port to the public internet.

## 4. Deployment

The repository root Dockerfile builds the v50 API from `neuralblitz-v50/requirements.txt` and its application sources. The root Compose file requires `JWT_SECRET` and binds the API, Prometheus, and Grafana management ports to localhost. Nginx exposes HTTP on port 80 for an upstream TLS-terminating ingress.

Example environment preparation:

```bash
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export GRAFANA_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up --build -d
```

Keep these values in the deployment secret store rather than shell history where possible. The Compose stack does not ship or mount repository-managed TLS keys. Configure TLS certificates at a trusted ingress/load balancer and rotate the previously exposed certificate pair.

## 5. Operational requirements before production

1. Replace the exposed TLS certificate/key pair and verify the new certificate chain.
2. Decide whether Git history cleanup is required; the deleted files remain recoverable from historical commits until history is rewritten.
3. Store `JWT_SECRET` and `GRAFANA_PASSWORD` in a secret manager.
4. Configure `CORS_ORIGINS` to the actual trusted frontend origins.
5. Keep demo authentication disabled.
6. Run the v50 authentication tests, container build, Compose validation, Nginx configuration check, and container health smoke test.
7. Confirm the application is reachable only through the intended ingress and that Prometheus management interfaces remain private.

No deployment or secret rotation is performed by this repository change.
