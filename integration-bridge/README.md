# opencode ↔ LRS-Agents Integration Bridge

A sophisticated bidirectional API integration service that enables seamless communication between opencode and LRS-Agents, featuring real-time WebSocket communication, enterprise-grade security, and intelligent state synchronization.

## Features

### 🔐 Enterprise Security
- **OAuth 2.0/JWT Authentication** with token management
- **Mutual TLS (mTLS)** for zero-trust security
- **Advanced Rate Limiting** with Redis-backed distributed limits
- **IP-based blocking** and suspicious activity detection
- **Prometheus metrics** for security monitoring

### 🌐 Bidirectional Communication
- **REST APIs** for agent management and tool execution
- **Real-time WebSockets** for state updates and events
- **Server-Sent Events (SSE)** for streaming notifications
- **Automatic fallback** between opencode and LRS systems

### 🔄 State Synchronization
- **Conflict Detection** with multiple resolution strategies
- **Automatic Merging** of agent states
- **Version Control** with change history
- **Manual Override** capabilities

### 🛠️ Tool Integration
- **Smart Routing** between opencode and LRS tools
- **Automatic Fallback** when tools fail
- **Parameter Validation** and type checking
- **Execution History** and analytics

### 📊 Monitoring & Observability
- **Prometheus Metrics** for performance monitoring
- **Structured Logging** with JSON format
- **Health Checks** and readiness probes
- **Grafana Dashboards** for visualization

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/opencode-lrs-bridge.git
cd opencode-lrs-bridge

# Copy environment configuration
cp .env.example .env

# Edit configuration
nano .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f integration-bridge

# Access services
# API: http://localhost:9000
# WebSocket: ws://localhost:9001
# Metrics: http://localhost:9090/metrics
# Grafana: http://localhost:3000 (configure administrator credentials)
```

### Using Kubernetes

```bash
# Create namespace
kubectl create namespace integration-bridge

# Deploy secrets
kubectl apply -f k8s/secrets.yaml

# Deploy the application
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n integration-bridge

# Port-forward for testing
kubectl port-forward svc/integration-bridge-service 9000:80 -n integration-bridge
```

## Configuration

### Environment Variables

```bash
# Core Configuration
ENVIRONMENT=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=9000
WS_PORT=9001
WORKERS=4

# Database
DB_URL=postgresql://user:pass@host:5432/dbname
DB_POOL_SIZE=10

# Redis
REDIS_URL=redis://host:6379/0

# Security
SECURITY_ENABLE_AUTH=true
SECURITY_ENABLE_MTLS=true
SECURITY_CERT_FILE=/app/certs/tls.crt
SECURITY_KEY_FILE=/app/certs/tls.key
SECURITY_CA_FILE=/app/certs/ca.crt

# External Services
OPENCODE_BASE_URL=http://opencode:8080
LRS_BASE_URL=http://lrs-agents:8000

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
RATE_LIMIT_WEBSOCKET_CONNECTIONS=100
```

### OAuth 2.0 Setup

Configure your OAuth provider (e.g., Keycloak, Auth0):

```yaml
SECURITY_OAUTH_PROVIDER_URL=https://your-oauth-provider.com
SECURITY_OAUTH_CLIENT_ID=integration-bridge
SECURITY_OAUTH_CLIENT_SECRET=your-client-secret
```

### Explicit API key authentication

API keys are **disabled unless configured**. Provision a unique high-entropy key
through your secrets manager and supply its SHA-256 digest to the bridge as
`SECURITY__API_KEY_SHA256` (64 hex characters). Send the original key only in the
`X-API-Key` request header over HTTPS; never commit or log the original key.
The bridge compares the configured digest in constant time, and does not accept
any built-in demo API key. Set `SECURITY__ENABLE_AUTH=true` explicitly in
production; the application refuses to start with authentication disabled.

The nested security environment variables use a **double underscore** (for
example `SECURITY__ENABLE_AUTH`). The legacy `SECURITY_ENABLE_AUTH` spelling
is not a Pydantic nested-settings override. The container's mTLS startup
switch is a separate setting and must be reviewed with its TLS termination
configuration before enabling production mTLS.

## API Usage

### Authentication

```bash
# Get JWT token
curl -X POST "https://your-domain.com/oauth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "client_credentials",
    "client_id": "integration-bridge",
    "client_secret": "your-secret"
  }'

# Use token in API calls
curl -X GET "https://your-domain.com/api/v1/agents" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Agent Management

```bash
# Create an agent
curl -X POST "https://your-domain.com/api/v1/agents" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_001",
    "agent_type": "hybrid",
    "config": {
      "goal": "Process user requests"
    },
    "tools": ["search", "file_operation"],
    "preferences": {
      "precision_threshold": 0.8
    }
  }'

# List agents
curl -X GET "https://your-domain.com/api/v1/agents" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Get agent state
curl -X GET "https://your-domain.com/api/v1/agents/agent_001" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Tool Execution

```bash
# Execute a tool
curl -X POST "https://your-domain.com/api/v1/tools/execute" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "search",
    "parameters": {
      "query": "machine learning",
      "max_results": 10
    },
    "agent_id": "agent_001",
    "timeout": 30.0
  }'
```

### WebSocket Connection

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:9001/ws');

// Subscribe to agent updates
ws.onopen = function() {
  ws.send(JSON.stringify({
    type: 'subscribe_agent',
    data: {
      agent_id: 'agent_001'
    }
  }));
};

// Handle messages
ws.onmessage = function(event) {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};

// Handle agent state updates
ws.addEventListener('agent_state_update', function(event) {
  console.log('Agent state changed:', event.detail);
});
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    opencode     │    │ Integration     │    │   LRS-Agents   │
│                 │◄──►│     Bridge      │◄──►│                 │
│  - REST API     │    │                 │    │  - TUI Bridge   │
│  - WebSocket    │    │  - REST API     │    │  - Tool Registry│
│  - Auth System  │    │  - WebSocket    │    │  - Precision    │
└─────────────────┘    │  - Auth Layer   │    │    Tracking     │
                       │  - Rate Limit   │    └─────────────────┘
                       │  - State Sync   │
                       │  - Tool Router  │
                       └──────────────────┘
                                ▲
                                │
                       ┌──────────────────┐
                       │  Infrastructure │
                       │                 │
                       │  - PostgreSQL   │
                       │  - Redis        │
                       │  - Prometheus   │
                       │  - Grafana      │
                       └──────────────────┘
```

## Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:9090/metrics`

Key metrics:
- `http_requests_total` - HTTP request count
- `http_request_duration_seconds` - Request latency
- `active_connections` - Active WebSocket connections
- `rate_limit_hits_total` - Rate limit violations
- `mtls_errors_total` - mTLS errors

### Grafana Dashboard

1. Access Grafana at `http://localhost:3000`
2. Login with admin/admin123
3. Import the provided dashboard configuration
4. Monitor:
   - API performance
   - Agent activity
   - Security events
   - Resource usage

## Development

### Local Development Setup

```bash
# Install dependencies
pip install -e .[dev]

# Run in development mode
uvicorn opencode_lrs_bridge.main:app --reload --port 9000

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

### Adding New Tools

1. Create a new tool adapter:

```python
class CustomToolAdapter(ToolAdapter):
    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        # Implement tool logic
        pass
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        # Return tool definitions
        pass
```

2. Register the adapter in `ToolRouter`:

```python
self.custom_adapter = CustomToolAdapter(config)
```

### State Synchronization

Configure conflict resolution strategies:

```python
# Available strategies:
# - tui_wins: opencode state takes precedence
# - lrs_wins: LRS state takes precedence  
# - timestamp: Most recent update wins
# - merge: Intelligent merging of states

sync_response = await synchronizer.sync_agent_state(
    agent_id="agent_001",
    lrs_state=lrs_state,
    opencode_state=opencode_state,
    strategy=ConflictResolutionStrategy.TIMESTAMP
)
```

## Security

### Certificate Setup

For mTLS, generate certificates:

```bash
# Create CA certificate
openssl req -x509 -sha256 -nodes -days 3650 -newkey rsa:2048 \
  -keyout ca.key -out ca.crt \
  -subj "/CN=Integration Bridge CA"

# Generate server certificate
openssl req -newkey rsa:2048 -nodes -keyout server.key \
  -out server.csr -subj "/CN=bridge.example.com"

# Sign with CA
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt -days 365 -sha256

# Generate client certificate
openssl req -newkey rsa:2048 -nodes -keyout client.key \
  -out client.csr -subj "/CN=client.example.com"

openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out client.crt -days 365 -sha256
```

### Rate Limiting

Configure different rate limits per client:

```python
# API key based rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=1000

# WebSocket connection limits
RATE_LIMIT_WEBSOCKET_CONNECTIONS=100

# Burst capacity
RATE_LIMIT_BURST_SIZE=100
```

## Troubleshooting

### Common Issues

1. **Authentication failures**
   - Check OAuth configuration
   - Verify JWT token validity
   - Ensure mTLS certificates are valid

2. **WebSocket connection issues**
   - Check firewall rules
   - Verify port accessibility
   - Review rate limiting settings

3. **State synchronization conflicts**
   - Review conflict resolution strategy
   - Check agent state format
   - Monitor sync logs

4. **Performance issues**
   - Monitor resource usage
   - Check rate limiting impact
   - Review database connections

### Health Checks

```bash
# Service health
curl http://localhost:9000/health

# Database connectivity
curl http://localhost:9000/health/db

# Redis connectivity  
curl http://localhost:9000/health/redis

# External service status
curl http://localhost:9000/health/external
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run code quality checks
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: [docs/](./docs/)
- Issues: [GitHub Issues](https://github.com/your-org/opencode-lrs-bridge/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/opencode-lrs-bridge/discussions)