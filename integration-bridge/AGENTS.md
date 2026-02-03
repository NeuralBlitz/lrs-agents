# AGENTS.md - Development Guidelines for Agentic Coding

This file contains essential information for agentic coding agents working in the opencode-lrs-bridge repository.

## Build, Lint, and Test Commands

### Development Setup
```bash
# Install dependencies in development mode
pip install -e .[dev]

# Run in development mode with auto-reload
uvicorn opencode_lrs_bridge.main:app --reload --port 9000

# Alternative using the CLI entry point
opencode-lrs-bridge --reload --port 9000
```

### Code Quality
```bash
# Format code with Black (line length: 88)
black src/ tests/

# Lint code with Ruff
ruff check src/ tests/

# Type checking with MyPy
mypy src/

# Run all three in sequence
black src/ tests/ && ruff check src/ tests/ && mypy src/
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_api.py -v

# Run specific test function
pytest tests/test_api.py::TestAgentManager::test_create_agent -v

# Run tests by marker
pytest tests/ -v -m "unit"          # Unit tests only
pytest tests/ -v -m "integration"   # Integration tests only
pytest tests/ -v -m "websocket"     # WebSocket tests only
pytest tests/ -v -m "security"      # Security tests only

# Run tests without coverage (faster)
pytest tests/ -v

# Run with specific Python version
python3.11 -m pytest tests/ -v
```

### Docker and Deployment
```bash
# Build Docker image
docker build -t opencode-lrs-bridge .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f integration-bridge

# Kubernetes deployment
kubectl apply -f k8s/
```

## Code Style Guidelines

### Import Organization
```python
# Standard library imports first
import asyncio
import logging
from typing import Dict, List, Optional

# Third-party imports next
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports last
from ..config.settings import IntegrationBridgeConfig
from ..models.schemas import AgentState
```

### Type Annotations
- Always use type hints for function parameters and return values
- Use `Optional[T]` for nullable types
- Use `Union[T, U]` or `T | U` for multiple types (Python 3.10+)
- Use `Dict[str, Any]` for generic dictionaries with specific key types
- Use `List[T]` instead of `list[T]` for compatibility

```python
def create_agent(
    request: AgentCreateRequest,
    timeout: Optional[float] = None
) -> AgentState:
    """Create a new agent with optional timeout."""
    pass
```

### Naming Conventions
- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Configuration classes**: `Config` suffix (e.g., `SecurityConfig`)

### Error Handling
```python
# Use structured logging with context
logger = structlog.get_logger(__name__)

try:
    result = await some_operation()
    logger.info("Operation successful", result=result)
except httpx.HTTPError as e:
    logger.error("HTTP request failed", error=str(e), url=url)
    raise HTTPException(status_code=500, detail="External service error")
except Exception as e:
    logger.error("Unexpected error", error=str(e))
    raise
```

### Async/Await Patterns
- All I/O operations must be async
- Use `asyncio.create_task()` for background operations
- Always handle timeouts with `httpx.AsyncClient(timeout=...)`

```python
async def fetch_agent_data(agent_id: str) -> Dict[str, Any]:
    """Fetch agent data from external service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base_url}/agents/{agent_id}")
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        logger.error("Request timeout", agent_id=agent_id)
        raise HTTPException(status_code=504, detail="Request timeout")
```

### Configuration Management
- Use Pydantic models for configuration
- Environment variables with `env_prefix` in Config classes
- Default values should be sensible for development
- Secrets must be loaded from environment, never hardcoded

```python
class SecurityConfig(BaseModel):
    enable_auth: bool = Field(True, description="Enable authentication")
    oauth_client_secret: str = Field(..., description="OAuth client secret")
    
    class Config:
        env_prefix = "SECURITY_"
```

### Testing Patterns
- Use pytest fixtures for common test data
- Mock external services with `unittest.mock.AsyncMock`
- Test both success and error cases
- Use descriptive test names

```python
@pytest.mark.asyncio
async def test_create_agent_success(test_config, mock_http_client):
    """Test successful agent creation."""
    agent_manager = AgentManager(test_config)
    mock_http_client.post.return_value.status_code = 201
    
    request = AgentCreateRequest(
        agent_id="test_agent",
        agent_type=AgentType.HYBRID,
        config={"goal": "test"}
    )
    
    result = await agent_manager.create_agent(request)
    assert result.agent_id == "test_agent"
    assert result.agent_type == AgentType.HYBRID
```

### FastAPI Patterns
- Use dependency injection for authentication and authorization
- Return Pydantic models directly for automatic serialization
- Use proper HTTP status codes
- Include request/response models in type hints

```python
@app.post("/agents", response_model=AgentState)
async def create_agent(
    request: AgentCreateRequest,
    current_user: Dict[str, Any] = require_permission("manage_agents"),
) -> AgentState:
    """Create new agent."""
    return await agent_manager.create_agent(request)
```

### Logging and Monitoring
- Use structured logging with JSON format
- Include relevant context in log messages
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Use Prometheus metrics for monitoring

```python
logger.info(
    "Agent created successfully",
    agent_id=agent_id,
    agent_type=request.agent_type,
    user_id=current_user["user_id"]
)
```

### Security Considerations
- Always validate input data with Pydantic models
- Use HTTPS for all external communications
- Implement rate limiting for API endpoints
- Never log sensitive information (passwords, tokens)
- Use environment variables for secrets

### Database and Redis
- Use async database drivers (asyncpg, aioredis)
- Implement connection pooling
- Use context managers for database connections
- Handle connection errors gracefully

### WebSocket Implementation
- Use connection IDs for tracking
- Implement heartbeat/ping-pong for connection health
- Clean up resources on disconnect
- Use message queues for high-volume scenarios

### Performance Guidelines
- Use async for all I/O operations
- Implement caching where appropriate
- Set reasonable timeouts for external calls
- Monitor memory usage in long-running processes

## Testing Markers

This project uses pytest markers for test categorization:
- `@pytest.mark.asyncio`: Async tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.security`: Security tests
- `@pytest.mark.websocket`: WebSocket tests

Run specific test categories: `pytest tests/ -v -m "unit"`

## Project Structure

```
src/opencode_lrs_bridge/
├── main.py              # Application entry point
├── config/              # Configuration management
├── models/              # Pydantic schemas
├── api/                 # FastAPI endpoints
├── auth/                # Authentication and security
├── websocket/           # WebSocket management
├── tools/               # Tool execution and routing
├── utils/               # Utilities (database, sync, etc.)
└── optimization/        # Performance optimization

tests/
├── conftest.py          # Test configuration and fixtures
├── test_api.py          # API endpoint tests
├── test_websocket.py    # WebSocket tests
└── integration_load_tests.py  # Load testing
```

## Common Patterns

### Dependency Injection
FastAPI's dependency system is used for configuration, authentication, and shared components. All components are initialized in `main.py` and stored in `app.state`.

### Error Responses
Use HTTP status codes consistently:
- 200: Success
- 201: Created
- 400: Bad Request (validation errors)
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 409: Conflict (resource exists)
- 500: Internal Server Error
- 504: Gateway Timeout

### External Service Integration
Always use httpx with proper timeout handling and retry logic. Implement circuit breakers for critical external services.