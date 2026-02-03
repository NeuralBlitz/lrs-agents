"""
Test configuration and fixtures for the integration bridge.
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock
import tempfile
import os

from opencode_lrs_bridge.config.settings import IntegrationBridgeConfig
from opencode_lrs_bridge.models.schemas import (
    AgentCreateRequest,
    AgentState,
    ToolExecutionRequest,
    AgentType,
    AgentStatus,
    ToolExecutionStatus,
)


@pytest.fixture
def test_config():
    """Test configuration fixture."""
    from opencode_lrs_bridge.config.settings import (
        SecurityConfig,
        APIConfig,
        WebSocketConfig,
        OpenCodeConfig,
        LRSConfig,
        DatabaseConfig,
        RedisConfig,
        MonitoringConfig,
    )

    return IntegrationBridgeConfig(
        environment="test",
        debug=True,
        api=APIConfig(
            host="127.0.0.1",
            port=9000,
            workers=1,
            enable_cors=True,
            allowed_origins=["http://localhost:3000"],
        ),
        websocket=WebSocketConfig(
            port=9001, max_connections=10, heartbeat_interval=5, message_queue_size=100
        ),
        security=SecurityConfig(
            enable_auth=True,
            oauth_provider_url="http://test-auth:8080",
            oauth_client_id="test_client",
            oauth_client_secret="test_secret",
            jwt_algorithm="RS256",
            jwt_public_key="test_public_key",
            enable_mtls=False,
            cert_file=None,
            key_file=None,
            ca_file=None,
        ),
        opencode=OpenCodeConfig(
            base_url="http://test-opencode:8080",
            api_key="test_api_key",
            timeout=30,
            max_retries=3,
        ),
        lrs=LRSConfig(
            base_url="http://test-lrs:8000",
            tui_bridge_port=8000,
            rest_port=8001,
            timeout=30,
            max_retries=3,
        ),
        database=DatabaseConfig(url="sqlite:///test.db", pool_size=5, max_overflow=10),
        redis=RedisConfig(url="redis://localhost:6379/1", max_connections=10),
        monitoring=MonitoringConfig(
            enable_metrics=True,
            metrics_port=9090,
            enable_logging=True,
            log_level="DEBUG",
        ),
    )


@pytest.fixture
def mock_http_client():
    """Mock HTTP client fixture."""
    client = AsyncMock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def sample_agent_create_request():
    """Sample agent creation request."""
    return AgentCreateRequest(
        agent_id="test_agent_001",
        agent_type=AgentType.HYBRID,
        config={"goal": "Test agent goal", "preferences": {"precision_threshold": 0.8}},
        tools=["search", "file_operation"],
        preferences={"max_retries": 3, "timeout": 30},
        opencode_session_id="session_123",
    )


@pytest.fixture
def sample_agent_state():
    """Sample agent state."""
    from opencode_lrs_bridge.models.schemas import PrecisionData

    return AgentState(
        agent_id="test_agent_001",
        agent_type=AgentType.HYBRID,
        status=AgentStatus.ACTIVE,
        current_task="Test task execution",
        precision_data=[
            PrecisionData(
                tool_name="search",
                alpha=10.0,
                beta=2.0,
                precision=0.83,
                confidence=0.85,
            )
        ],
        belief_state={"current_goal": "Test goal", "context": {"test": True}},
        tool_history=[
            {
                "tool_name": "search",
                "timestamp": "2024-01-01T12:00:00Z",
                "status": "completed",
                "duration": 1.5,
            }
        ],
    )


@pytest.fixture
def sample_tool_execution_request():
    """Sample tool execution request."""
    return ToolExecutionRequest(
        tool_name="search",
        parameters={"query": "test search query", "max_results": 5},
        timeout=30.0,
        agent_id="test_agent_001",
        context={"session_id": "session_123", "user_request": True},
    )


@pytest.fixture
def temp_db_file():
    """Temporary database file fixture."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)


@pytest.fixture
def mock_websocket():
    """Mock WebSocket fixture."""
    websocket = Mock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def mock_redis():
    """Mock Redis fixture."""
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.zadd = AsyncMock(return_value=1)
    redis.zcard = AsyncMock(return_value=0)
    redis.zremrangebyscore = AsyncMock(return_value=[])
    redis.expire = AsyncMock(return_value=True)
    return redis


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_oauth_token():
    """Mock OAuth token fixture."""
    return {
        "access_token": "mock_access_token_12345",
        "refresh_token": "mock_refresh_token_67890",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": "read write",
    }


@pytest.fixture
def mock_jwt_payload():
    """Mock JWT payload fixture."""
    return {
        "user_id": "test_user_001",
        "permissions": ["read_state", "write_state", "execute_tools"],
        "client_id": "test_client",
        "exp": 1735689600,  # Future timestamp
        "iat": 1735686000,
        "type": "access",
    }


@pytest.fixture
def sample_sync_request():
    """Sample state synchronization request."""
    return {
        "agent_id": "test_agent_001",
        "state": {
            "status": "active",
            "current_task": "Updated task",
            "belief_state": {"updated": True, "timestamp": "2024-01-01T12:00:00Z"},
        },
        "strategy": "timestamp",
    }


@pytest.fixture
def mock_certificate_info():
    """Mock certificate information for mTLS testing."""
    return {
        "subject": "CN=test-client",
        "issuer": "CN=test-ca",
        "serial_number": "123456789",
        "not_before": "2024-01-01T00:00:00Z",
        "not_after": "2025-01-01T00:00:00Z",
        "version": 3,
        "signature_algorithm": "sha256WithRSAEncryption",
    }


@pytest.fixture
def sample_analytics_event():
    """Sample analytics event."""
    return {
        "timestamp": "2024-01-01T12:00:00Z",
        "event_type": "tool_execution",
        "source_system": "lrs_agents",
        "agent_id": "test_agent_001",
        "user_id": "test_user_001",
        "session_id": "session_123",
        "duration": 1.5,
        "status": "completed",
        "metadata": {
            "tool_name": "search",
            "parameters": {"query": "test"},
            "result_count": 5,
        },
    }


class IntegrationTestCase:
    """Base class for integration tests."""

    def setup_method(self):
        """Set up test method."""
        # Create a test config if not provided by subclass
        from opencode_lrs_bridge.config.settings import (
            IntegrationBridgeConfig,
            SecurityConfig,
            APIConfig,
            WebSocketConfig,
            OpenCodeConfig,
            LRSConfig,
            DatabaseConfig,
            RedisConfig,
            MonitoringConfig,
        )

        self.config = IntegrationBridgeConfig(
            _env_file=None,  # Disable env file loading for tests
            environment="test",
            debug=True,
            security=SecurityConfig(
                enable_auth=True,
                oauth_provider_url="http://test-auth:8080",
                oauth_client_id="test_client",
                oauth_client_secret="test_secret",
                jwt_algorithm="RS256",
                jwt_public_key="test_public_key",
                enable_mtls=False,
                cert_file=None,
                key_file=None,
                ca_file=None,
            ),
            api=APIConfig(
                host="127.0.0.1",
                port=9000,
                workers=1,
                enable_cors=True,
                allowed_origins=["http://localhost:3000"],
                api_prefix="/api/v1",
            ),
            websocket=WebSocketConfig(
                port=9001,
                max_connections=10,
                heartbeat_interval=5,
                message_queue_size=100,
            ),
            opencode=OpenCodeConfig(
                base_url="http://test-opencode:8080",
                api_key="test_api_key",
                timeout=30,
                max_retries=3,
                retry_backoff=1.0,
            ),
            lrs=LRSConfig(
                base_url="http://test-lrs:8000",
                tui_bridge_port=8000,
                rest_port=8001,
                timeout=30,
                max_retries=3,
                retry_backoff=1.0,
            ),
            database=DatabaseConfig(
                url="sqlite:///test.db", pool_size=5, max_overflow=10, pool_timeout=30
            ),
            redis=RedisConfig(
                url="redis://localhost:6379/1", max_connections=10, socket_timeout=5
            ),
            monitoring=MonitoringConfig(
                enable_metrics=True,
                metrics_port=9090,
                enable_logging=True,
                log_level="DEBUG",
                log_format="json",
            ),
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def teardown_method(self):
        """Tear down test method."""
        self.loop.close()


class WebSocketTestCase(IntegrationTestCase):
    """Base class for WebSocket tests."""

    @pytest.fixture(autouse=True)
    def setup_websocket_test(self, mock_websocket):
        """Set up WebSocket test."""
        self.websocket = mock_websocket
        self.connection_id = "test_connection_001"
        self.user_info = {
            "user_id": "test_user_001",
            "permissions": ["read_state", "write_state"],
        }


class SecurityTestCase(IntegrationTestCase):
    """Base class for security tests."""

    @pytest.fixture(autouse=True)
    def setup_security_test(self, test_config):
        """Set up security test."""
        self.config = test_config
        self.security_middleware = SecurityMiddleware(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.mtls_validator = MTLSValidator(self.config)


class ToolsTestCase(IntegrationTestCase):
    """Base class for tools tests."""

    @pytest.fixture(autouse=True)
    def setup_tools_test(self, test_config, mock_http_client):
        """Set up tools test."""
        self.config = test_config
        self.lrs_adapter = LRSToolAdapter(self.config)
        self.opencode_adapter = OpenCodeToolAdapter(self.config)
        self.tool_router = ToolRouter(self.config)
        self.execution_manager = ToolExecutionManager(self.config)

        # Mock HTTP clients
        self.lrs_adapter.base_url = "http://mock-lrs:8000"
        self.opencode_adapter.base_url = "http://mock-opencode:8080"


class SyncTestCase(IntegrationTestCase):
    """Base class for synchronization tests."""

    @pytest.fixture(autouse=True)
    def setup_sync_test(self, test_config):
        """Set up synchronization test."""
        self.config = test_config
        self.state_merger = StateMerger(self.config)
        self.state_synchronizer = StateSynchronizer(self.config)
        self.change_detector = StateChangeDetector(self.config)


# Import needed classes for test fixtures
from opencode_lrs_bridge.auth.security import (
    SecurityMiddleware,
    RateLimiter,
    MTLSValidator,
)
from opencode_lrs_bridge.tools.integration import (
    LRSToolAdapter,
    OpenCodeToolAdapter,
    ToolRouter,
    ToolExecutionManager,
)
from opencode_lrs_bridge.utils.sync import (
    StateMerger,
    StateSynchronizer,
    StateChangeDetector,
)
