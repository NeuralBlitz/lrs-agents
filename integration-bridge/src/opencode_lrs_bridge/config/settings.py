"""
Configuration management for opencode ↔ LRS-Agents integration bridge.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class SecurityConfig(BaseModel):
    """Security configuration settings."""

    enable_auth: bool = Field(True, description="Enable authentication")
    oauth_provider_url: str = Field(
        "http://localhost:8080/oauth", description="OAuth provider URL"
    )
    oauth_client_id: str = Field("integration_bridge", description="OAuth client ID")
    oauth_client_secret: str = Field(
        "test_secret", description="OAuth client secret (must be set via environment)"
    )
    jwt_algorithm: str = Field("RS256", description="JWT algorithm")
    jwt_public_key: Optional[str] = Field(
        "test_public_key", description="JWT public key (must be set via environment)"
    )
    enable_mtls: bool = Field(False, description="Enable mutual TLS")
    cert_file: Optional[str] = Field(None, description="Certificate file path")
    key_file: Optional[str] = Field(None, description="Private key file path")
    ca_file: Optional[str] = Field(None, description="CA certificate file path")

    class Config:
        env_prefix = "SECURITY_"


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = 1000
    websocket_connections: int = 100
    burst_size: int = 100

    class Config:
        env_prefix = "RATE_LIMIT_"


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 9000
    workers: int = 4
    enable_cors: bool = True
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    api_prefix: str = "/api/v1"

    class Config:
        env_prefix = "API_"


class WebSocketConfig(BaseModel):
    """WebSocket configuration."""

    port: int = 9001
    max_connections: int = 100
    heartbeat_interval: int = 30
    message_queue_size: int = 1000

    class Config:
        env_prefix = "WS_"


class OpenCodeConfig(BaseModel):
    """opencode integration configuration."""

    base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.0

    class Config:
        env_prefix = "OPENCODE_"


class LRSConfig(BaseModel):
    """LRS-Agents integration configuration."""

    base_url: str = "http://localhost:8000"
    tui_bridge_port: int = 8000
    rest_port: int = 8001
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.0

    class Config:
        env_prefix = "LRS_"


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "postgresql://localhost/integration_bridge"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseModel):
    """Redis configuration."""

    url: str = "redis://localhost:6379/0"
    max_connections: int = 100
    socket_timeout: int = 5

    class Config:
        env_prefix = "REDIS_"


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_prefix = "MONITORING_"


class IntegrationBridgeConfig(BaseSettings):
    """Main configuration for the integration bridge."""

    environment: str = "development"
    debug: bool = False

    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig())
    rate_limit: RateLimitConfig = Field(default_factory=lambda: RateLimitConfig())
    api: APIConfig = Field(default_factory=lambda: APIConfig())
    websocket: WebSocketConfig = Field(default_factory=lambda: WebSocketConfig())
    opencode: OpenCodeConfig = Field(default_factory=lambda: OpenCodeConfig())
    lrs: LRSConfig = Field(default_factory=lambda: LRSConfig())
    database: DatabaseConfig = Field(default_factory=lambda: DatabaseConfig())
    redis: RedisConfig = Field(default_factory=lambda: RedisConfig())
    monitoring: MonitoringConfig = Field(default_factory=lambda: MonitoringConfig())

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
