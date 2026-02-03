"""
TUI Configuration Management: Centralized configuration for TUI integration.

This component provides configuration management for all TUI-related settings,
including environment-based configuration, validation, and runtime updates.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum


class ConfigFormat(Enum):
    """Supported configuration file formats."""

    JSON = "json"
    YAML = "yaml"
    ENV = "env"


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class WebSocketConfig:
    """WebSocket configuration settings."""

    port: int = 8000
    host: str = "0.0.0.0"
    max_connections: int = 1000
    ping_interval: int = 30
    ping_timeout: int = 10
    buffer_size: int = 1024 * 1024  # 1MB
    enable_compression: bool = True


@dataclass
class RESTConfig:
    """REST API configuration settings."""

    port: int = 8001
    host: str = "0.0.0.0"
    enable_cors: bool = True
    allowed_origins: List[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )
    rate_limit: int = 100  # requests per minute
    enable_auth: bool = False
    api_key_header: str = "X-API-Key"


@dataclass
class StateSyncConfig:
    """State synchronization configuration."""

    sync_interval: float = 1.0  # seconds
    debounce_time: float = 0.1  # seconds
    max_state_history: int = 1000
    conflict_resolution: str = "tui_wins"  # tui_wins, lrs_wins, merge, timestamp
    enable_persistence: bool = True
    persistence_path: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""

    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds
    enable_tracing: bool = False
    log_level: LogLevel = LogLevel.INFO
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds


@dataclass
class UIConfig:
    """User interface configuration."""

    refresh_rate: int = 1000  # milliseconds
    theme: str = "default"  # default, dark, light
    language: str = "en"
    timezone: str = "UTC"
    enable_animations: bool = True
    auto_refresh: bool = True


@dataclass
class AgentConfig:
    """Agent-specific configuration."""

    default_tool_timeout: float = 30.0  # seconds
    max_concurrent_tasks: int = 10
    enable_tool_caching: bool = True
    cache_ttl: int = 300  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


@dataclass
class TUIIntegrationConfig:
    """Complete TUI integration configuration."""

    # Component configurations
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    rest: RESTConfig = field(default_factory=RESTConfig)
    state_sync: StateSyncConfig = field(default_factory=StateSyncConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Global settings
    debug: bool = False
    environment: str = "development"  # development, staging, production
    version: str = "1.0.0"

    # Feature flags
    enable_websockets: bool = True
    enable_rest_api: bool = True
    enable_state_mirroring: bool = True
    enable_precision_mapping: bool = True
    enable_multi_agent_coordination: bool = True


class TUIConfigManager:
    """
    Configuration manager for TUI integration.

    Provides:
    - Environment-based configuration loading
    - Configuration validation
    - Runtime configuration updates
    - Configuration file support (JSON/YAML)
    - Environment variable overrides
    - Configuration hot-reloading

    Examples:
        >>> # Load from file
        >>> config_manager = TUIConfigManager.from_file("config.yaml")
        >>> config = config_manager.get_config()
        >>>
        >>> # Load from environment
        >>> config_manager = TUIConfigManager.from_env()
        >>>
        >>> # Update configuration at runtime
        >>> config_manager.update_config({
        ...     "websocket": {"port": 8080}
        ... })
        >>>
        >>> # Validate configuration
        >>> is_valid = config_manager.validate_config()
    """

    def __init__(self, config: Optional[TUIIntegrationConfig] = None):
        """
        Initialize configuration manager.

        Args:
            config: Initial configuration (optional)
        """
        self.config = config or TUIIntegrationConfig()
        self.logger = logging.getLogger(__name__)

        # Configuration sources
        self._config_sources: List[str] = []

        # Validation rules
        self._validation_rules = self._setup_validation_rules()

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "TUIConfigManager":
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            ConfigManager instance
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Determine format from file extension
        if config_path.suffix.lower() == ".json":
            format_type = ConfigFormat.JSON
        elif config_path.suffix.lower() in [".yaml", ".yml"]:
            format_type = ConfigFormat.YAML
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")

        # Load configuration
        with open(config_path, "r") as f:
            if format_type == ConfigFormat.JSON:
                config_data = json.load(f)
            else:  # YAML
                config_data = yaml.safe_load(f)

        # Create configuration object
        config = cls._dict_to_config(config_data)

        manager = cls(config)
        manager._config_sources.append(f"file:{config_path}")

        return manager

    @classmethod
    def from_env(cls, prefix: str = "LRS_TUI_") -> "TUIConfigManager":
        """
        Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix

        Returns:
            ConfigManager instance
        """
        config_data = {}

        # Scan environment variables
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix) :].lower()

                # Convert nested keys (double underscore)
                if "__" in config_key:
                    parts = config_key.split("__")
                    current = config_data
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = cls._convert_env_value(value)
                else:
                    config_data[config_key] = cls._convert_env_value(value)

        # Create configuration object
        config = cls._dict_to_config(config_data)

        manager = cls(config)
        manager._config_sources.append(f"env:{prefix}")

        return manager

    @classmethod
    def from_defaults(cls) -> "TUIConfigManager":
        """
        Create config manager with default values.

        Returns:
            ConfigManager instance with default configuration
        """
        manager = cls()
        manager._config_sources.append("defaults")

        return manager

    def get_config(self) -> TUIIntegrationConfig:
        """
        Get current configuration.

        Returns:
            Current configuration
        """
        return self.config

    def update_config(self, updates: Dict[str, Any], merge: bool = True) -> bool:
        """
        Update configuration with new values.

        Args:
            updates: Configuration updates
            merge: Whether to merge with existing config or replace

        Returns:
            Success status
        """
        try:
            if merge:
                # Convert updates to config and merge
                update_config = self._dict_to_config(updates)
                self._merge_configs(self.config, update_config)
            else:
                # Replace entire configuration
                self.config = self._dict_to_config(updates)

            self._config_sources.append("runtime_update")

            # Validate updated configuration
            if not self.validate_config():
                self.logger.error("Configuration validation failed after update")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if configuration is valid
        """
        try:
            # Apply validation rules
            for rule_path, rule_func in self._validation_rules.items():
                value = self._get_nested_value(self.config, rule_path)

                if not rule_func(value):
                    self.logger.error(f"Configuration validation failed for: {rule_path}")
                    return False

            # Port range validation
            if not (1 <= self.config.websocket.port <= 65535):
                self.logger.error(f"Invalid WebSocket port: {self.config.websocket.port}")
                return False

            if not (1 <= self.config.rest.port <= 65535):
                self.logger.error(f"Invalid REST port: {self.config.rest.port}")
                return False

            # Ensure ports are different
            if self.config.websocket.port == self.config.rest.port:
                self.logger.error("WebSocket and REST ports cannot be the same")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    def save_to_file(
        self, file_path: Union[str, Path], format_type: ConfigFormat = ConfigFormat.YAML
    ) -> bool:
        """
        Save current configuration to file.

        Args:
            file_path: Output file path
            format_type: File format

        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert configuration to dictionary
            config_dict = self._config_to_dict(self.config)

            # Save based on format
            with open(file_path, "w") as f:
                if format_type == ConfigFormat.JSON:
                    json.dump(config_dict, f, indent=2)
                else:  # YAML
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for display.

        Returns:
            Configuration summary
        """
        return {
            "version": self.config.version,
            "environment": self.config.environment,
            "debug": self.config.debug,
            "sources": self._config_sources,
            "features": {
                "websockets": self.config.enable_websockets,
                "rest_api": self.config.enable_rest_api,
                "state_mirroring": self.config.enable_state_mirroring,
                "precision_mapping": self.config.enable_precision_mapping,
                "multi_agent_coordination": self.config.enable_multi_agent_coordination,
            },
            "endpoints": {
                "websocket": f"{self.config.websocket.host}:{self.config.websocket.port}",
                "rest": f"{self.config.rest.host}:{self.config.rest.port}",
            },
            "monitoring": {
                "enabled": self.config.monitoring.enable_metrics,
                "log_level": self.config.monitoring.log_level.value,
                "health_checks": self.config.monitoring.enable_health_checks,
            },
        }

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> TUIIntegrationConfig:
        """Convert dictionary to configuration object."""

        # Handle nested dictionaries
        websocket_config = WebSocketConfig(**config_dict.get("websocket", {}))
        rest_config = RESTConfig(**config_dict.get("rest", {}))
        state_sync_config = StateSyncConfig(**config_dict.get("state_sync", {}))
        monitoring_config = MonitoringConfig(**config_dict.get("monitoring", {}))
        ui_config = UIConfig(**config_dict.get("ui", {}))
        agent_config = AgentConfig(**config_dict.get("agent", {}))

        # Convert log level string to enum
        if isinstance(monitoring_config.log_level, str):
            try:
                monitoring_config.log_level = LogLevel(monitoring_config.log_level.lower())
            except ValueError:
                monitoring_config.log_level = LogLevel.INFO

        return TUIIntegrationConfig(
            websocket=websocket_config,
            rest=rest_config,
            state_sync=state_sync_config,
            monitoring=monitoring_config,
            ui=ui_config,
            agent=agent_config,
            debug=config_dict.get("debug", False),
            environment=config_dict.get("environment", "development"),
            version=config_dict.get("version", "1.0.0"),
            enable_websockets=config_dict.get("enable_websockets", True),
            enable_rest_api=config_dict.get("enable_rest_api", True),
            enable_state_mirroring=config_dict.get("enable_state_mirroring", True),
            enable_precision_mapping=config_dict.get("enable_precision_mapping", True),
            enable_multi_agent_coordination=config_dict.get(
                "enable_multi_agent_coordination", True
            ),
        )

    @staticmethod
    def _config_to_dict(config: TUIIntegrationConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""

        return {
            "websocket": asdict(config.websocket),
            "rest": asdict(config.rest),
            "state_sync": asdict(config.state_sync),
            "monitoring": {
                **asdict(config.monitoring),
                "log_level": config.monitoring.log_level.value,
            },
            "ui": asdict(config.ui),
            "agent": asdict(config.agent),
            "debug": config.debug,
            "environment": config.environment,
            "version": config.version,
            "enable_websockets": config.enable_websockets,
            "enable_rest_api": config.enable_rest_api,
            "enable_state_mirroring": config.enable_state_mirroring,
            "enable_precision_mapping": config.enable_precision_mapping,
            "enable_multi_agent_coordination": config.enable_multi_agent_coordination,
        }

    def _merge_configs(self, base: TUIIntegrationConfig, update: TUIIntegrationConfig):
        """Merge configuration updates into base configuration."""

        # Simple field-level merge
        for field_name in update.__dataclass_fields__:
            update_value = getattr(update, field_name)

            if update_value is not None:
                if hasattr(update_value, "__dataclass_fields__"):
                    # Nested dataclass - merge recursively
                    base_value = getattr(base, field_name)
                    self._merge_dataclass(base_value, update_value)
                else:
                    # Simple value - replace
                    setattr(base, field_name, update_value)

    def _merge_dataclass(self, base: Any, update: Any):
        """Merge nested dataclass objects."""

        for field_name in update.__dataclass_fields__:
            update_value = getattr(update, field_name)

            if update_value is not None:
                setattr(base, field_name, update_value)

    def _setup_validation_rules(self) -> Dict[str, callable]:
        """Setup configuration validation rules."""

        return {
            "websocket.port": lambda x: 1 <= x <= 65535,
            "websocket.max_connections": lambda x: x > 0,
            "rest.port": lambda x: 1 <= x <= 65535,
            "rest.rate_limit": lambda x: x > 0,
            "state_sync.sync_interval": lambda x: x > 0,
            "state_sync.debounce_time": lambda x: x >= 0,
            "state_sync.max_state_history": lambda x: x > 0,
            "monitoring.metrics_interval": lambda x: x > 0,
            "monitoring.health_check_interval": lambda x: x > 0,
            "ui.refresh_rate": lambda x: x > 0,
            "agent.default_tool_timeout": lambda x: x > 0,
            "agent.max_concurrent_tasks": lambda x: x > 0,
            "agent.cache_ttl": lambda x: x >= 0,
            "agent.retry_attempts": lambda x: x >= 0,
            "agent.retry_delay": lambda x: x >= 0,
        }

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get nested value using dot notation."""

        parts = path.split(".")
        current = obj

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    @staticmethod
    def _convert_env_value(value: str) -> Union[str, int, float, bool, List[str]]:
        """Convert environment variable string to appropriate type."""

        # Handle boolean values
        if value.lower() in ("true", "yes", "1"):
            return True
        elif value.lower() in ("false", "no", "0"):
            return False

        # Handle integers
        try:
            return int(value)
        except ValueError:
            pass

        # Handle floats
        try:
            return float(value)
        except ValueError:
            pass

        # Handle comma-separated lists
        if "," in value:
            return [item.strip() for item in value.split(",")]

        # Default to string
        return value


# Default configuration template
DEFAULT_CONFIG_TEMPLATE = {
    "websocket": {
        "port": 8000,
        "host": "0.0.0.0",
        "max_connections": 1000,
        "ping_interval": 30,
        "ping_timeout": 10,
    },
    "rest": {
        "port": 8001,
        "host": "0.0.0.0",
        "enable_cors": True,
        "allowed_origins": ["http://localhost:3000", "http://localhost:8080"],
        "rate_limit": 100,
    },
    "state_sync": {
        "sync_interval": 1.0,
        "debounce_time": 0.1,
        "max_state_history": 1000,
        "conflict_resolution": "tui_wins",
    },
    "monitoring": {
        "enable_metrics": True,
        "metrics_interval": 60,
        "log_level": "info",
        "enable_health_checks": True,
    },
    "ui": {"refresh_rate": 1000, "theme": "default", "language": "en", "timezone": "UTC"},
    "agent": {
        "default_tool_timeout": 30.0,
        "max_concurrent_tasks": 10,
        "enable_tool_caching": True,
        "cache_ttl": 300,
    },
    "debug": False,
    "environment": "development",
    "enable_websockets": True,
    "enable_rest_api": True,
    "enable_state_mirroring": True,
    "enable_precision_mapping": True,
    "enable_multi_agent_coordination": True,
}
