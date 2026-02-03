"""
TUI Plugin Architecture: Extensible plugin system for TUI integration.

This component provides a plugin architecture that allows extending TUI functionality
with custom visualizations, handlers, and integrations.
"""

import importlib
import inspect
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class PluginType(Enum):
    """Types of TUI plugins."""

    VISUALIZATION = "visualization"
    HANDLER = "handler"
    TRANSFORMER = "transformer"
    INTEGRATION = "integration"
    THEME = "theme"
    WIDGET = "widget"


class PluginStatus(Enum):
    """Plugin status states."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Metadata for TUI plugins."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    min_lrs_version: str = "0.1.0"
    max_lrs_version: Optional[str] = None
    config_schema: Optional[Dict[str, Any]] = None
    permissions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class PluginInfo:
    """Runtime information about a loaded plugin."""

    plugin_class: Type
    instance: Any
    metadata: PluginMetadata
    status: PluginStatus
    loaded_at: datetime
    config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    usage_stats: Dict[str, int] = field(default_factory=dict)


class TUIPlugin(ABC):
    """
    Abstract base class for TUI plugins.

    All TUI plugins must inherit from this class and implement
    the required methods based on their plugin type.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.

        Args:
            config: Plugin configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self.initialized = False

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            True if initialization successful
        """
        pass

    async def cleanup(self) -> bool:
        """
        Cleanup plugin resources.

        Returns:
            True if cleanup successful
        """
        return True

    async def handle_event(self, event_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle TUI event (optional).

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            Response data or None
        """
        return None

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get plugin health status.

        Returns:
            Health status information
        """
        return {
            "status": "healthy" if self.initialized else "uninitialized",
            "last_check": datetime.now().isoformat(),
        }


class VisualizationPlugin(TUIPlugin):
    """
    Base class for visualization plugins.

    Visualization plugins provide custom visual representations
    of agent state, precision data, or other information.
    """

    @abstractmethod
    async def render(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render visualization data.

        Args:
            data: Data to visualize
            context: Rendering context

        Returns:
            Rendered visualization data
        """
        pass

    def get_supported_data_types(self) -> List[str]:
        """
        Get list of supported data types.

        Returns:
            List of supported data types
        """
        return []


class HandlerPlugin(TUIPlugin):
    """
    Base class for handler plugins.

    Handler plugins process specific types of events or requests.
    """

    @abstractmethod
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle request or event.

        Args:
            request: Request data

        Returns:
            Handler response
        """
        pass

    def get_handled_events(self) -> List[str]:
        """
        Get list of handled event types.

        Returns:
            List of event types
        """
        return []


class TransformerPlugin(TUIPlugin):
    """
    Base class for data transformer plugins.

    Transformer plugins modify or transform data before it's
    sent to the TUI or processed by other components.
    """

    @abstractmethod
    async def transform(self, data: Dict[str, Any], direction: str) -> Dict[str, Any]:
        """
        Transform data.

        Args:
            data: Data to transform
            direction: Transformation direction ('to_tui', 'from_tui')

        Returns:
            Transformed data
        """
        pass

    def get_transform_types(self) -> List[str]:
        """
        Get list of data types this transformer handles.

        Returns:
            List of data types
        """
        return []


class TUIPluginManager:
    """
    Plugin manager for TUI integration.

    Features:
    - Plugin discovery and loading
    - Dependency management
    - Lifecycle management
    - Configuration management
    - Health monitoring
    - Event routing to plugins
    - Plugin permissions and security

    Examples:
        >>> manager = TUIPluginManager()
        >>>
        >>> # Load plugins from directory
        >>> await manager.load_plugins_from_directory("./plugins")
        >>>
        >>> # Load specific plugin
        >>> await manager.load_plugin("my_plugin.MyVisualizationPlugin")
        >>>
        >>> # Handle event with plugins
        >>> responses = await manager.handle_event("precision_update", data)
        >>>
        >>> # Get plugin status
        >>> status = manager.get_plugin_status()
    """

    def __init__(self, plugin_directory: Optional[str] = None):
        """
        Initialize plugin manager.

        Args:
            plugin_directory: Directory to scan for plugins
        """
        self.plugin_directory = plugin_directory
        self.plugins: Dict[str, PluginInfo] = {}
        self.event_handlers: Dict[str, List[str]] = {}  # event_type -> plugin_ids
        self.data_transformers: Dict[str, List[str]] = {}  # data_type -> plugin_ids

        self.logger = logging.getLogger(__name__)

        # Plugin security
        self.allowed_permissions = {
            "read_state",
            "write_state",
            "send_events",
            "receive_events",
            "modify_ui",
            "access_agents",
            "system_config",
        }

    async def load_plugins_from_directory(self, directory: str) -> int:
        """
        Load all plugins from directory.

        Args:
            directory: Directory containing plugins

        Returns:
            Number of plugins loaded
        """
        loaded_count = 0

        try:
            plugin_path = Path(directory)
            if not plugin_path.exists():
                self.logger.warning(f"Plugin directory not found: {directory}")
                return 0

            # Import all Python modules in directory
            for module_info in pkgutil.iter_modules([str(plugin_path)]):
                module_name = module_info.name

                try:
                    # Import module
                    spec = importlib.util.spec_from_file_location(
                        module_name, plugin_path / f"{module_name}.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find plugin classes in module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj, TUIPlugin)
                            and obj != TUIPlugin
                            and not inspect.isabstract(obj)
                        ):
                            plugin_id = f"{module_name}.{name}"
                            await self._load_plugin_class(plugin_id, obj)
                            loaded_count += 1

                except Exception as e:
                    self.logger.error(f"Error loading plugin module {module_name}: {e}")

            self.logger.info(f"Loaded {loaded_count} plugins from {directory}")

        except Exception as e:
            self.logger.error(f"Error scanning plugin directory {directory}: {e}")

        return loaded_count

    async def load_plugin(self, plugin_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load specific plugin.

        Args:
            plugin_path: Plugin class path (module.Class)
            config: Plugin configuration

        Returns:
            Success status
        """
        try:
            # Parse plugin path
            if "." not in plugin_path:
                self.logger.error(f"Invalid plugin path: {plugin_path}")
                return False

            module_name, class_name = plugin_path.rsplit(".", 1)

            # Import module
            module = importlib.import_module(module_name)

            # Get plugin class
            plugin_class = getattr(module, class_name)

            # Validate plugin class
            if not (issubclass(plugin_class, TUIPlugin) and plugin_class != TUIPlugin):
                self.logger.error(f"Invalid plugin class: {plugin_path}")
                return False

            return await self._load_plugin_class(plugin_path, plugin_class, config)

        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_path}: {e}")
            return False

    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Success status
        """
        if plugin_id not in self.plugins:
            self.logger.warning(f"Plugin not found: {plugin_id}")
            return False

        try:
            plugin_info = self.plugins[plugin_id]

            # Cleanup plugin
            if plugin_info.instance:
                await plugin_info.instance.cleanup()

            # Remove from handlers
            self._remove_plugin_handlers(plugin_id)

            # Remove from transformers
            self._remove_plugin_transformers(plugin_id)

            # Remove from plugins
            del self.plugins[plugin_id]

            self.logger.info(f"Unloaded plugin: {plugin_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False

    async def handle_event(self, event_type: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle event with appropriate plugins.

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            List of plugin responses
        """
        responses = []

        # Get plugins that handle this event type
        plugin_ids = self.event_handlers.get(event_type, [])

        for plugin_id in plugin_ids:
            if plugin_id in self.plugins:
                plugin_info = self.plugins[plugin_id]

                if plugin_info.status == PluginStatus.ACTIVE:
                    try:
                        response = await plugin_info.instance.handle_event(event_type, data)
                        if response:
                            responses.append({"plugin_id": plugin_id, "response": response})

                        # Update usage stats
                        plugin_info.usage_stats[f"events_{event_type}"] = (
                            plugin_info.usage_stats.get(f"events_{event_type}", 0) + 1
                        )

                    except Exception as e:
                        self.logger.error(f"Plugin {plugin_id} error handling {event_type}: {e}")

        return responses

    async def transform_data(
        self, data: Dict[str, Any], data_type: str, direction: str
    ) -> Dict[str, Any]:
        """
        Transform data using appropriate transformer plugins.

        Args:
            data: Data to transform
            data_type: Type of data
            direction: Transformation direction

        Returns:
            Transformed data
        """
        # Get transformers for this data type
        plugin_ids = self.data_transformers.get(data_type, [])

        transformed_data = data.copy()

        for plugin_id in plugin_ids:
            if plugin_id in self.plugins:
                plugin_info = self.plugins[plugin_id]

                if plugin_info.status == PluginStatus.ACTIVE and isinstance(
                    plugin_info.instance, TransformerPlugin
                ):
                    try:
                        transformed_data = await plugin_info.instance.transform(
                            transformed_data, direction
                        )

                        # Update usage stats
                        plugin_info.usage_stats[f"transforms_{data_type}"] = (
                            plugin_info.usage_stats.get(f"transforms_{data_type}", 0) + 1
                        )

                    except Exception as e:
                        self.logger.error(f"Plugin {plugin_id} error transforming {data_type}: {e}")

        return transformed_data

    def get_plugin_status(self) -> Dict[str, Any]:
        """
        Get status of all loaded plugins.

        Returns:
            Plugin status information
        """
        status = {
            "total_plugins": len(self.plugins),
            "active_plugins": len(
                [p for p in self.plugins.values() if p.status == PluginStatus.ACTIVE]
            ),
            "plugins": {},
        }

        for plugin_id, plugin_info in self.plugins.items():
            health = plugin_info.instance.get_health_status()

            status["plugins"][plugin_id] = {
                "type": plugin_info.metadata.plugin_type.value,
                "status": plugin_info.status.value,
                "version": plugin_info.metadata.version,
                "loaded_at": plugin_info.loaded_at.isoformat(),
                "error_message": plugin_info.error_message,
                "health": health,
                "usage_stats": plugin_info.usage_stats.copy(),
            }

        return status

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """
        Get plugins of specific type.

        Args:
            plugin_type: Type of plugin

        Returns:
            List of plugin info
        """
        return [
            plugin_info
            for plugin_info in self.plugins.values()
            if plugin_info.metadata.plugin_type == plugin_type
        ]

    def get_visualization_plugins(self) -> List[VisualizationPlugin]:
        """Get all visualization plugins."""
        return [
            plugin_info.instance
            for plugin_info in self.plugins.values()
            if (
                isinstance(plugin_info.instance, VisualizationPlugin)
                and plugin_info.status == PluginStatus.ACTIVE
            )
        ]

    def get_handler_plugins(self) -> List[HandlerPlugin]:
        """Get all handler plugins."""
        return [
            plugin_info.instance
            for plugin_info in self.plugins.values()
            if (
                isinstance(plugin_info.instance, HandlerPlugin)
                and plugin_info.status == PluginStatus.ACTIVE
            )
        ]

    async def _load_plugin_class(
        self, plugin_id: str, plugin_class: Type, config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Load plugin class and create instance."""

        try:
            # Create instance
            plugin_instance = plugin_class(config or {})

            # Get metadata
            metadata = plugin_instance.metadata

            # Check dependencies
            if not await self._check_dependencies(metadata.dependencies):
                self.logger.error(f"Plugin {plugin_id} dependencies not satisfied")
                return False

            # Check permissions
            if not self._check_permissions(metadata.permissions):
                self.logger.error(f"Plugin {plugin_id} permissions not allowed")
                return False

            # Initialize plugin
            plugin_info = PluginInfo(
                plugin_class=plugin_class,
                instance=plugin_instance,
                metadata=metadata,
                status=PluginStatus.LOADING,
                loaded_at=datetime.now(),
                config=config or {},
            )

            self.plugins[plugin_id] = plugin_info

            # Initialize plugin
            if await plugin_instance.initialize():
                plugin_info.status = PluginStatus.ACTIVE
                self.logger.info(f"Loaded plugin: {plugin_id}")

                # Register plugin handlers
                self._register_plugin_handlers(plugin_id, plugin_instance)

                # Register plugin transformers
                self._register_plugin_transformers(plugin_id, plugin_instance)

                return True
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Initialization failed"
                self.logger.error(f"Plugin {plugin_id} initialization failed")
                return False

        except Exception as e:
            if plugin_id in self.plugins:
                self.plugins[plugin_id].status = PluginStatus.ERROR
                self.plugins[plugin_id].error_message = str(e)

            self.logger.error(f"Error loading plugin {plugin_id}: {e}")
            return False

    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if plugin dependencies are satisfied."""

        for dependency in dependencies:
            if dependency not in self.plugins:
                return False

        return True

    def _check_permissions(self, permissions: List[str]) -> bool:
        """Check if plugin permissions are allowed."""

        for permission in permissions:
            if permission not in self.allowed_permissions:
                return False

        return True

    def _register_plugin_handlers(self, plugin_id: str, plugin_instance: TUIPlugin):
        """Register plugin event handlers."""

        if isinstance(plugin_instance, HandlerPlugin):
            handled_events = plugin_instance.get_handled_events()

            for event_type in handled_events:
                if event_type not in self.event_handlers:
                    self.event_handlers[event_type] = []

                self.event_handlers[event_type].append(plugin_id)

    def _register_plugin_transformers(self, plugin_id: str, plugin_instance: TUIPlugin):
        """Register plugin data transformers."""

        if isinstance(plugin_instance, TransformerPlugin):
            transform_types = plugin_instance.get_transform_types()

            for data_type in transform_types:
                if data_type not in self.data_transformers:
                    self.data_transformers[data_type] = []

                self.data_transformers[data_type].append(plugin_id)

    def _remove_plugin_handlers(self, plugin_id: str):
        """Remove plugin from event handlers."""

        for event_type in self.event_handlers:
            if plugin_id in self.event_handlers[event_type]:
                self.event_handlers[event_type].remove(plugin_id)

            # Clean up empty event types
            if not self.event_handlers[event_type]:
                del self.event_handlers[event_type]

    def _remove_plugin_transformers(self, plugin_id: str):
        """Remove plugin from data transformers."""

        for data_type in self.data_transformers:
            if plugin_id in self.data_transformers[data_type]:
                self.data_transformers[data_type].remove(plugin_id)

            # Clean up empty data types
            if not self.data_transformers[data_type]:
                del self.data_transformers[data_type]


# Example plugin implementations for documentation
class ExampleVisualizationPlugin(VisualizationPlugin):
    """Example visualization plugin."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_visualization",
            version="1.0.0",
            description="Example visualization plugin",
            author="LRS Team",
            plugin_type=PluginType.VISUALIZATION,
            tags=["example", "visualization"],
        )

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def render(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "example_chart", "data": data, "rendered_at": datetime.now().isoformat()}

    def get_supported_data_types(self) -> List[str]:
        return ["precision", "tool_execution"]
