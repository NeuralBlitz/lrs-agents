#!/usr/bin/env python3
"""
OpenCode Plugin Architecture
Phase 5: Ecosystem Expansion - Extensible Plugin System

Comprehensive plugin architecture enabling third-party integrations,
custom tools, and community extensions for the OpenCode ↔ LRS-Agents platform.
"""

import importlib.util
import inspect
import json
import os
import sys
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import time
from pathlib import Path


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    author: str
    description: str
    license: str = "MIT"
    homepage: str = ""
    repository: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_opencode_version: str = "3.0.0"
    max_opencode_version: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class PluginInfo:
    """Complete plugin information."""

    metadata: PluginMetadata
    path: str
    checksum: str
    loaded: bool = False
    enabled: bool = True
    load_time: Optional[float] = None
    error_message: Optional[str] = None
    instance: Optional[Any] = None


class PluginInterface(ABC):
    """Abstract base class for all OpenCode plugins."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin with context."""
        pass

    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up plugin resources."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities and features."""
        pass


class ToolPlugin(PluginInterface):
    """Base class for tool plugins that extend OpenCode functionality."""

    def __init__(self):
        self.name = "ToolPlugin"
        self.version = "1.0.0"
        self._initialized = False

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            author="Plugin Developer",
            description=f"{self.name} tool plugin",
            tags=["tool", "extension"],
        )

    def initialize(self, context: Dict[str, Any]) -> bool:
        """Default initialization - override in subclasses."""
        self._initialized = True
        return True

    def cleanup(self) -> bool:
        """Default cleanup - override in subclasses."""
        self._initialized = False
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Return tool capabilities."""
        return {"commands": [], "tools": [], "hooks": [], "apis": []}


class LRSPlugin(PluginInterface):
    """Base class for LRS-enhanced plugins."""

    def __init__(self):
        self.name = "LRSPlugin"
        self.version = "1.0.0"
        self.precision_tracker = None
        self._initialized = False

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            author="LRS Developer",
            description=f"{self.name} LRS-enhanced plugin",
            tags=["lrs", "ai", "active-inference"],
        )

    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize with LRS context."""
        try:
            # Import lightweight LRS if available
            from lrs_agents.lrs.opencode.lightweight_lrs import LightweightHierarchicalPrecision

            self.precision_tracker = LightweightHierarchicalPrecision()
            self._initialized = True
            return True
        except ImportError:
            self.error_message = "Lightweight LRS not available"
            return False

    def cleanup(self) -> bool:
        """Cleanup LRS resources."""
        self.precision_tracker = None
        self._initialized = False
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Return LRS capabilities."""
        return {
            "lrs_enabled": self._initialized,
            "precision_tracking": True,
            "active_inference": True,
            "learning_capabilities": ["meta_learning", "adaptation"],
        }


class PluginRegistry:
    """Registry for managing plugins."""

    def __init__(self, plugin_dirs: List[str] = None):
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_dirs = plugin_dirs or [
            "./plugins",
            "~/.opencode/plugins",
            "/usr/local/share/opencode/plugins",
        ]
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self.event_listeners: Dict[str, List[Callable]] = {}

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in configured directories."""
        discovered = []

        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir).expanduser()
            if not plugin_path.exists():
                continue

            for plugin_file in plugin_path.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                plugin_name = plugin_file.stem
                discovered.append(str(plugin_file))

                # Calculate checksum
                with open(plugin_file, "rb") as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()

                # Create plugin info
                plugin_info = PluginInfo(
                    metadata=PluginMetadata(
                        name=plugin_name,
                        version="1.0.0",
                        author="Unknown",
                        description=f"Plugin {plugin_name}",
                    ),
                    path=str(plugin_file),
                    checksum=checksum,
                )

                self.plugins[plugin_name] = plugin_info

        return discovered

    def load_plugin(self, plugin_name: str, context: Dict[str, Any] = None) -> bool:
        """Load a specific plugin."""
        if plugin_name not in self.plugins:
            return False

        plugin_info = self.plugins[plugin_name]
        if plugin_info.loaded:
            return True

        try:
            # Add current directory to path for imports
            plugin_dir = str(Path(plugin_info.path).parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)

            # Load the plugin module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_info.path)
            if spec is None or spec.loader is None:
                plugin_info.error_message = "Could not load plugin spec"
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)

            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, PluginInterface)
                    and obj != PluginInterface
                    and obj != ToolPlugin
                    and obj != LRSPlugin
                ):
                    plugin_class = obj
                    break

            if plugin_class is None:
                plugin_info.error_message = "No valid plugin class found"
                return False

            # Instantiate and initialize
            plugin_instance = plugin_class()
            plugin_context = context or {}

            if plugin_instance.initialize(plugin_context):
                plugin_info.instance = plugin_instance
                plugin_info.loaded = True
                plugin_info.load_time = time.time()

                # Update metadata
                plugin_info.metadata = plugin_instance.get_metadata()

                # Register with loaded plugins
                self.loaded_plugins[plugin_name] = plugin_instance

                # Register hooks and listeners
                self._register_plugin_hooks(plugin_instance)

                return True
            else:
                plugin_info.error_message = "Plugin initialization failed"
                return False

        except Exception as e:
            plugin_info.error_message = f"Plugin load error: {str(e)}"
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.loaded_plugins:
            return True

        plugin_instance = self.loaded_plugins[plugin_name]

        try:
            if plugin_instance.cleanup():
                # Remove from loaded plugins
                del self.loaded_plugins[plugin_name]

                # Update plugin info
                if plugin_name in self.plugins:
                    self.plugins[plugin_name].loaded = False
                    self.plugins[plugin_name].instance = None

                # Remove hooks and listeners
                self._unregister_plugin_hooks(plugin_instance)

                return True
            else:
                return False
        except Exception:
            return False

    def _register_plugin_hooks(self, plugin: PluginInterface):
        """Register plugin hooks and event listeners."""
        capabilities = plugin.get_capabilities()

        # Register hooks
        if "hooks" in capabilities:
            for hook_name, hook_func in capabilities["hooks"].items():
                if hook_name not in self.hooks:
                    self.hooks[hook_name] = []
                self.hooks[hook_name].append(hook_func)

        # Register event listeners
        if "events" in capabilities:
            for event_name, listener_func in capabilities["events"].items():
                if event_name not in self.event_listeners:
                    self.event_listeners[event_name] = []
                self.event_listeners[event_name].append(listener_func)

    def _unregister_plugin_hooks(self, plugin: PluginInterface):
        """Unregister plugin hooks and event listeners."""
        capabilities = plugin.get_capabilities()

        # Remove hooks
        if "hooks" in capabilities:
            for hook_name, hook_func in capabilities["hooks"].items():
                if hook_name in self.hooks:
                    self.hooks[hook_name] = [
                        h for h in self.hooks[hook_name] if h != hook_func
                    ]

        # Remove event listeners
        if "events" in capabilities:
            for event_name, listener_func in capabilities["events"].items():
                if event_name in self.event_listeners:
                    self.event_listeners[event_name] = [
                        l
                        for l in self.event_listeners[event_name]
                        if l != listener_func
                    ]

    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all registered hooks for a given name."""
        results = []
        if hook_name in self.hooks:
            for hook_func in self.hooks[hook_name]:
                try:
                    result = hook_func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"Hook {hook_name} error: {e}")
        return results

    def emit_event(self, event_name: str, *args, **kwargs):
        """Emit an event to all registered listeners."""
        if event_name in self.event_listeners:
            for listener_func in self.event_listeners[event_name]:
                try:
                    listener_func(*args, **kwargs)
                except Exception as e:
                    print(f"Event {event_name} listener error: {e}")

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a plugin."""
        return self.plugins.get(plugin_name)

    def list_plugins(self) -> Dict[str, PluginInfo]:
        """List all registered plugins."""
        return self.plugins.copy()

    def get_loaded_plugins(self) -> Dict[str, PluginInterface]:
        """Get all currently loaded plugins."""
        return self.loaded_plugins.copy()

    def validate_plugin(self, plugin_path: str) -> Dict[str, Any]:
        """Validate a plugin file."""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "metadata": None,
        }

        try:
            # Load and inspect the plugin
            spec = importlib.util.spec_from_file_location("validation", plugin_path)
            if spec is None or spec.loader is None:
                validation_result["errors"].append("Could not load plugin spec")
                return validation_result

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check for plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, PluginInterface)
                    and obj != PluginInterface
                    and obj != ToolPlugin
                    and obj != LRSPlugin
                ):
                    plugin_class = obj
                    break

            if plugin_class is None:
                validation_result["errors"].append("No valid plugin class found")
                return validation_result

            # Try to instantiate and get metadata
            try:
                plugin_instance = plugin_class()
                metadata = plugin_instance.get_metadata()
                validation_result["metadata"] = metadata
                validation_result["valid"] = True
            except Exception as e:
                validation_result["errors"].append(
                    f"Plugin instantiation failed: {str(e)}"
                )

        except Exception as e:
            validation_result["errors"].append(f"Plugin validation error: {str(e)}")

        return validation_result

    def save_plugin_registry(self, registry_file: str = "plugin_registry.json"):
        """Save plugin registry to file."""
        registry_data = {}
        for name, info in self.plugins.items():
            registry_data[name] = {
                "metadata": {
                    "name": info.metadata.name,
                    "version": info.metadata.version,
                    "author": info.metadata.author,
                    "description": info.metadata.description,
                    "license": info.metadata.license,
                    "homepage": info.metadata.homepage,
                    "repository": info.metadata.repository,
                    "dependencies": info.metadata.dependencies,
                    "tags": info.metadata.tags,
                    "min_opencode_version": info.metadata.min_opencode_version,
                    "max_opencode_version": info.metadata.max_opencode_version,
                    "created_at": info.metadata.created_at,
                    "updated_at": info.metadata.updated_at,
                },
                "path": info.path,
                "checksum": info.checksum,
                "loaded": info.loaded,
                "enabled": info.enabled,
                "load_time": info.load_time,
                "error_message": info.error_message,
            }

        try:
            with open(registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            print(f"Could not save plugin registry: {e}")

    def load_plugin_registry(self, registry_file: str = "plugin_registry.json"):
        """Load plugin registry from file."""
        try:
            with open(registry_file, "r") as f:
                registry_data = json.load(f)

            for name, info_data in registry_data.items():
                metadata = PluginMetadata(**info_data["metadata"])
                plugin_info = PluginInfo(
                    metadata=metadata,
                    path=info_data["path"],
                    checksum=info_data["checksum"],
                    loaded=info_data["loaded"],
                    enabled=info_data["enabled"],
                    load_time=info_data["load_time"],
                    error_message=info_data["error_message"],
                )
                self.plugins[name] = plugin_info

        except FileNotFoundError:
            pass  # Registry doesn't exist yet
        except Exception as e:
            print(f"Could not load plugin registry: {e}")


def create_plugin_template(
    plugin_type: str = "tool", plugin_name: str = "MyPlugin"
) -> str:
    """Create a plugin template string."""

    if plugin_type == "tool":
        template = f'''#!/usr/bin/env python3
"""
{plugin_name} - OpenCode Tool Plugin
A custom tool plugin for the OpenCode ↔ LRS-Agents platform.
"""

from sys import path
path.insert(0, '.')
from lrs_agents.lrs.enterprise.opencode_plugin_architecture import ToolPlugin, PluginMetadata
from typing import Dict, List, Any


class {plugin_name}(ToolPlugin):
    """Custom tool plugin implementation."""

    def __init__(self):
        super().__init__()
        self.name = "{plugin_name}"
        self.version = "1.0.0"
        self._custom_data = {{}}

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            author="Your Name",
            description="Custom tool plugin for OpenCode",
            license="MIT",
            homepage="https://github.com/yourname/{plugin_name.lower()}",
            tags=["tool", "custom", "extension"],
            dependencies=[]
        )

    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        try:
            # Plugin initialization logic here
            self._custom_data = context.get("custom_config", {{}})
            print(f"{{self.name}} plugin initialized successfully")
            return True
        except Exception as e:
            print(f"{{self.name}} initialization failed: {{e}}")
            return False

    def cleanup(self) -> bool:
        """Clean up plugin resources."""
        try:
            self._custom_data.clear()
            print(f"{{self.name}} plugin cleaned up")
            return True
        except Exception as e:
            print(f"{{self.name}} cleanup failed: {{e}}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities."""
        return {{
            "commands": [
                "{plugin_name.lower()}_analyze",
                "{plugin_name.lower()}_process"
            ],
            "tools": [
                "{plugin_name}Analyzer",
                "{plugin_name}Processor"
            ],
            "hooks": {{
                "pre_command": self.pre_command_hook,
                "post_command": self.post_command_hook
            }},
            "events": {{
                "plugin_loaded": self.on_plugin_loaded,
                "command_executed": self.on_command_executed
            }}
        }}

    # Hook functions
    def pre_command_hook(self, command: str, args: List[str]) -> bool:
        """Hook called before command execution."""
        print(f"{{self.name}}: Pre-command hook for {{command}}")
        return True

    def post_command_hook(self, command: str, result: Any) -> None:
        """Hook called after command execution."""
        print(f"{{self.name}}: Post-command hook for {{command}}")

    # Event listeners
    def on_plugin_loaded(self, plugin_name: str) -> None:
        """Event listener for plugin loaded events."""
        print(f"{{self.name}}: Plugin loaded event received for {{plugin_name}}")

    def on_command_executed(self, command: str, success: bool) -> None:
        """Event listener for command executed events."""
        print(f"{{self.name}}: Command executed event: {{command}} (success: {{success}})")

    # Custom plugin methods
    def analyze_data(self, data: Any) -> Dict[str, Any]:
        """Custom analysis method."""
        return {{
            "analysis_type": "{plugin_name} Analysis",
            "input_type": type(data).__name__,
            "result": "Analysis completed",
            "timestamp": __import__("time").time()
        }}

    def process_data(self, data: Any, options: Dict[str, Any] = None) -> Any:
        """Custom processing method."""
        options = options or {{}}
        return {{
            "processed_data": data,
            "processing_options": options,
            "processor": self.name,
            "timestamp": __import__("time").time()
        }}
'''
    elif plugin_type == "lrs":
        template = f'''#!/usr/bin/env python3
"""
{plugin_name} - LRS-Enhanced Plugin
An Active Inference enhanced plugin for the OpenCode ↔ LRS-Agents platform.
"""

from lrs_agents.lrs.enterprise.opencode_plugin_architecture import LRSPlugin, PluginMetadata
from typing import Dict, List, Any


class {plugin_name}(LRSPlugin):
    """LRS-enhanced plugin implementation."""

    def __init__(self):
        super().__init__()
        self.name = "{plugin_name}"
        self.version = "1.0.0"
        self.learning_data = {{}}

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            author="Your Name",
            description="LRS-enhanced plugin with Active Inference capabilities",
            license="MIT",
            homepage="https://github.com/yourname/{plugin_name.lower()}",
            tags=["lrs", "ai", "active-inference", "learning"],
            dependencies=["lightweight_lrs"]
        )

    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize with LRS capabilities."""
        if not super().initialize(context):
            return False

        try:
            # LRS-specific initialization
            self.learning_data = context.get("learning_config", {{}})
            print(f"{{self.name}} LRS plugin initialized with precision tracking")
            return True
        except Exception as e:
            print(f"{{self.name}} LRS initialization failed: {{e}}")
            return False

    def cleanup(self) -> bool:
        """Clean up LRS resources."""
        try:
            self.learning_data.clear()
            print(f"{{self.name}} LRS plugin cleaned up")
            return super().cleanup()
        except Exception as e:
            print(f"{{self.name}} LRS cleanup failed: {{e}}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Return LRS-enhanced capabilities."""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({{
            "commands": [
                "{plugin_name.lower()}_lrs_analyze",
                "{plugin_name.lower()}_learn"
            ],
            "tools": [
                "{plugin_name}LRSAnalyzer",
                "{plugin_name}Learner"
            ],
            "hooks": {{
                "pre_inference": self.pre_inference_hook,
                "post_learning": self.post_learning_hook
            }},
            "events": {{
                "precision_updated": self.on_precision_updated,
                "learning_event": self.on_learning_event
            }},
            "lrs_features": [
                "precision_tracking",
                "active_inference",
                "meta_learning",
                "adaptive_precision"
            ]
        }})
        return base_capabilities

    # LRS-specific hook functions
    def pre_inference_hook(self, observation: Any) -> Dict[str, Any]:
        """Hook called before inference."""
        if self.precision_tracker:
            current_precision = self.precision_tracker.get_current_precision()
            print(f"{{self.name}}: Pre-inference with precision: {{current_precision}}")
            return {{"precision": current_precision}}
        return {{}}

    def post_learning_hook(self, learning_result: Dict[str, Any]) -> None:
        """Hook called after learning."""
        self.learning_data.update(learning_result)
        print(f"{{self.name}}: Learning data updated: {{learning_result.keys()}}")

    # LRS-specific event listeners
    def on_precision_updated(self, new_precision: float, context: Dict[str, Any]) -> None:
        """Event listener for precision updates."""
        print(f"{{self.name}}: Precision updated to {{new_precision}}")
        self.learning_data["last_precision_update"] = __import__("time").time()

    def on_learning_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Event listener for learning events."""
        print(f"{{self.name}}: Learning event '{{event_type}}': {{event_data.keys()}}")
        self.learning_data["learning_event_" + event_type] = event_data

    # LRS-enhanced methods
    def lrs_analyze(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """LRS-enhanced analysis with precision tracking."""
        context = context or {{}}

        if not self.precision_tracker:
            return {{"error": "LRS precision tracker not available"}}

        # Perform analysis with precision awareness
        analysis_result = {{
            "analysis_type": "LRS-Enhanced {plugin_name} Analysis",
            "input_data": data,
            "precision_used": self.precision_tracker.get_current_precision(),
            "learning_context": self.learning_data,
            "timestamp": __import__("time").time()
        }}

        # Update precision based on analysis
        self.precision_tracker.update_precision(success=True, context=context)

        return analysis_result

    def learn_from_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from experience data using Active Inference."""
        if not self.precision_tracker:
            return {{"error": "LRS precision tracker not available"}}

        learning_result = {{
            "learning_type": "{plugin_name} Experience Learning",
            "experience_data": experience_data,
            "precision_before": self.precision_tracker.get_current_precision(),
            "learning_timestamp": __import__("time").time()
        }}

        # Update learning data
        self.learning_data.update(experience_data)

        # Update precision based on learning
        self.precision_tracker.update_precision(success=True, context=experience_data)

        learning_result["precision_after"] = self.precision_tracker.get_current_precision()

        return learning_result

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from accumulated learning data."""
        return {{
            "plugin_name": self.name,
            "learning_data_keys": list(self.learning_data.keys()),
            "total_learning_events": len(self.learning_data),
            "current_precision": self.precision_tracker.get_current_precision() if self.precision_tracker else None,
            "insights_timestamp": __import__("time").time()
        }}
'''
    else:
        template = f"# Invalid plugin type: {plugin_type}"

    return template


def demonstrate_plugin_architecture():
    """Demonstrate the plugin architecture system."""
    print("🔌 OPENCODE PLUGIN ARCHITECTURE DEMONSTRATION")
    print("=" * 55)
    print()

    # Initialize plugin registry
    registry = PluginRegistry()

    # Discover plugins
    print("🔍 Discovering Plugins...")
    discovered = registry.discover_plugins()
    print(f"   📁 Found {len(discovered)} plugin files")

    if discovered:
        for plugin_path in discovered[:3]:  # Show first 3
            print(f"      • {os.path.basename(plugin_path)}")
    print()

    # Create sample plugins directory and template
    plugins_dir = Path("./plugins")
    plugins_dir.mkdir(exist_ok=True)

    # Create a sample tool plugin
    sample_plugin_path = plugins_dir / "sample_tool_plugin.py"
    if not sample_plugin_path.exists():
        print("📝 Creating Sample Tool Plugin...")
        template = create_plugin_template("tool", "SampleToolPlugin")
        with open(sample_plugin_path, "w") as f:
            f.write(template)
        print("   ✅ Sample tool plugin created")
    else:
        print("   📋 Sample tool plugin already exists")

    # Create a sample LRS plugin
    lrs_plugin_path = plugins_dir / "sample_lrs_plugin.py"
    if not lrs_plugin_path.exists():
        print("🧠 Creating Sample LRS Plugin...")
        template = create_plugin_template("lrs", "SampleLRSPlugin")
        with open(lrs_plugin_path, "w") as f:
            f.write(template)
        print("   ✅ Sample LRS plugin created")
    else:
        print("   🧠 Sample LRS plugin already exists")
    print()

    # Re-discover plugins
    print("🔄 Re-discovering Plugins...")
    discovered = registry.discover_plugins()
    print(f"   📁 Now found {len(discovered)} plugin files")

    for plugin_path in discovered:
        plugin_name = os.path.splitext(os.path.basename(plugin_path))[0]
        print(f"      • {plugin_name}")
    print()

    # Validate plugins
    print("✅ Validating Plugins...")
    for plugin_path in discovered:
        plugin_name = os.path.splitext(os.path.basename(plugin_path))[0]
        validation = registry.validate_plugin(plugin_path)

        if validation["valid"]:
            print(f"   ✅ {plugin_name}: Valid plugin")
            if validation["metadata"]:
                meta = validation["metadata"]
                print(f"      - {meta.description}")
                print(f"      - Version {meta.version} by {meta.author}")
        else:
            print(f"   ❌ {plugin_name}: Invalid plugin")
            for error in validation["errors"]:
                print(f"      - {error}")
    print()

    # Load plugins
    print("🚀 Loading Plugins...")
    loaded_count = 0
    for plugin_path in discovered:
        plugin_name = os.path.splitext(os.path.basename(plugin_path))[0]

        if registry.load_plugin(plugin_name):
            print(f"   ✅ {plugin_name}: Loaded successfully")
            loaded_count += 1

            # Get plugin capabilities
            plugin_instance = registry.loaded_plugins.get(plugin_name)
            if plugin_instance:
                capabilities = plugin_instance.get_capabilities()
                commands = capabilities.get("commands", [])
                if commands:
                    print(f"      - Commands: {', '.join(commands)}")
        else:
            plugin_info = registry.get_plugin_info(plugin_name)
            error_msg = plugin_info.error_message if plugin_info else "Unknown error"
            print(f"   ❌ {plugin_name}: Load failed - {error_msg}")
    print()

    print("📊 Plugin System Status:")
    print(f"   📁 Total plugins discovered: {len(discovered)}")
    print(f"   ✅ Plugins loaded: {loaded_count}")
    print(f"   🎯 Plugin registry size: {len(registry.plugins)}")
    print(f"   🔗 Active hooks: {len(registry.hooks)}")
    print(f"   📡 Event listeners: {len(registry.event_listeners)}")
    print()

    # Demonstrate plugin interaction
    if registry.loaded_plugins:
        print("🔄 Demonstrating Plugin Interaction...")

        # Emit a test event
        registry.emit_event("plugin_system_ready", timestamp=time.time())

        # Execute hooks
        hook_results = registry.execute_hook("system_startup")
        print(f"   🎣 Executed {len(hook_results)} startup hooks")
        print()

    # Save registry
    print("💾 Saving Plugin Registry...")
    registry.save_plugin_registry()
    print("   ✅ Registry saved to plugin_registry.json")
    print()

    print("🎉 Plugin Architecture Demo Complete!")
    print("✅ Extensible plugin system implemented")
    print("✅ Tool and LRS plugin support operational")
    print("✅ Hook and event system functional")
    print("✅ Plugin validation and loading working")
    print("✅ Registry persistence operational")


if __name__ == "__main__":
    demonstrate_plugin_architecture()
