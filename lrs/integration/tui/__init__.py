"""
TUI Integration for LRS-Agents.

Provides comprehensive bidirectional communication between opencode TUI and LRS-Agents
with advanced AI capabilities, analytics, optimization, and distributed networking.
"""

# Core Components
from .bridge import TUIBridge
from .tool import TUIInteractionTool
from .state_mirror import TUIStateMirror
from .websocket_manager import WebSocketManager
from .rest_endpoints import RESTEndpoints
from .precision_mapper import TUIPrecisionMapper
from .coordinator import TUIMultiAgentCoordinator
from .config import TUIConfigManager, TUIIntegrationConfig

# Extended Components
from .ai_assistant import TUIAIAssistant
from .analytics import AdvancedAnalyticsDashboard, MetricType, ForecastModel
from .optimizer import AgentOptimizer, OptimizationType, OptimizationAlgorithm
from .mesh_network import DistributedMeshNetwork, MessageType, NodeStatus
from .plugins import TUIPluginManager, TUIPlugin, PluginType, VisualizationPlugin
from .tests import *  # Comprehensive test suite

__all__ = [
    # Core Components
    "TUIBridge",
    "TUIInteractionTool",
    "TUIStateMirror",
    "WebSocketManager",
    "RESTEndpoints",
    "TUIPrecisionMapper",
    "TUIMultiAgentCoordinator",
    # Configuration
    "TUIConfigManager",
    "TUIIntegrationConfig",
    # Extended AI Components
    "TUIAIAssistant",
    "IntentType",
    "ConfidenceLevel",
    # Analytics and Forecasting
    "AdvancedAnalyticsDashboard",
    "MetricType",
    "ForecastModel",
    # Optimization System
    "AgentOptimizer",
    "OptimizationType",
    "OptimizationAlgorithm",
    # Distributed Networking
    "DistributedMeshNetwork",
    "MessageType",
    "NodeStatus",
    # Plugin System
    "TUIPluginManager",
    "TUIPlugin",
    "PluginType",
    "VisualizationPlugin",
    # Testing
    "tests",
]
