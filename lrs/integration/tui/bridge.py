"""
TUI Bridge: Core bidirectional communication between opencode TUI and LRS-Agents.

This component provides:
- WebSocket and REST API endpoints
- State synchronization between TUI and LRS
- Real-time event streaming
- Agent lifecycle management
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

from ..core.registry import ToolRegistry
from ..multi_agent.shared_state import SharedWorldState
from ..multi_agent.coordinator import MultiAgentCoordinator
from .websocket_manager import WebSocketManager
from .rest_endpoints import RESTEndpoints
from .state_mirror import TUIStateMirror
from .precision_mapper import TUIPrecisionMapper


@dataclass
class TUIConfig:
    """Configuration for TUI integration."""

    websocket_port: int = 8000
    rest_port: int = 8001
    event_buffer_size: int = 1000
    precision_update_interval: float = 0.1
    tool_execution_buffer: int = 50
    adaptation_alert_threshold: float = 0.4
    enable_real_time_dashboard: bool = True
    precision_history_limit: int = 1000
    tool_execution_history_limit: int = 500
    require_authentication: bool = False
    allowed_origins: List[str] = None

    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:3000", "http://localhost:8080"]


class TUIBridge:
    """
    Bidirectional bridge between opencode TUI and LRS-Agents.

    This is the main entry point for TUI integration, providing:
    - WebSocket connections for real-time communication
    - REST API for state management and agent control
    - Event streaming for precision, tool execution, and adaptations
    - State synchronization between TUI and LRS
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        shared_state: SharedWorldState,
        coordinator: Optional[MultiAgentCoordinator] = None,
        config: Optional[TUIConfig] = None,
    ):
        """
        Initialize TUI Bridge.

        Args:
            tool_registry: LRS tool registry
            shared_state: Shared world state for coordination
            coordinator: Optional multi-agent coordinator
            config: TUI configuration
        """
        self.tool_registry = tool_registry
        self.shared_state = shared_state
        self.coordinator = coordinator
        self.config = config or TUIConfig()

        # Initialize components
        self.websocket_manager = WebSocketManager(self.config)
        self.rest_endpoints = RESTEndpoints(self)
        self.state_mirror = TUIStateMirror(shared_state, self)
        self.precision_mapper = TUIPrecisionMapper()

        # Event subscribers
        self._subscribers: Dict[str, List[Callable]] = {}

        # FastAPI app for REST and WebSocket
        self.app = FastAPI(
            title="LRS-Agents TUI Bridge",
            description="Bidirectional API for opencode TUI integration",
            version="1.0.0",
        )

        self._setup_routes()
        self._setup_event_listeners()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        """Setup FastAPI routes for REST and WebSocket."""

        # WebSocket routes
        self.app.websocket("/ws/agents/{agent_id}/state")(self._websocket_agent_state)
        self.app.websocket("/ws/precision")(self._websocket_precision)
        self.app.websocket("/ws/tools/{agent_id}/executions")(self._websocket_tool_executions)
        self.app.websocket("/ws/tui/events")(self._websocket_tui_events)

        # REST routes (delegated to RESTEndpoints)
        self.app.include_router(self.rest_endpoints.router, prefix="/api/v1")

        # Health check
        self.app.get("/health")(self._health_check)

    def _setup_event_listeners(self):
        """Setup event listeners for shared state changes."""

        # Subscribe to precision updates
        self.shared_state.subscribe("precision", self._on_precision_change)

        # Subscribe to tool executions
        self.shared_state.subscribe("tool_execution", self._on_tool_execution)

        # Subscribe to adaptations
        self.shared_state.subscribe("adaptation", self._on_adaptation)

    async def _websocket_agent_state(self, websocket: WebSocket, agent_id: str):
        """WebSocket endpoint for agent state streaming."""
        await self.websocket_manager.connect(websocket, f"agent_{agent_id}")

        try:
            # Send current state
            current_state = self.shared_state.get_agent_state(agent_id)
            if current_state:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "state_snapshot",
                            "agent_id": agent_id,
                            "state": current_state,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                )

            # Keep connection alive and handle messages
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await self._handle_tui_message(agent_id, message)

        except WebSocketDisconnect:
            self.websocket_manager.disconnect(websocket, f"agent_{agent_id}")

    async def _websocket_precision(self, websocket: WebSocket):
        """WebSocket endpoint for precision updates."""
        await self.websocket_manager.connect(websocket, "precision")

        try:
            while True:
                # This will be handled by event broadcasting
                await websocket.receive_text()
        except WebSocketDisconnect:
            self.websocket_manager.disconnect(websocket, "precision")

    async def _websocket_tool_executions(self, websocket: WebSocket, agent_id: str):
        """WebSocket endpoint for tool execution streaming."""
        await self.websocket_manager.connect(websocket, f"tools_{agent_id}")

        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            self.websocket_manager.disconnect(websocket, f"tools_{agent_id}")

    async def _websocket_tui_events(self, websocket: WebSocket):
        """WebSocket endpoint for general TUI events."""
        await self.websocket_manager.connect(websocket, "tui_events")

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await self._handle_tui_command(message)
        except WebSocketDisconnect:
            self.websocket_manager.disconnect(websocket, "tui_events")

    async def _handle_tui_message(self, agent_id: str, message: Dict[str, Any]):
        """Handle incoming TUI message for specific agent."""

        message_type = message.get("type")

        if message_type == "get_state":
            state = self.shared_state.get_agent_state(agent_id)
            await self.websocket_manager.send_to_agent(
                agent_id,
                {
                    "type": "state_response",
                    "agent_id": agent_id,
                    "state": state,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        elif message_type == "update_preferences":
            if self.coordinator:
                agent = self.coordinator.get_agent(agent_id)
                if agent and hasattr(agent, "update_preferences"):
                    agent.update_preferences(message.get("preferences", {}))

    async def _handle_tui_command(self, command: Dict[str, Any]):
        """Handle general TUI commands."""

        cmd_type = command.get("type")
        request_id = command.get("request_id")

        if cmd_type == "create_agent":
            # Create new agent with TUI integration
            agent_config = command.get("config", {})
            agent_id = command.get("agent_id")

            if self.coordinator:
                agent = await self._create_tui_agent(agent_id, agent_config)
                if agent:
                    self.coordinator.register_agent(agent_id, agent)

                    await self.websocket_manager.broadcast(
                        {
                            "type": "agent_created",
                            "agent_id": agent_id,
                            "config": agent_config,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        elif cmd_type == "execute_tool":
            agent_id = command.get("agent_id")
            tool_name = command.get("tool_name")
            params = command.get("params", {})

            result = await self._execute_agent_tool(agent_id, tool_name, params)

            await self.websocket_manager.send_response(request_id, result)

        elif cmd_type == "list_agents":
            if self.coordinator:
                agents = self.coordinator.list_agents()
                await self.websocket_manager.send_response(
                    request_id, {"agents": agents, "timestamp": datetime.now().isoformat()}
                )

    async def _create_tui_agent(self, agent_id: str, config: Dict[str, Any]):
        """Create agent with TUI integration capabilities."""
        # This would integrate with specific agent creation logic
        # For now, return a placeholder
        return {"agent_id": agent_id, "config": config}

    async def _execute_agent_tool(self, agent_id: str, tool_name: str, params: Dict[str, Any]):
        """Execute tool on specific agent."""
        # This would integrate with tool execution logic
        return {
            "agent_id": agent_id,
            "tool": tool_name,
            "params": params,
            "result": "executed",
            "timestamp": datetime.now().isoformat(),
        }

    def _on_precision_change(self, agent_id: str, precision_changes: Dict[str, Any]):
        """Handle precision change event."""

        # Map precision to TUI confidence levels
        confidence_data = self.precision_mapper.precision_to_confidence(precision_changes)

        # Broadcast to WebSocket clients
        asyncio.create_task(
            self.websocket_manager.broadcast(
                "precision",
                {
                    "type": "precision_update",
                    "agent_id": agent_id,
                    "precision": precision_changes,
                    "confidence": confidence_data,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        )

        # Sync state to TUI
        asyncio.create_task(self.state_mirror.sync_precision_to_tui(agent_id, precision_changes))

    def _on_tool_execution(self, agent_id: str, execution_result: Dict[str, Any]):
        """Handle tool execution event."""

        # Broadcast execution result
        asyncio.create_task(
            self.websocket_manager.broadcast(
                f"tools_{agent_id}",
                {
                    "type": "tool_execution",
                    "agent_id": agent_id,
                    "tool": execution_result.get("tool"),
                    "success": execution_result.get("success"),
                    "prediction_error": execution_result.get("prediction_error"),
                    "timestamp": datetime.now().isoformat(),
                },
            )
        )

        # Sync to TUI state
        asyncio.create_task(
            self.state_mirror.sync_tool_execution_to_tui(agent_id, execution_result)
        )

    def _on_adaptation(self, agent_id: str, adaptation_event: Dict[str, Any]):
        """Handle adaptation event."""

        # Map adaptation to TUI alert
        alert_data = self.precision_mapper.adaptation_to_tui_alert(adaptation_event)

        # Broadcast adaptation event
        asyncio.create_task(
            self.websocket_manager.broadcast(
                "tui_events",
                {
                    "type": "adaptation",
                    "agent_id": agent_id,
                    "alert": alert_data,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        )

    async def _health_check(self):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "websocket_connections": len(self.websocket_manager.connections),
            "active_agents": len(self.shared_state.get_all_states()) if self.shared_state else 0,
        }

    async def start(self, host: str = "0.0.0.0", port: Optional[int] = None):
        """Start the TUI bridge server."""

        if port is None:
            port = self.config.websocket_port

        self.logger.info(f"Starting TUI Bridge on {host}:{port}")

        # Start background tasks
        self._start_background_tasks()

        # Start server
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    def _start_background_tasks(self):
        """Start background tasks for monitoring and synchronization."""

        # Precision monitoring task
        task = asyncio.create_task(self._precision_monitoring_loop())
        self._background_tasks.append(task)

        # State synchronization task
        task = asyncio.create_task(self._state_sync_loop())
        self._background_tasks.append(task)

    async def _precision_monitoring_loop(self):
        """Background loop for precision monitoring."""
        while True:
            try:
                # Monitor precision levels and trigger alerts
                if self.shared_state:
                    all_states = self.shared_state.get_all_states()
                    for agent_id, state in all_states.items():
                        precision = state.get("precision")
                        if (
                            precision
                            and precision.get("value", 1.0) < self.config.adaptation_alert_threshold
                        ):
                            await self.websocket_manager.broadcast(
                                "tui_events",
                                {
                                    "type": "precision_alert",
                                    "agent_id": agent_id,
                                    "precision": precision,
                                    "threshold": self.config.adaptation_alert_threshold,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )

                await asyncio.sleep(self.config.precision_update_interval)

            except Exception as e:
                self.logger.error(f"Error in precision monitoring: {e}")
                await asyncio.sleep(1)

    async def _state_sync_loop(self):
        """Background loop for state synchronization."""
        while True:
            try:
                # Periodic state synchronization
                await self.state_mirror.periodic_sync()
                await asyncio.sleep(1.0)  # Sync every second

            except Exception as e:
                self.logger.error(f"Error in state sync: {e}")
                await asyncio.sleep(1)

    def stop(self):
        """Stop the TUI bridge and cleanup resources."""

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Close WebSocket connections
        self.websocket_manager.close_all_connections()

        self.logger.info("TUI Bridge stopped")
