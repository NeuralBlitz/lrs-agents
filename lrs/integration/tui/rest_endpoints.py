"""
REST Endpoints: HTTP API for TUI integration.

This component provides RESTful API endpoints for TUI communication,
complementing the real-time WebSocket interface with state management
and agent control capabilities.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from fastapi import APIRouter, HTTPException, Query, Path, Body
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback types if FastAPI not available
    APIRouter = Any
    HTTPException = Exception
    Query = Any
    Path = Any
    Body = Any
    BaseModel = object
    Field = Any
    StreamingResponse = Any


# Pydantic models for request/response validation
class AgentCreateRequest(BaseModel):
    """Request model for agent creation."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Type of agent to create")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    tools: List[str] = Field(default_factory=list, description="List of tools for agent")


class AgentUpdateRequest(BaseModel):
    """Request model for agent state updates."""

    state: Dict[str, Any] = Field(..., description="State updates to apply")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution."""

    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    timeout: Optional[float] = Field(30.0, description="Execution timeout in seconds")


class TUIInteractionRequest(BaseModel):
    """Request model for TUI interactions."""

    action: str = Field(..., description="Type of interaction")
    agent_id: Optional[str] = Field(None, description="Target agent ID")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Interaction parameters")


class StateQuery(BaseModel):
    """Query model for state filtering."""

    agent_ids: Optional[List[str]] = Field(None, description="Filter by agent IDs")
    state_keys: Optional[List[str]] = Field(None, description="Filter by state keys")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range filter")


class RESTEndpoints:
    """
    REST API endpoints for TUI integration.

    Provides comprehensive HTTP API for:
    - Agent lifecycle management
    - State queries and updates
    - Tool execution
    - TUI interactions
    - System information and metrics
    - Event streaming (Server-Sent Events)

    Examples:
        >>> endpoints = RESTEndpoints(tui_bridge)
        >>> app.include_router(endpoints.router, prefix="/api/v1")
        >>>
        >>> # Create agent
        >>> POST /api/v1/agents
        >>> {"agent_id": "agent_1", "agent_type": "lrs", "config": {...}}
        >>>
        >>> # Get agent state
        >>> GET /api/v1/agents/agent_1/state
        >>>
        >>> # Execute tool
        >>> POST /api/v1/agents/agent_1/tools/execute
        >>> {"tool_name": "search", "parameters": {...}}
    """

    def __init__(self, tui_bridge):
        """
        Initialize REST endpoints.

        Args:
            tui_bridge: TUIBridge instance
        """
        self.tui_bridge = tui_bridge
        self.logger = logging.getLogger(__name__)

        # Setup router
        self.router = APIRouter(prefix="/api/v1", tags=["TUI Integration"])

        # Setup routes
        self._setup_agent_routes()
        self._setup_state_routes()
        self._setup_tool_routes()
        self._setup_tui_routes()
        self._setup_system_routes()

    def _setup_agent_routes(self):
        """Setup agent management routes."""

        @self.router.get("/agents", response_model=List[Dict[str, Any]])
        async def list_agents():
            """List all active agents."""
            try:
                if self.tui_bridge.coordinator:
                    agents = self.tui_bridge.coordinator.list_agents()
                else:
                    agents = list(self.tui_bridge.shared_state.get_all_states().keys())

                # Get detailed agent information
                agent_details = []
                for agent_id in agents:
                    state = self.tui_bridge.shared_state.get_agent_state(agent_id)
                    agent_details.append(
                        {
                            "agent_id": agent_id,
                            "state": state,
                            "last_update": state.get("last_update") if state else None,
                        }
                    )

                return agent_details

            except Exception as e:
                self.logger.error(f"Error listing agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/agents", response_model=Dict[str, Any])
        async def create_agent(request: AgentCreateRequest):
            """Create new agent with TUI integration."""
            try:
                # Check if agent already exists
                existing_state = self.tui_bridge.shared_state.get_agent_state(request.agent_id)
                if existing_state:
                    raise HTTPException(
                        status_code=409, detail=f"Agent {request.agent_id} already exists"
                    )

                # Create agent through bridge
                agent = await self.tui_bridge._create_tui_agent(
                    request.agent_id,
                    {
                        "agent_type": request.agent_type,
                        "config": request.config,
                        "tools": request.tools,
                    },
                )

                if not agent:
                    raise HTTPException(status_code=500, detail="Failed to create agent")

                # Register with coordinator
                if self.tui_bridge.coordinator:
                    self.tui_bridge.coordinator.register_agent(request.agent_id, agent)

                # Initialize agent state
                self.tui_bridge.shared_state.update(
                    request.agent_id,
                    {
                        "agent_type": request.agent_type,
                        "config": request.config,
                        "tools": request.tools,
                        "created_at": datetime.now().isoformat(),
                        "status": "active",
                    },
                )

                return {
                    "agent_id": request.agent_id,
                    "status": "created",
                    "timestamp": datetime.now().isoformat(),
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error creating agent {request.agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/agents/{agent_id}", response_model=Dict[str, Any])
        async def get_agent(agent_id: str = Path(..., description="Agent ID")):
            """Get detailed information about specific agent."""
            try:
                state = self.tui_bridge.shared_state.get_agent_state(agent_id)

                if not state:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

                # Add TUI state information
                tui_state = self.tui_bridge.state_mirror.get_tui_state(agent_id)

                return {
                    "agent_id": agent_id,
                    "lrs_state": state,
                    "tui_state": tui_state,
                    "last_sync": tui_state.get("last_tui_sync") if tui_state else None,
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting agent {agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.put("/agents/{agent_id}/state", response_model=Dict[str, Any])
        async def update_agent_state(
            request: AgentUpdateRequest, agent_id: str = Path(..., description="Agent ID")
        ):
            """Update agent state."""
            try:
                # Verify agent exists
                existing_state = self.tui_bridge.shared_state.get_agent_state(agent_id)
                if not existing_state:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

                # Update LRS state
                self.tui_bridge.shared_state.update(agent_id, request.state)

                # Update TUI state if preferences provided
                if request.preferences:
                    for key, value in request.preferences.items():
                        self.tui_bridge.state_mirror.set_user_preference(key, value)

                return {
                    "agent_id": agent_id,
                    "updated": True,
                    "timestamp": datetime.now().isoformat(),
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating agent {agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.delete("/agents/{agent_id}", response_model=Dict[str, Any])
        async def delete_agent(agent_id: str = Path(..., description="Agent ID")):
            """Delete agent."""
            try:
                # Verify agent exists
                existing_state = self.tui_bridge.shared_state.get_agent_state(agent_id)
                if not existing_state:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

                # Unregister from coordinator
                if self.tui_bridge.coordinator:
                    self.tui_bridge.coordinator.unregister_agent(agent_id)

                # Clear state (shared state doesn't have delete, so we mark as deleted)
                self.tui_bridge.shared_state.update(
                    agent_id, {"status": "deleted", "deleted_at": datetime.now().isoformat()}
                )

                return {
                    "agent_id": agent_id,
                    "deleted": True,
                    "timestamp": datetime.now().isoformat(),
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting agent {agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_state_routes(self):
        """Setup state management routes."""

        @self.router.get("/agents/{agent_id}/precision", response_model=Dict[str, Any])
        async def get_agent_precision(agent_id: str = Path(..., description="Agent ID")):
            """Get agent precision information."""
            try:
                state = self.tui_bridge.shared_state.get_agent_state(agent_id)

                if not state:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

                precision = state.get("precision", {})

                return {
                    "agent_id": agent_id,
                    "precision": precision,
                    "confidence_level": self.tui_bridge.precision_mapper.precision_to_confidence(
                        precision
                    ),
                    "timestamp": datetime.now().isoformat(),
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting precision for {agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/agents/{agent_id}/tools", response_model=List[Dict[str, Any]])
        async def get_agent_tools(agent_id: str = Path(..., description="Agent ID")):
            """Get available tools for agent."""
            try:
                state = self.tui_bridge.shared_state.get_agent_state(agent_id)

                if not state:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

                # Get tools from registry
                tools = []
                if hasattr(self.tui_bridge.tool_registry, "list_tools"):
                    tool_names = self.tui_bridge.tool_registry.list_tools()

                    for tool_name in tool_names:
                        tool_info = {
                            "name": tool_name,
                            "description": f"Tool: {tool_name}",
                            "available": True,
                        }
                        tools.append(tool_info)

                return tools

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting tools for {agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/agents/{agent_id}/history", response_model=List[Dict[str, Any]])
        async def get_agent_history(
            agent_id: str = Path(..., description="Agent ID"),
            limit: int = Query(100, description="Maximum number of records"),
            event_type: Optional[str] = Query(None, description="Filter by event type"),
        ):
            """Get agent execution history."""
            try:
                state = self.tui_bridge.shared_state.get_agent_state(agent_id)

                if not state:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

                # Get history from state mirror
                history = self.tui_bridge.state_mirror.get_state_history(agent_id, limit)

                # Filter by event type if specified
                if event_type:
                    history = [
                        record for record in history if event_type in record.get("state_data", {})
                    ]

                return history

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting history for {agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_tool_routes(self):
        """Setup tool execution routes."""

        @self.router.post("/agents/{agent_id}/tools/execute", response_model=Dict[str, Any])
        async def execute_tool(
            request: ToolExecutionRequest, agent_id: str = Path(..., description="Agent ID")
        ):
            """Execute tool on agent."""
            try:
                # Verify agent exists
                state = self.tui_bridge.shared_state.get_agent_state(agent_id)
                if not state:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

                # Execute tool through bridge
                result = await self.tui_bridge._execute_agent_tool(
                    agent_id, request.tool_name, request.parameters
                )

                return {
                    "agent_id": agent_id,
                    "tool_name": request.tool_name,
                    "parameters": request.parameters,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error executing tool {request.tool_name} on {agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/tools", response_model=List[Dict[str, Any]])
        async def list_tools():
            """List all available tools."""
            try:
                tools = []

                if hasattr(self.tui_bridge.tool_registry, "list_tools"):
                    tool_names = self.tui_bridge.tool_registry.list_tools()

                    for tool_name in tool_names:
                        tool_info = {
                            "name": tool_name,
                            "description": f"Tool: {tool_name}",
                            "schema": {},  # Would need to get actual schema
                        }
                        tools.append(tool_info)

                return tools

            except Exception as e:
                self.logger.error(f"Error listing tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_tui_routes(self):
        """Setup TUI interaction routes."""

        @self.router.post("/tui/interaction", response_model=Dict[str, Any])
        async def tui_interaction(request: TUIInteractionRequest):
            """Handle TUI interaction request."""
            try:
                # Create interaction state
                interaction_state = {
                    "action": request.action,
                    "agent_id": request.agent_id,
                    **request.parameters,
                }

                # This would typically use the TUIInteractionTool
                # For now, handle basic interactions

                if request.action == "refresh_state":
                    # Trigger state refresh
                    all_states = self.tui_bridge.shared_state.get_all_states()

                    return {
                        "action": request.action,
                        "states": all_states,
                        "timestamp": datetime.now().isoformat(),
                    }

                elif request.action == "get_preferences":
                    # Get user preferences
                    preferences = {}
                    # This would get from state mirror

                    return {
                        "action": request.action,
                        "preferences": preferences,
                        "timestamp": datetime.now().isoformat(),
                    }

                else:
                    return {
                        "action": request.action,
                        "status": "received",
                        "timestamp": datetime.now().isoformat(),
                    }

            except Exception as e:
                self.logger.error(f"Error handling TUI interaction {request.action}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/tui/events")
        async def stream_tui_events():
            """Stream TUI events using Server-Sent Events."""

            async def event_stream():
                """Generate event stream."""
                try:
                    while True:
                        # Get recent events from state mirror
                        # This is a simplified implementation
                        event = {"type": "heartbeat", "timestamp": datetime.now().isoformat()}

                        yield f"data: {json.dumps(event)}\n\n"

                        # Wait before next event
                        import asyncio

                        await asyncio.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error in event stream: {e}")

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

    def _setup_system_routes(self):
        """Setup system information routes."""

        @self.router.get("/system/info", response_model=Dict[str, Any])
        async def get_system_info():
            """Get system information."""
            try:
                info = {
                    "bridge_status": "active",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "active_agents": len(self.tui_bridge.shared_state.get_all_states()),
                    "websocket_connections": len(self.tui_bridge.websocket_manager.connections),
                }

                # Add WebSocket metrics
                ws_metrics = self.tui_bridge.websocket_manager.get_metrics()
                info["websocket_metrics"] = ws_metrics

                return info

            except Exception as e:
                self.logger.error(f"Error getting system info: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/system/health")
        async def health_check():
            """Health check endpoint."""
            try:
                # Basic health checks
                health = {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "checks": {
                        "shared_state": "ok",
                        "websocket_manager": "ok",
                        "state_mirror": "ok",
                    },
                }

                # Check if background tasks are running
                if not self.tui_bridge._background_tasks:
                    health["status"] = "degraded"
                    health["checks"]["background_tasks"] = "missing"

                return health

            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                raise HTTPException(status_code=500, detail=str(e))
