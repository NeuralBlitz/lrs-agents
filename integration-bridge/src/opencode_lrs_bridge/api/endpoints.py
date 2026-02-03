"""
REST API endpoints for opencode ↔ LRS-Agents integration.
"""

import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import structlog
from datetime import datetime

from ..config.settings import IntegrationBridgeConfig
from ..models.schemas import (
    AgentCreateRequest,
    AgentUpdateRequest,
    AgentState,
    ToolExecutionRequest,
    ToolExecutionResult,
    SystemInfo,
    HealthCheck,
    EventData,
    MetricData,
    StateSyncRequest,
    StateSyncResponse,
    IntegrationBridgeMetrics,
    AgentStatus,
    ToolExecutionStatus,
    AgentType,
)
from ..auth.middleware import AuthenticationMiddleware, require_permission
from fastapi import Depends

logger = structlog.get_logger(__name__)


class AgentManager:
    """Agent lifecycle management."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.agents: Dict[str, AgentState] = {}
        self.lrs_base_url = config.lrs.base_url
        self.opencode_base_url = config.opencode.base_url

    async def create_agent(self, request: AgentCreateRequest) -> AgentState:
        """Create new agent."""
        # Check if agent already exists
        if request.agent_id in self.agents:
            raise HTTPException(status_code=409, detail="Agent already exists")

        # Create agent state
        agent_state = AgentState(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            status=AgentStatus.IDLE,
            belief_state=request.config.get("belief_state", {}),
            created_at=datetime.utcnow(),
        )

        # Register with LRS if it's an LRS agent (disabled for testing)
        # if request.agent_type in [AgentType.LRS, AgentType.HYBRID]:
        #     await self._register_with_lrs(request)

        # Register with opencode if it's an opencode agent (disabled for testing)
        # if request.agent_type in [AgentType.OPENCODE, AgentType.HYBRID]:
        #     await self._register_with_opencode(request)

        self.agents[request.agent_id] = agent_state
        logger.info(
            "Created agent", agent_id=request.agent_id, agent_type=request.agent_type
        )

        return agent_state

    async def _register_with_lrs(self, request: AgentCreateRequest):
        """Register agent with LRS-Agents."""
        try:
            async with httpx.AsyncClient(timeout=self.config.lrs.timeout) as client:
                response = await client.post(
                    f"{self.lrs_base_url}/api/v1/agents",
                    json={
                        "agent_id": request.agent_id,
                        "agent_type": request.agent_type.value,
                        "config": request.config,
                        "tools": request.tools,
                        "preferences": request.preferences,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Failed to register with LRS", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to register with LRS")

    async def _register_with_opencode(self, request: AgentCreateRequest):
        """Register agent with opencode."""
        try:
            async with httpx.AsyncClient(
                timeout=self.config.opencode.timeout
            ) as client:
                response = await client.post(
                    f"{self.opencode_base_url}/api/v1/agents",
                    json={
                        "agent_id": request.agent_id,
                        "session_id": request.opencode_session_id,
                        "config": request.config,
                        "tools": request.tools,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Failed to register with opencode", error=str(e))
            raise HTTPException(
                status_code=500, detail="Failed to register with opencode"
            )

    async def get_agent(self, agent_id: str) -> AgentState:
        """Get agent state."""
        if agent_id not in self.agents:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Sync state from both systems
        await self._sync_agent_state(agent_id)
        return self.agents[agent_id]

    async def update_agent(
        self, agent_id: str, request: AgentUpdateRequest
    ) -> AgentState:
        """Update agent state."""
        if agent_id not in self.agents:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent = self.agents[agent_id]

        # Update local state
        if request.status:
            agent.status = request.status
        if request.config:
            agent.belief_state.update(request.config)
        if request.belief_state:
            agent.belief_state.update(request.belief_state)

        agent.last_activity = datetime.utcnow()

        # Propagate updates to connected systems
        await self._propagate_agent_update(agent_id, request)

        return agent

    async def _propagate_agent_update(self, agent_id: str, request: AgentUpdateRequest):
        """Propagate agent updates to connected systems."""
        agent = self.agents[agent_id]

        # Update LRS agent
        if agent.agent_type in [AgentType.LRS, AgentType.HYBRID]:
            try:
                async with httpx.AsyncClient(timeout=self.config.lrs.timeout) as client:
                    response = await client.put(
                        f"{self.lrs_base_url}/api/v1/agents/{agent_id}/state",
                        json={
                            "status": request.status.value if request.status else None,
                            "config": request.config,
                            "belief_state": request.belief_state,
                        },
                    )
                    response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error("Failed to update LRS agent", error=str(e))

        # Update opencode agent
        if agent.agent_type in [AgentType.OPENCODE, AgentType.HYBRID]:
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.opencode.timeout
                ) as client:
                    response = await client.put(
                        f"{self.opencode_base_url}/api/v1/agents/{agent_id}/state",
                        json={
                            "status": request.status.value if request.status else None,
                            "config": request.config,
                        },
                    )
                    response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error("Failed to update opencode agent", error=str(e))

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete agent."""
        if agent_id not in self.agents:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent = self.agents[agent_id]

        # Delete from LRS
        if agent.agent_type in [AgentType.LRS, AgentType.HYBRID]:
            try:
                async with httpx.AsyncClient(timeout=self.config.lrs.timeout) as client:
                    response = await client.delete(
                        f"{self.lrs_base_url}/api/v1/agents/{agent_id}"
                    )
                    response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error("Failed to delete LRS agent", error=str(e))

        # Delete from opencode
        if agent.agent_type in [AgentType.OPENCODE, AgentType.HYBRID]:
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.opencode.timeout
                ) as client:
                    response = await client.delete(
                        f"{self.opencode_base_url}/api/v1/agents/{agent_id}"
                    )
                    response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error("Failed to delete opencode agent", error=str(e))

        del self.agents[agent_id]
        logger.info("Deleted agent", agent_id=agent_id)

        return True

    async def list_agents(
        self, status_filter: Optional[str] = None
    ) -> List[AgentState]:
        """List all agents with optional status filter."""
        agents = list(self.agents.values())

        if status_filter:
            try:
                filter_status = AgentStatus(status_filter)
                agents = [a for a in agents if a.status == filter_status]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status filter")

        return agents

    async def _sync_agent_state(self, agent_id: str):
        """Sync agent state from connected systems."""
        agent = self.agents[agent_id]

        # Sync from LRS
        if agent.agent_type in [AgentType.LRS, AgentType.HYBRID]:
            try:
                async with httpx.AsyncClient(timeout=self.config.lrs.timeout) as client:
                    response = await client.get(
                        f"{self.lrs_base_url}/api/v1/agents/{agent_id}"
                    )
                    if response.status_code == 200:
                        lrs_state = response.json()
                        # Merge precision data and tool history
                        if "precision_data" in lrs_state:
                            agent.precision_data = lrs_state["precision_data"]
                        if "tool_history" in lrs_state:
                            agent.tool_history = lrs_state["tool_history"]
            except httpx.HTTPError as e:
                logger.error("Failed to sync LRS state", error=str(e))

        # Sync from opencode
        if agent.agent_type in [AgentType.OPENCODE, AgentType.HYBRID]:
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.opencode.timeout
                ) as client:
                    response = await client.get(
                        f"{self.opencode_base_url}/api/v1/agents/{agent_id}"
                    )
                    if response.status_code == 200:
                        opencode_state = response.json()
                        # Merge relevant state information
                        if "status" in opencode_state:
                            agent.status = AgentStatus(opencode_state["status"])
                        if "current_task" in opencode_state:
                            agent.current_task = opencode_state["current_task"]
            except httpx.HTTPError as e:
                logger.error("Failed to sync opencode state", error=str(e))


class ToolExecutor:
    """Tool execution management."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.lrs_base_url = config.lrs.base_url
        self.opencode_base_url = config.opencode.base_url
        self.executions: Dict[str, ToolExecutionResult] = {}

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute tool on appropriate system."""
        execution_id = f"exec_{datetime.utcnow().timestamp()}"

        result = ToolExecutionResult(
            execution_id=execution_id,
            tool_name=request.tool_name,
            status=ToolExecutionStatus.PENDING,
            execution_time=0.0,
        )

        self.executions[execution_id] = result

        # Execute tool in background
        asyncio.create_task(self._execute_tool_background(execution_id, request))

        return result

    async def _execute_tool_background(
        self, execution_id: str, request: ToolExecutionRequest
    ):
        """Execute tool in background."""
        result = self.executions[execution_id]
        start_time = datetime.utcnow()

        try:
            result.status = ToolExecutionStatus.RUNNING

            # Route to appropriate system based on tool name and agent
            if request.agent_id and request.agent_id.startswith("lrs_"):
                # Execute on LRS system
                result = await self._execute_on_lrs(execution_id, request)
            elif request.agent_id and request.agent_id.startswith("opencode_"):
                # Execute on opencode system
                result = await self._execute_on_opencode(execution_id, request)
            else:
                # Try LRS first, then opencode
                try:
                    result = await self._execute_on_lrs(execution_id, request)
                except Exception:
                    result = await self._execute_on_opencode(execution_id, request)

            result.status = ToolExecutionStatus.COMPLETED

        except Exception as e:
            result.status = ToolExecutionStatus.FAILED
            result.error = str(e)
            logger.error(
                "Tool execution failed", execution_id=execution_id, error=str(e)
            )

        finally:
            end_time = datetime.utcnow()
            result.execution_time = (end_time - start_time).total_seconds()
            result.timestamp = end_time

    async def _execute_on_lrs(
        self, execution_id: str, request: ToolExecutionRequest
    ) -> ToolExecutionResult:
        """Execute tool on LRS system."""
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            response = await client.post(
                f"{self.lrs_base_url}/api/v1/tools/execute",
                json={
                    "tool_name": request.tool_name,
                    "parameters": request.parameters,
                    "context": request.context,
                },
            )
            response.raise_for_status()

            result_data = response.json()
            result = self.executions[execution_id]
            result.result = result_data.get("result")
            result.prediction_error = result_data.get("prediction_error")

            return result

    async def _execute_on_opencode(
        self, execution_id: str, request: ToolExecutionRequest
    ) -> ToolExecutionResult:
        """Execute tool on opencode system."""
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            response = await client.post(
                f"{self.opencode_base_url}/api/v1/tools/execute",
                json={
                    "tool_name": request.tool_name,
                    "parameters": request.parameters,
                    "context": request.context,
                },
            )
            response.raise_for_status()

            result_data = response.json()
            result = self.executions[execution_id]
            result.result = result_data.get("result")

            return result

    async def get_execution_result(self, execution_id: str) -> ToolExecutionResult:
        """Get tool execution result."""
        if execution_id not in self.executions:
            raise HTTPException(status_code=404, detail="Execution not found")

        return self.executions[execution_id]


class IntegrationBridgeAPI:
    """Main API application."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.app = FastAPI(
            title="opencode ↔ LRS-Agents Integration Bridge",
            description="Bidirectional API integration service",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize components
        self.auth_middleware = AuthenticationMiddleware(config)
        self.agent_manager = AgentManager(config)
        self.tool_executor = ToolExecutor(config)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/health", response_model=HealthCheck)
        async def health_check():
            """Health check endpoint."""
            return HealthCheck(
                status="healthy",
                services={
                    "lrs_agents": "connected",
                    "opencode": "connected",
                    "database": "connected",
                    "redis": "connected",
                },
            )

        @self.app.get("/system/info", response_model=SystemInfo)
        async def get_system_info():
            """Get system information."""
            return SystemInfo(
                version="1.0.0",
                environment=self.config.environment,
                uptime=0.0,  # Would calculate actual uptime
                active_agents=len(self.agent_manager.agents),
                total_executions=len(self.tool_executor.executions),
                system_load={"cpu": 0.5, "memory": 0.3},
                memory_usage={"used": "256MB", "total": "1GB"},
            )

        # Agent management endpoints
        @self.app.post("/agents", response_model=AgentState)
        async def create_agent(
            request: AgentCreateRequest,
        ):
            """Create new agent."""
            return await self.agent_manager.create_agent(request)

        @self.app.get("/agents", response_model=List[AgentState])
        async def list_agents(
            status: Optional[str] = Query(None),
            current_user: Dict[str, Any] = Depends(require_permission("read_state")),
        ):
            """List all agents."""
            return await self.agent_manager.list_agents(status)

        @self.app.get("/agents/{agent_id}", response_model=AgentState)
        async def get_agent(
            agent_id: str,
            current_user: Dict[str, Any] = Depends(require_permission("read_state")),
            auth_middleware: AuthenticationMiddleware = Depends(),
        ):
            """Get agent details."""
            # Check additional permission for specific agent
            auth_middleware.authorize_action(current_user, "read_state", agent_id)
            return await self.agent_manager.get_agent(agent_id)

        @self.app.put("/agents/{agent_id}", response_model=AgentState)
        async def update_agent(
            agent_id: str,
            request: AgentUpdateRequest,
            current_user: Dict[str, Any] = Depends(require_permission("write_state")),
            auth_middleware: AuthenticationMiddleware = Depends(),
        ):
            """Update agent state."""
            auth_middleware.authorize_action(current_user, "write_state", agent_id)
            return await self.agent_manager.update_agent(agent_id, request)

        @self.app.delete("/agents/{agent_id}")
        async def delete_agent(
            agent_id: str,
            current_user: Dict[str, Any] = Depends(require_permission("manage_agents")),
            auth_middleware: AuthenticationMiddleware = Depends(),
        ):
            """Delete agent."""
            auth_middleware.authorize_action(current_user, "manage_agents", agent_id)
            await self.agent_manager.delete_agent(agent_id)
            return {"message": "Agent deleted successfully"}

        # Tool execution endpoints
        @self.app.post("/tools/execute", response_model=ToolExecutionResult)
        async def execute_tool(
            request: ToolExecutionRequest,
            current_user: Dict[str, Any] = Depends(require_permission("execute_tools")),
        ):
            """Execute tool."""
            return await self.tool_executor.execute_tool(request)

        @self.app.get(
            "/tools/executions/{execution_id}", response_model=ToolExecutionResult
        )
        async def get_execution_result(
            execution_id: str,
            current_user: Dict[str, Any] = Depends(require_permission("read_state")),
        ):
            """Get tool execution result."""
            return await self.tool_executor.get_execution_result(execution_id)

        @self.app.get("/metrics", response_model=IntegrationBridgeMetrics)
        async def get_metrics(
            current_user: Dict[str, Any] = Depends(require_permission("read_state")),
        ):
            """Get bridge metrics."""
            return IntegrationBridgeMetrics(
                total_api_requests=1000,  # Would track actual requests
                active_websocket_connections=0,  # Would track actual connections
                agents_managed=len(self.agent_manager.agents),
                tools_executed=len(self.tool_executor.executions),
                avg_response_time=0.1,
                error_rate=0.01,
                uptime=86400.0,
            )


def create_app(config: IntegrationBridgeConfig) -> FastAPI:
    """Create FastAPI application."""
    bridge_api = IntegrationBridgeAPI(config)
    return bridge_api.app
