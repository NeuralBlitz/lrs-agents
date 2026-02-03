"""
Tool integration layer for bidirectional tool execution between opencode and LRS-Agents.
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from abc import ABC, abstractmethod
import httpx
import structlog

from ..config.settings import IntegrationBridgeConfig
from ..models.schemas import (
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolExecutionStatus,
    AgentState,
)

logger = structlog.get_logger(__name__)


class ToolAdapter(ABC):
    """Abstract base class for tool adapters."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.timeout = 30.0

    @abstractmethod
    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute tool and return result."""
        pass

    @abstractmethod
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        pass

    @abstractmethod
    async def validate_tool_parameters(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> bool:
        """Validate tool parameters."""
        pass


class LRSToolAdapter(ToolAdapter):
    """Adapter for LRS-Agents tools."""

    def __init__(self, config: IntegrationBridgeConfig):
        super().__init__(config)
        self.base_url = config.lrs.base_url
        self.timeout = config.lrs.timeout

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute tool on LRS-Agents system."""
        execution_id = f"lrs_{datetime.utcnow().timestamp()}"
        start_time = datetime.utcnow()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Make request to LRS tool execution endpoint
                response = await client.post(
                    f"{self.base_url}/api/v1/tools/execute",
                    json={
                        "tool_name": request.tool_name,
                        "parameters": request.parameters,
                        "context": request.context,
                        "agent_id": request.agent_id,
                    },
                    timeout=request.timeout,
                )

                response.raise_for_status()
                result_data = response.json()

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                return ToolExecutionResult(
                    execution_id=execution_id,
                    tool_name=request.tool_name,
                    status=ToolExecutionStatus.COMPLETED,
                    result=result_data.get("result"),
                    prediction_error=result_data.get("prediction_error"),
                    execution_time=execution_time,
                    timestamp=datetime.utcnow(),
                )

        except httpx.TimeoutException:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=request.tool_name,
                status=ToolExecutionStatus.TIMEOUT,
                error="Tool execution timed out",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
            )

        except httpx.HTTPError as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "LRS tool execution failed", tool_name=request.tool_name, error=str(e)
            )

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"HTTP error: {str(e)}",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "LRS tool execution error", tool_name=request.tool_name, error=str(e)
            )

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"Execution error: {str(e)}",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from LRS-Agents."""
        # This would query LRS for available tools
        return [
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "max_results": {"type": "integer", "default": 10},
                },
            },
            {
                "name": "file_operation",
                "description": "Perform file operations",
                "parameters": {
                    "operation": {"type": "string", "required": True},
                    "path": {"type": "string", "required": True},
                    "content": {"type": "string"},
                },
            },
            {
                "name": "database_query",
                "description": "Execute database queries",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "parameters": {"type": "object"},
                },
            },
        ]

    async def validate_tool_parameters(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> bool:
        """Validate LRS tool parameters."""
        available_tools = self.get_available_tools()
        tool_info = next((t for t in available_tools if t["name"] == tool_name), None)

        if not tool_info:
            return False

        required_params = tool_info["parameters"]
        for param_name, param_info in required_params.items():
            if param_info.get("required", False) and param_name not in parameters:
                return False

        return True


class OpenCodeToolAdapter(ToolAdapter):
    """Adapter for opencode tools."""

    def __init__(self, config: IntegrationBridgeConfig):
        super().__init__(config)
        self.base_url = config.opencode.base_url
        self.timeout = config.opencode.timeout

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute tool on opencode system."""
        execution_id = f"opencode_{datetime.utcnow().timestamp()}"
        start_time = datetime.utcnow()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Make request to opencode tool execution endpoint
                response = await client.post(
                    f"{self.base_url}/api/v1/tools/execute",
                    json={
                        "tool_name": request.tool_name,
                        "parameters": request.parameters,
                        "context": request.context,
                        "agent_id": request.agent_id,
                    },
                    timeout=request.timeout,
                )

                response.raise_for_status()
                result_data = response.json()

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                return ToolExecutionResult(
                    execution_id=execution_id,
                    tool_name=request.tool_name,
                    status=ToolExecutionStatus.COMPLETED,
                    result=result_data.get("result"),
                    execution_time=execution_time,
                    timestamp=datetime.utcnow(),
                )

        except httpx.TimeoutException:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=request.tool_name,
                status=ToolExecutionStatus.TIMEOUT,
                error="Tool execution timed out",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
            )

        except httpx.HTTPError as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "opencode tool execution failed",
                tool_name=request.tool_name,
                error=str(e),
            )

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"HTTP error: {str(e)}",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "opencode tool execution error",
                tool_name=request.tool_name,
                error=str(e),
            )

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"Execution error: {str(e)}",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from opencode."""
        # This would query opencode for available tools
        return [
            {
                "name": "code_execution",
                "description": "Execute code in various languages",
                "parameters": {
                    "code": {"type": "string", "required": True},
                    "language": {"type": "string", "required": True},
                    "timeout": {"type": "integer", "default": 30},
                },
            },
            {
                "name": "file_system",
                "description": "File system operations",
                "parameters": {
                    "operation": {"type": "string", "required": True},
                    "path": {"type": "string", "required": True},
                    "content": {"type": "string"},
                },
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        ]

    async def validate_tool_parameters(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> bool:
        """Validate opencode tool parameters."""
        available_tools = self.get_available_tools()
        tool_info = next((t for t in available_tools if t["name"] == tool_name), None)

        if not tool_info:
            return False

        required_params = tool_info["parameters"]
        for param_name, param_info in required_params.items():
            if param_info.get("required", False) and param_name not in parameters:
                return False

        return True


class ToolRouter:
    """Routes tool execution requests to appropriate adapters."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.lrs_adapter = LRSToolAdapter(config)
        self.opencode_adapter = OpenCodeToolAdapter(config)
        self.tool_registry: Dict[str, str] = {}  # tool_name -> adapter_name

        # Initialize tool registry
        self._initialize_tool_registry()

    def _initialize_tool_registry(self):
        """Initialize tool registry with available tools."""
        # Register LRS tools
        for tool in self.lrs_adapter.get_available_tools():
            self.tool_registry[f"lrs_{tool['name']}"] = "lrs"
            self.tool_registry[tool["name"]] = "lrs"  # Also register without prefix

        # Register opencode tools
        for tool in self.opencode_adapter.get_available_tools():
            self.tool_registry[f"opencode_{tool['name']}"] = "opencode"
            self.tool_registry[tool["name"]] = (
                "opencode"  # Also register without prefix
            )

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute tool using appropriate adapter."""
        # Determine which adapter to use
        adapter_name = self._get_adapter_for_tool(request.tool_name)

        if not adapter_name:
            return ToolExecutionResult(
                execution_id=f"failed_{datetime.utcnow().timestamp()}",
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"Unknown tool: {request.tool_name}",
                execution_time=0.0,
                timestamp=datetime.utcnow(),
            )

        # Get the appropriate adapter
        adapter = self._get_adapter(adapter_name)

        # Validate parameters
        if not await adapter.validate_tool_parameters(
            request.tool_name, request.parameters
        ):
            return ToolExecutionResult(
                execution_id=f"failed_{datetime.utcnow().timestamp()}",
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"Invalid parameters for tool: {request.tool_name}",
                execution_time=0.0,
                timestamp=datetime.utcnow(),
            )

        # Execute tool
        try:
            return await adapter.execute_tool(request)
        except Exception as e:
            logger.error(
                "Tool execution failed", tool_name=request.tool_name, error=str(e)
            )

            return ToolExecutionResult(
                execution_id=f"failed_{datetime.utcnow().timestamp()}",
                tool_name=request.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"Tool execution failed: {str(e)}",
                execution_time=0.0,
                timestamp=datetime.utcnow(),
            )

    def _get_adapter_for_tool(self, tool_name: str) -> Optional[str]:
        """Get adapter name for a tool."""
        # Direct lookup
        if tool_name in self.tool_registry:
            return self.tool_registry[tool_name]

        # Pattern matching
        if tool_name.startswith("lrs_"):
            return "lrs"
        elif tool_name.startswith("opencode_"):
            return "opencode"

        # Try to find in any adapter
        if tool_name in [t["name"] for t in self.lrs_adapter.get_available_tools()]:
            return "lrs"
        elif tool_name in [
            t["name"] for t in self.opencode_adapter.get_available_tools()
        ]:
            return "opencode"

        return None

    def _get_adapter(self, adapter_name: str) -> ToolAdapter:
        """Get adapter by name."""
        if adapter_name == "lrs":
            return self.lrs_adapter
        elif adapter_name == "opencode":
            return self.opencode_adapter
        else:
            raise ValueError(f"Unknown adapter: {adapter_name}")

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools from both adapters."""
        tools = []

        # Get LRS tools
        for tool in self.lrs_adapter.get_available_tools():
            tool_copy = tool.copy()
            tool_copy["source"] = "lrs"
            tool_copy["names"] = [tool["name"], f"lrs_{tool['name']}"]
            tools.append(tool_copy)

        # Get opencode tools
        for tool in self.opencode_adapter.get_available_tools():
            tool_copy = tool.copy()
            tool_copy["source"] = "opencode"
            tool_copy["names"] = [tool["name"], f"opencode_{tool['name']}"]
            tools.append(tool_copy)

        return tools

    async def register_custom_tool(
        self, tool_name: str, adapter_name: str, tool_info: Dict[str, Any]
    ):
        """Register a custom tool."""
        self.tool_registry[tool_name] = adapter_name

        # This could also register with the specific adapter
        # if the adapter supports dynamic tool registration

        logger.info("Registered custom tool", tool_name=tool_name, adapter=adapter_name)


class ToolExecutionManager:
    """Manages tool execution with fallback and retry logic."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.tool_router = ToolRouter(config)
        self.execution_history: Dict[str, List[ToolExecutionResult]] = {}
        self.retry_attempts = config.lrs.max_retries

    async def execute_tool_with_fallback(
        self, request: ToolExecutionRequest
    ) -> ToolExecutionResult:
        """Execute tool with automatic fallback and retry."""
        execution_results = []

        # First attempt with requested tool
        result = await self.tool_router.execute_tool(request)
        execution_results.append(result)

        # If failed and we have alternative tools, try them
        if result.status == ToolExecutionStatus.FAILED:
            alternative_tools = await self._find_alternative_tools(request.tool_name)

            for alt_tool in alternative_tools:
                alt_request = ToolExecutionRequest(
                    tool_name=alt_tool,
                    parameters=request.parameters,
                    timeout=request.timeout,
                    agent_id=request.agent_id,
                    context=request.context,
                )

                alt_result = await self.tool_router.execute_tool(alt_request)
                execution_results.append(alt_result)

                # Stop if we get a successful result
                if alt_result.status == ToolExecutionStatus.COMPLETED:
                    break

        # Store execution history
        if request.agent_id:
            if request.agent_id not in self.execution_history:
                self.execution_history[request.agent_id] = []

            self.execution_history[request.agent_id].extend(execution_results)

            # Keep only last 100 executions per agent
            if len(self.execution_history[request.agent_id]) > 100:
                self.execution_history[request.agent_id] = self.execution_history[
                    request.agent_id
                ][-100:]

        # Return the best result (prefer successful results)
        successful_results = [
            r for r in execution_results if r.status == ToolExecutionStatus.COMPLETED
        ]
        if successful_results:
            return successful_results[0]

        # Return the first failed result if all failed
        return execution_results[0]

    async def _find_alternative_tools(self, tool_name: str) -> List[str]:
        """Find alternative tools for a given tool."""
        all_tools = await self.tool_router.get_available_tools()
        alternatives = []

        # Find tools with similar names or descriptions
        target_tool = next(
            (t for t in all_tools if tool_name in t.get("names", [])), None
        )

        if not target_tool:
            return alternatives

        target_desc = target_tool.get("description", "").lower()

        for tool in all_tools:
            if tool_name in tool.get("names", []):
                continue

            tool_desc = tool.get("description", "").lower()

            # Simple similarity check
            if any(word in tool_desc for word in target_desc.split() if len(word) > 3):
                alternatives.extend(tool.get("names", []))

        return alternatives[:3]  # Limit to 3 alternatives

    def get_execution_history(
        self, agent_id: str, limit: int = 50
    ) -> List[ToolExecutionResult]:
        """Get execution history for an agent."""
        if agent_id not in self.execution_history:
            return []

        return self.execution_history[agent_id][-limit:]

    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        total_executions = sum(
            len(history) for history in self.execution_history.values()
        )
        successful_executions = 0
        failed_executions = 0

        for history in self.execution_history.values():
            for execution in history:
                if execution.status == ToolExecutionStatus.COMPLETED:
                    successful_executions += 1
                else:
                    failed_executions += 1

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions
            if total_executions > 0
            else 0,
            "agents_active": len(self.execution_history),
        }
