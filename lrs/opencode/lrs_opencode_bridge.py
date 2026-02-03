#!/usr/bin/env python3
"""Bidirectional OpenCode ↔ LRS-Agents Integration."""

import asyncio
import json
import os
import subprocess
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# LRS-Agents imports
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.core.registry import ToolRegistry
from lrs.core.precision import HierarchicalPrecision
from lrs.integration.langgraph import create_lrs_agent

# FastAPI for API integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


class OpenCodeLRSBridge:
    """Bridge between OpenCode CLI and LRS-Agents framework."""

    def __init__(self):
        self.lrs_agents: Dict[str, Any] = {}
        self.opencode_path = self._find_opencode()
        self.api_app = self._create_api_app()

    def _find_opencode(self) -> Optional[str]:
        """Locate OpenCode executable."""
        candidates = ["opencode", "./node_modules/.bin/opencode"]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return candidate
            except:
                continue
        return None

    def _create_api_app(self) -> FastAPI:
        """Create FastAPI app for bidirectional communication."""
        app = FastAPI(title="OpenCode ↔ LRS-Agents Bridge")

        class LRSRequest(BaseModel):
            agent_id: str
            task: str
            context: Optional[Dict[str, Any]] = None

        class OpenCodeRequest(BaseModel):
            command: str
            working_dir: Optional[str] = "."
            timeout: Optional[int] = 30

        @app.post("/lrs/create-agent")
        async def create_lrs_agent_endpoint(request: LRSRequest):
            """Create new LRS agent accessible from OpenCode."""
            try:
                # Create a simple LRS agent with OpenCode tool
                registry = ToolRegistry()
                opencode_tool = OpenCodeToolLens()
                registry.register(opencode_tool)

                agent = create_lrs_agent(
                    llm=None,  # Simplified for demo
                    tools=[opencode_tool],
                )

                agent_id = request.agent_id
                self.lrs_agents[agent_id] = agent

                return {"status": "success", "agent_id": agent_id}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/lrs/execute")
        async def execute_lrs_task(request: LRSRequest):
            """Execute task using LRS agent."""
            if request.agent_id not in self.lrs_agents:
                raise HTTPException(status_code=404, detail="Agent not found")

            try:
                agent = self.lrs_agents[request.agent_id]
                result = agent.invoke(
                    {
                        "messages": [{"role": "user", "content": request.task}],
                        **(request.context or {}),
                    }
                )
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/opencode/execute")
        async def execute_opencode_command(request: OpenCodeRequest):
            """Execute OpenCode command from LRS agent."""
            if not self.opencode_path:
                raise HTTPException(status_code=503, detail="OpenCode not available")

            try:
                # Parse command for opencode
                cmd_parts = request.command.split()
                if not cmd_parts:
                    raise HTTPException(status_code=400, detail="Empty command")

                cmd = [self.opencode_path] + cmd_parts[1:]  # Skip 'opencode' prefix

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    cwd=request.working_dir,
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=408, detail="Command timed out")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return app

    def create_opencode_tool_for_lrs(self) -> "OpenCodeToolLens":
        """Create ToolLens that LRS agents can use to call OpenCode."""
        return OpenCodeToolLens()

    async def run_api_server(self, host: str = "localhost", port: int = 8765):
        """Run the bridge API server."""
        config = uvicorn.Config(self.api_app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()


class OpenCodeToolLens(ToolLens):
    """LRS ToolLens for calling OpenCode functions."""

    def __init__(self):
        super().__init__("opencode_tool")
        self.bridge_url = "http://localhost:8765"

    def get(self, belief_state: Dict[str, Any]) -> ExecutionResult:
        """Execute OpenCode operation via bridge API."""
        try:
            import requests

            task = belief_state.get("current_task", "")
            if not task:
                return ExecutionResult(False, None, 0.8, "No task specified")

            # Map belief state to opencode command
            command = self._belief_state_to_command(belief_state)

            response = requests.post(
                f"{self.bridge_url}/opencode/execute",
                json={
                    "command": command,
                    "working_dir": belief_state.get("working_directory", "."),
                    "timeout": 30,
                },
                timeout=35,
            )

            if response.status_code == 200:
                result = response.json()
                return ExecutionResult(
                    success=result["success"],
                    value=result,
                    prediction_error=0.0 if result["success"] else 0.5,
                    error=result.get("stderr") if not result["success"] else None,
                )
            else:
                return ExecutionResult(
                    success=False,
                    value=None,
                    prediction_error=0.7,
                    error=f"Bridge API error: {response.status_code}",
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                value=None,
                prediction_error=0.9,
                error=f"OpenCode tool error: {str(e)}",
            )

    def _belief_state_to_command(self, belief_state: Dict[str, Any]) -> str:
        """Convert belief state to OpenCode command."""
        task = belief_state.get("current_task", "").lower()

        if "search" in task or "find" in task:
            pattern = belief_state.get("search_pattern", ".*")
            return f"grep {pattern}"
        elif "read" in task or "view" in task:
            file_path = belief_state.get("target_file", "")
            return f"read {file_path}" if file_path else "list"
        elif "run" in task or "execute" in task:
            cmd = belief_state.get("command", "")
            return f"bash {cmd}"
        else:
            return f"question {task}"

    def set(self, belief_state: Dict[str, Any], value: Any) -> Dict[str, Any]:
        """Update belief state with OpenCode results."""
        if isinstance(value, dict):
            belief_state["opencode_result"] = value
            belief_state["information_gained"] = len(value.get("stdout", ""))
            belief_state["last_command_success"] = value.get("success", False)
        return belief_state


# =============================================================================
# Integration Examples
# =============================================================================


def example_lrs_using_opencode():
    """Example: LRS agent using OpenCode for code analysis."""
    print("=== LRS Agent using OpenCode ===")

    # Create bridge
    bridge = OpenCodeLRSBridge()

    # Create LRS agent with OpenCode tool
    registry = ToolRegistry()
    opencode_tool = bridge.create_opencode_tool_for_lrs()
    registry.register(opencode_tool)

    # This would normally use an LLM, but simplified for demo
    print("LRS agent configured with OpenCode tool")
    print("Agent can now use OpenCode for file operations, searches, etc.")


def example_opencode_using_lrs():
    """Example: OpenCode calling LRS agents for complex reasoning."""
    print("=== OpenCode using LRS Agents ===")

    print("""
    To integrate LRS agents into OpenCode workflow:

    1. Start the bridge API server:
       bridge = OpenCodeLRSBridge()
       asyncio.run(bridge.run_api_server())

    2. In OpenCode, call LRS agents via HTTP requests:
       - Create agent: POST /lrs/create-agent
       - Execute tasks: POST /lrs/execute

    3. LRS agents can then handle complex multi-step tasks
       that require active inference and precision tracking
    """)


def example_bidirectional_workflow():
    """Example of full bidirectional workflow."""
    print("=== Bidirectional OpenCode ↔ LRS-Agents Workflow ===")

    workflow = """
    1. OpenCode receives complex coding task
    2. OpenCode calls LRS agent for strategic planning
    3. LRS agent uses active inference to break down task
    4. LRS agent calls OpenCode tools for implementation
    5. OpenCode executes specific file operations
    6. Results feed back to LRS agent for precision updates
    7. Cycle continues until goal satisfaction
    """

    print(workflow)


if __name__ == "__main__":
    print("OpenCode ↔ LRS-Agents Integration Examples")
    print("=" * 50)

    example_lrs_using_opencode()
    print()
    example_opencode_using_lrs()
    print()
    example_bidirectional_workflow()
