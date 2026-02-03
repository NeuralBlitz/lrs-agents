"""
TUI Interaction Tool: ToolLens implementation for bidirectional TUI communication.

This tool treats TUI interactions as first-class LRS tools, allowing
agents to query, command, and synchronize with the opencode TUI.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...core.lens import ToolLens, ExecutionResult
from ...core.precision import PrecisionParameters


class TUIInteractionTool(ToolLens):
    """
    Tool for bidirectional TUI communication using ToolLens framework.

    This tool enables LRS agents to:
    - Query TUI state and user preferences
    - Execute commands in the TUI interface
    - Receive user feedback and input
    - Synchronize agent state with TUI display
    - Trigger TUI notifications and alerts

    Examples:
        >>> tui_tool = TUIInteractionTool(tui_bridge)
        >>> # Query user preference
        >>> result = tui_tool.get({
        ...     'action': 'query',
        ...     'query_type': 'user_preference',
        ...     'preference_key': 'debug_mode'
        ... })
        >>>
        >>> # Execute TUI command
        >>> result = tui_tool.get({
        ...     'action': 'command',
        ...     'command': 'show_notification',
        ...     'message': 'Task completed successfully'
        ... })
        >>>
        >>> # Sync agent state to TUI
        >>> result = tui_tool.get({
        ...     'action': 'state_update',
        ...     'state_data': {'status': 'working', 'progress': 0.75}
        ... })
    """

    def __init__(self, tui_bridge, name: str = "tui_interaction"):
        """
        Initialize TUI Interaction Tool.

        Args:
            tui_bridge: TUIBridge instance for communication
            name: Tool identifier
        """
        super().__init__(
            name=name,
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query", "command", "state_update", "notification", "user_input"],
                    },
                    "query_type": {
                        "type": "string",
                        "enum": ["user_preference", "tui_state", "agent_list", "system_info"],
                    },
                    "preference_key": {"type": "string"},
                    "command": {
                        "type": "string",
                        "enum": [
                            "show_notification",
                            "update_display",
                            "refresh_data",
                            "open_panel",
                        ],
                    },
                    "message": {"type": "string"},
                    "notification_type": {
                        "type": "string",
                        "enum": ["info", "warning", "error", "success"],
                    },
                    "state_data": {"type": "object"},
                    "tui_context": {"type": "object"},
                    "timeout": {"type": "number", "default": 30.0},
                },
                "required": ["action"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "response": {"type": "object"},
                    "error": {"type": "string"},
                    "prediction_error": {"type": "number"},
                },
            },
        )

        self.tui_bridge = tui_bridge
        self.pending_user_inputs: Dict[str, asyncio.Future] = {}

        # Track TUI interaction success rates for precision
        self.action_stats: Dict[str, Dict[str, int]] = {
            "query": {"success": 0, "failure": 0},
            "command": {"success": 0, "failure": 0},
            "state_update": {"success": 0, "failure": 0},
            "notification": {"success": 0, "failure": 0},
            "user_input": {"success": 0, "failure": 0},
        }

    def get(self, state: Dict[str, Any]) -> ExecutionResult:
        """
        Execute TUI interaction (forward operation).

        Args:
            state: Input state containing action and parameters

        Returns:
            ExecutionResult with TUI response and prediction error
        """
        self.call_count += 1

        action = state.get("action")

        try:
            if action == "query":
                result = self._handle_query(state)
            elif action == "command":
                result = self._handle_command(state)
            elif action == "state_update":
                result = self._handle_state_update(state)
            elif action == "notification":
                result = self._handle_notification(state)
            elif action == "user_input":
                result = self._handle_user_input(state)
            else:
                result = ExecutionResult(
                    success=False,
                    value=None,
                    error=f"Unknown action: {action}",
                    prediction_error=0.8,
                )

            # Update success statistics
            self._update_action_stats(action, result.success)

            # Calculate prediction error based on success rate
            prediction_error = self._calculate_prediction_error(action)

            return ExecutionResult(
                success=result.success,
                value=result.value,
                error=result.error,
                prediction_error=prediction_error,
            )

        except Exception as e:
            self.failure_count += 1
            self._update_action_stats(action, False)

            return ExecutionResult(success=False, value=None, error=str(e), prediction_error=0.9)

    def set(self, state: Dict[str, Any], observation: Any) -> Dict[str, Any]:
        """
        Update belief state with TUI observation (backward operation).

        Args:
            state: Current state
            observation: TUI response/result

        Returns:
            Updated state with TUI feedback
        """
        # Extract relevant information from TUI response
        if observation and isinstance(observation, dict):
            tui_feedback = observation.get("response", {})

            # Update state with TUI-specific information
            updated_state = state.copy()

            # Add TUI context
            if "tui_context" not in updated_state:
                updated_state["tui_context"] = {}

            # Merge TUI feedback
            updated_state["tui_context"].update(tui_feedback)

            # Update timestamp
            updated_state["tui_context"]["last_interaction"] = datetime.now().isoformat()

            # Add user preferences if queried
            if "user_preferences" in tui_feedback:
                updated_state["user_preferences"] = tui_feedback["user_preferences"]

            # Add TUI state information
            if "tui_state" in tui_feedback:
                updated_state["tui_state"] = tui_feedback["tui_state"]

            return updated_state

        return state

    def _handle_query(self, state: Dict[str, Any]) -> ExecutionResult:
        """Handle TUI query actions."""

        query_type = state.get("query_type")

        if query_type == "user_preference":
            preference_key = state.get("preference_key")
            if not preference_key:
                return ExecutionResult(
                    success=False,
                    value=None,
                    error="Missing preference_key for user_preference query",
                    prediction_error=0.6,
                )

            # Query preference from TUI bridge
            preference = self.tui_bridge.state_mirror.get_user_preference(preference_key)

            return ExecutionResult(
                success=True,
                value={"response": {"user_preferences": {preference_key: preference}}},
                error=None,
                prediction_error=0.1,
            )

        elif query_type == "tui_state":
            # Get current TUI state
            tui_state = self.tui_bridge.state_mirror.get_tui_state()

            return ExecutionResult(
                success=True,
                value={"response": {"tui_state": tui_state}},
                error=None,
                prediction_error=0.1,
            )

        elif query_type == "agent_list":
            # Get list of active agents
            if self.tui_bridge.coordinator:
                agents = self.tui_bridge.coordinator.list_agents()
            else:
                agents = list(self.tui_bridge.shared_state.get_all_states().keys())

            return ExecutionResult(
                success=True,
                value={"response": {"agents": agents}},
                error=None,
                prediction_error=0.1,
            )

        elif query_type == "system_info":
            # Get system information
            system_info = {
                "bridge_status": "active",
                "websocket_connections": len(self.tui_bridge.websocket_manager.connections),
                "active_agents": len(self.tui_bridge.shared_state.get_all_states()),
                "timestamp": datetime.now().isoformat(),
            }

            return ExecutionResult(
                success=True,
                value={"response": {"system_info": system_info}},
                error=None,
                prediction_error=0.1,
            )

        else:
            return ExecutionResult(
                success=False,
                value=None,
                error=f"Unknown query_type: {query_type}",
                prediction_error=0.7,
            )

    def _handle_command(self, state: Dict[str, Any]) -> ExecutionResult:
        """Handle TUI command actions."""

        command = state.get("command")

        if command == "show_notification":
            message = state.get("message")
            notification_type = state.get("notification_type", "info")

            if not message:
                return ExecutionResult(
                    success=False,
                    value=None,
                    error="Missing message for show_notification command",
                    prediction_error=0.6,
                )

            # Send notification through TUI bridge
            asyncio.create_task(
                self.tui_bridge.websocket_manager.broadcast(
                    "tui_events",
                    {
                        "type": "notification",
                        "message": message,
                        "notification_type": notification_type,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            )

            return ExecutionResult(
                success=True,
                value={"response": {"notification_sent": True}},
                error=None,
                prediction_error=0.1,
            )

        elif command == "update_display":
            display_data = state.get("state_data", {})

            # Update TUI display
            asyncio.create_task(
                self.tui_bridge.websocket_manager.broadcast(
                    "tui_events",
                    {
                        "type": "display_update",
                        "data": display_data,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            )

            return ExecutionResult(
                success=True,
                value={"response": {"display_updated": True}},
                error=None,
                prediction_error=0.1,
            )

        elif command == "refresh_data":
            # Trigger data refresh
            asyncio.create_task(
                self.tui_bridge.websocket_manager.broadcast(
                    "tui_events",
                    {"type": "refresh_request", "timestamp": datetime.now().isoformat()},
                )
            )

            return ExecutionResult(
                success=True,
                value={"response": {"refresh_triggered": True}},
                error=None,
                prediction_error=0.1,
            )

        elif command == "open_panel":
            panel_name = state.get("panel_name")
            if not panel_name:
                return ExecutionResult(
                    success=False,
                    value=None,
                    error="Missing panel_name for open_panel command",
                    prediction_error=0.6,
                )

            asyncio.create_task(
                self.tui_bridge.websocket_manager.broadcast(
                    "tui_events",
                    {
                        "type": "open_panel",
                        "panel_name": panel_name,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            )

            return ExecutionResult(
                success=True,
                value={"response": {"panel_opened": panel_name}},
                error=None,
                prediction_error=0.1,
            )

        else:
            return ExecutionResult(
                success=False, value=None, error=f"Unknown command: {command}", prediction_error=0.7
            )

    def _handle_state_update(self, state: Dict[str, Any]) -> ExecutionResult:
        """Handle state synchronization actions."""

        state_data = state.get("state_data", {})

        if not state_data:
            return ExecutionResult(
                success=False,
                value=None,
                error="Missing state_data for state_update action",
                prediction_error=0.6,
            )

        # Update shared state
        agent_id = state.get("agent_id", "tui_agent")
        self.tui_bridge.shared_state.update(agent_id, state_data)

        # Broadcast state update to TUI
        asyncio.create_task(
            self.tui_bridge.websocket_manager.broadcast(
                "tui_events",
                {
                    "type": "state_update",
                    "agent_id": agent_id,
                    "state_data": state_data,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        )

        return ExecutionResult(
            success=True,
            value={"response": {"state_updated": True, "agent_id": agent_id}},
            error=None,
            prediction_error=0.1,
        )

    def _handle_notification(self, state: Dict[str, Any]) -> ExecutionResult:
        """Handle notification actions (alias for command)."""

        # Convert to command format
        notification_state = {
            "action": "command",
            "command": "show_notification",
            "message": state.get("message", ""),
            "notification_type": state.get("notification_type", "info"),
        }

        return self._handle_command(notification_state)

    def _handle_user_input(self, state: Dict[str, Any]) -> ExecutionResult:
        """Handle user input requests."""

        prompt = state.get("prompt", "Please provide input:")
        input_type = state.get("input_type", "text")
        timeout = state.get("timeout", 30.0)

        # Generate unique request ID
        request_id = f"input_{datetime.now().timestamp()}"

        # Create future for response
        future = asyncio.Future()
        self.pending_user_inputs[request_id] = future

        # Broadcast input request to TUI
        asyncio.create_task(
            self.tui_bridge.websocket_manager.broadcast(
                "tui_events",
                {
                    "type": "user_input_request",
                    "request_id": request_id,
                    "prompt": prompt,
                    "input_type": input_type,
                    "timeout": timeout,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        )

        try:
            # Wait for user response (with timeout)
            response = asyncio.wait_for(future, timeout=timeout)

            return ExecutionResult(
                success=True,
                value={"response": {"user_input": response}},
                error=None,
                prediction_error=0.2,  # User input can be unpredictable
            )

        except asyncio.TimeoutError:
            del self.pending_user_inputs[request_id]

            return ExecutionResult(
                success=False,
                value=None,
                error=f"User input timed out after {timeout}s",
                prediction_error=0.8,
            )

    def handle_user_input_response(self, request_id: str, response: Any):
        """Handle incoming user input response from TUI."""

        if request_id in self.pending_user_inputs:
            future = self.pending_user_inputs.pop(request_id)
            if not future.done():
                future.set_result(response)

    def _update_action_stats(self, action: str, success: bool):
        """Update success/failure statistics for action."""
        if action in self.action_stats:
            if success:
                self.action_stats[action]["success"] += 1
            else:
                self.action_stats[action]["failure"] += 1

    def _calculate_prediction_error(self, action: str) -> float:
        """Calculate prediction error based on action success rate."""
        if action not in self.action_stats:
            return 0.5  # Default uncertainty

        stats = self.action_stats[action]
        total = stats["success"] + stats["failure"]

        if total == 0:
            return 0.5

        success_rate = stats["success"] / total

        # Higher success rate = lower prediction error
        return 1.0 - success_rate

    @property
    def success_rate(self) -> float:
        """Get overall success rate across all actions."""
        total_success = sum(stats["success"] for stats in self.action_stats.values())
        total_failure = sum(stats["failure"] for stats in self.action_stats.values())
        total = total_success + total_failure

        if total == 0:
            return 0.5

        return total_success / total
