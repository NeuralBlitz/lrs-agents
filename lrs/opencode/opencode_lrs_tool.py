#!/usr/bin/env python3
"""OpenCode ToolLens for LRS-Agents integration."""

import json
import subprocess
import sys
from typing import Dict, Any, Optional
from pathlib import Path

from lrs.core.lens import ToolLens, ExecutionResult
from lrs.core.free_energy import calculate_epistemic_value, calculate_pragmatic_value


class OpenCodeTool(ToolLens):
    """ToolLens wrapper for OpenCode CLI capabilities."""

    def __init__(self, name: str = "opencode_tool"):
        super().__init__(name)
        self.opencode_path = self._find_opencode()

    def _find_opencode(self) -> Optional[str]:
        """Find opencode executable in system."""
        # Check common locations
        candidates = [
            "opencode",
            "./node_modules/.bin/opencode",
            "/usr/local/bin/opencode",
            "/usr/bin/opencode",
        ]

        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return None

    def get(self, belief_state: Dict[str, Any]) -> ExecutionResult:
        """Execute opencode query based on belief state."""
        if not self.opencode_path:
            return ExecutionResult(
                success=False,
                value=None,
                prediction_error=1.0,
                error="OpenCode not found in system",
            )

        try:
            # Extract task from belief state
            task = belief_state.get("current_task", "")
            if not task:
                return ExecutionResult(
                    success=False,
                    value=None,
                    prediction_error=0.8,
                    error="No task specified in belief state",
                )

            # Formulate opencode command
            cmd = [self.opencode_path]

            # Map task types to opencode commands
            if "search" in task.lower() or "find" in task.lower():
                cmd.extend(["grep", task])
            elif "read" in task.lower() or "view" in task.lower():
                # Would need file path from belief state
                file_path = belief_state.get("target_file")
                if file_path:
                    cmd.extend(["read", file_path])
                else:
                    cmd.extend(["list"])  # List directory
            elif "edit" in task.lower() or "modify" in task.lower():
                # Would need more context for editing
                return ExecutionResult(
                    success=False,
                    value=None,
                    prediction_error=0.6,
                    error="Edit operations require specific context",
                )
            else:
                # General query
                cmd.append(task)

            # Execute opencode command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=belief_state.get("working_directory", "."),
            )

            # Calculate success based on return code and output
            success = result.returncode == 0
            prediction_error = 0.0 if success else 0.5

            # Parse output
            value = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": " ".join(cmd),
            }

            return ExecutionResult(
                success=success,
                value=value,
                prediction_error=prediction_error,
                error=result.stderr if not success else None,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                value=None,
                prediction_error=0.9,
                error="OpenCode command timed out",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                value=None,
                prediction_error=0.8,
                error=f"OpenCode execution error: {str(e)}",
            )

    def set(self, belief_state: Dict[str, Any], value: Any) -> Dict[str, Any]:
        """Update belief state based on opencode execution result."""
        if isinstance(value, dict) and "stdout" in value:
            # Update belief state with execution results
            belief_state["last_opencode_output"] = value["stdout"]
            belief_state["last_opencode_error"] = value.get("stderr", "")
            belief_state["last_opencode_success"] = value.get("returncode", -1) == 0

            # Extract insights from output for epistemic value
            if value["stdout"].strip():
                belief_state["information_gained"] = len(value["stdout"].split())
            else:
                belief_state["information_gained"] = 0

        return belief_state

    def calculate_epistemic_value(self, belief_state: Dict[str, Any]) -> float:
        """Calculate information gain from opencode execution."""
        info_gain = belief_state.get("information_gained", 0)
        uncertainty_reduction = min(info_gain / 100.0, 1.0)  # Normalize

        # Higher value for successful information retrieval
        return uncertainty_reduction * 2.0

    def calculate_pragmatic_value(
        self, belief_state: Dict[str, Any], preferences: Dict[str, float]
    ) -> float:
        """Calculate goal-directed value of opencode execution."""
        success = belief_state.get("last_opencode_success", False)
        info_gain = belief_state.get("information_gained", 0)

        # Positive value for successful task completion
        success_value = (
            preferences.get("success", 1.0)
            if success
            else preferences.get("error", -1.0)
        )

        # Bonus for information gain
        info_bonus = min(info_gain / 50.0, 2.0) * preferences.get("information", 0.5)

        return success_value + info_bonus


def create_opencode_integration():
    """Create OpenCode integration for LRS-Agents."""
    return OpenCodeTool()


# Example usage
if __name__ == "__main__":
    # Test the integration
    opencode_tool = create_opencode_integration()

    test_belief_state = {
        "current_task": "list files in current directory",
        "working_directory": ".",
        "goal": "explore project structure",
    }

    result = opencode_tool.get(test_belief_state)
    print(f"OpenCode Tool Result: {result.success}")
    print(f"Output: {result.value}")
    print(f"Prediction Error: {result.prediction_error}")

    if result.success:
        updated_state = opencode_tool.set(test_belief_state, result.value)
        epistemic = opencode_tool.calculate_epistemic_value(updated_state)
        pragmatic = opencode_tool.calculate_pragmatic_value(
            updated_state, {"success": 2.0, "error": -1.0}
        )

        print(f"Epistemic Value: {epistemic}")
        print(f"Pragmatic Value: {pragmatic}")
