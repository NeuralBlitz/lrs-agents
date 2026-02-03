#!/usr/bin/env python3
"""Simplified OpenCode ↔ LRS-Agents Integration Demo (without numpy dependencies)."""

import subprocess
from typing import Dict, Any, Optional


class SimplifiedToolLens:
    """Simplified ToolLens interface for demo purposes."""

    def __init__(self, name: str):
        self.name = name

    def get(self, belief_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operation and return result."""
        raise NotImplementedError

    def set(self, belief_state: Dict[str, Any], value: Any) -> Dict[str, Any]:
        """Update belief state with result."""
        belief_state["last_result"] = value
        return belief_state


class OpenCodeTool(SimplifiedToolLens):
    """OpenCode tool that can be used by LRS-style agents."""

    def __init__(self):
        super().__init__("opencode_tool")
        self.opencode_path = self._find_opencode()

    def _find_opencode(self) -> Optional[str]:
        """Find opencode executable."""
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

    def get(self, belief_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute OpenCode command based on belief state."""
        if not self.opencode_path:
            return {
                "success": False,
                "error": "OpenCode not found",
                "prediction_error": 1.0,
            }

        task = belief_state.get("current_task", "")
        if not task:
            return {
                "success": False,
                "error": "No task specified",
                "prediction_error": 0.8,
            }

        try:
            # Map task to opencode command
            cmd = self._task_to_command(task, belief_state)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=belief_state.get("working_directory", "."),
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": " ".join(cmd),
                "prediction_error": 0.0 if result.returncode == 0 else 0.5,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out",
                "prediction_error": 0.9,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "prediction_error": 0.8}

    def _task_to_command(self, task: str, belief_state: Dict[str, Any]) -> list:
        """Convert natural language task to opencode command."""
        task_lower = task.lower()

        if "list" in task_lower and "file" in task_lower:
            return [self.opencode_path, "ls"]
        elif "search" in task_lower or "find" in task_lower:
            pattern = belief_state.get("search_pattern", ".*")
            return [self.opencode_path, "grep", pattern]
        elif "read" in task_lower:
            file_path = belief_state.get("target_file", "")
            return (
                [self.opencode_path, "read", file_path]
                if file_path
                else [self.opencode_path, "ls"]
            )
        elif "run" in task_lower or "execute" in task_lower:
            cmd = belief_state.get("command", "ls")
            return [self.opencode_path, "bash", cmd]
        else:
            # Default to question/ask interface
            return [self.opencode_path, "question", task]


class SimplifiedLRSAgent:
    """Simplified LRS-style agent for demonstration."""

    def __init__(self, tools: list):
        self.tools = tools
        self.belief_state = {
            "goal": "",
            "current_task": "",
            "precision": 0.8,
            "adaptation_count": 0,
        }

    def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute a task using available tools."""
        self.belief_state["goal"] = task
        self.belief_state["current_task"] = task

        print(f"🤖 LRS Agent executing: {task}")
        print(f"🎯 Current precision: {self.belief_state['precision']}")

        # Try tools in sequence (simplified policy selection)
        for tool in self.tools:
            print(f"🔧 Using tool: {tool.name}")

            result = tool.get(self.belief_state)

            if result["success"]:
                print("✅ Tool succeeded")
                self._update_precision(result["prediction_error"])

                # Update belief state
                self.belief_state = tool.set(self.belief_state, result)

                return {
                    "success": True,
                    "result": result,
                    "final_state": self.belief_state,
                }
            else:
                print(f"❌ Tool failed: {result.get('error', 'Unknown error')}")
                self._update_precision(result["prediction_error"])

        return {
            "success": False,
            "error": "All tools failed",
            "final_state": self.belief_state,
        }

    def _update_precision(self, prediction_error: float):
        """Update precision based on prediction error (simplified)."""
        # Simplified precision update (normally uses Beta distribution)
        if prediction_error > 0.7:
            self.belief_state["precision"] *= 0.9  # Decrease precision
            self.belief_state["adaptation_count"] += 1
        elif prediction_error < 0.3:
            self.belief_state["precision"] = min(
                1.0, self.belief_state["precision"] * 1.05
            )


def demo_integration():
    """Demonstrate OpenCode ↔ LRS integration."""
    print("🚀 OpenCode ↔ LRS-Agents Integration Demo")
    print("=" * 50)

    # Create OpenCode tool
    opencode_tool = OpenCodeTool()

    if not opencode_tool.opencode_path:
        print("❌ OpenCode not found. Please ensure it's installed and in PATH.")
        return

    print(f"✅ Found OpenCode at: {opencode_tool.opencode_path}")

    # Create simplified LRS agent with OpenCode tool
    agent = SimplifiedLRSAgent(tools=[opencode_tool])

    # Test tasks
    test_tasks = [
        "list files in current directory",
        "search for TODO comments",
        "read the README file",
    ]

    for task in test_tasks:
        print(f"\n{'=' * 30}")
        result = agent.execute_task(task)

        if result["success"]:
            output = result["result"].get("stdout", "").strip()
            if output:
                print("📄 Output (first 200 chars):")
            print(output[:200] + "..." if len(output) > 200 else output)
        else:
            print(f"💥 Task failed: {result.get('error', 'Unknown error')}")

        print(f"📊 Final precision: {agent.belief_state['precision']:.2f}")
        print(f"🔄 Adaptations: {agent.belief_state['adaptation_count']}")


def demo_bidirectional():
    """Show bidirectional communication concepts."""
    print(f"\n{'=' * 50}")
    print("🔄 Bidirectional Integration Concepts")
    print("=" * 50)

    concepts = [
        "1. OpenCode calls LRS agents via HTTP API for complex reasoning",
        "2. LRS agents use OpenCode tools for file operations and code analysis",
        "3. Real-time WebSocket communication for interactive sessions",
        "4. Shared belief states between systems",
        "5. Precision feedback loops improving both systems",
    ]

    for concept in concepts:
        print(f"🔗 {concept}")

    print("\n💡 Benefits:")
    print("   • Active Inference for resilient task execution")
    print("   • Precision tracking and adaptation")
    print("   • Goal-directed behavior with epistemic exploration")
    print("   • Hierarchical planning from abstract to concrete actions")


if __name__ == "__main__":
    demo_integration()
    demo_bidirectional()
