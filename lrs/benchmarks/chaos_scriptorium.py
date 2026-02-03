"""
Chaos Scriptorium: Benchmark for volatile environments.

Tests agent resilience when environment behavior changes unpredictably.
Goal: Find secret key in directory tree with randomly changing permissions.
"""

import os
import random
import tempfile
import shutil
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import subprocess

from lrs.core.lens import ToolLens, ExecutionResult
from lrs.core.registry import ToolRegistry
from lrs.monitoring.tracker import LRSStateTracker


@dataclass
class ChaosConfig:
    """
    Configuration for Chaos Scriptorium environment.

    Attributes:
        chaos_interval: Steps between permission changes
        lock_probability: Probability of locking on chaos tick
        num_directories: Number of nested directories
        num_decoy_files: Number of fake key files
        timeout_seconds: Maximum time for benchmark
    """

    chaos_interval: int = 3
    lock_probability: float = 0.5
    num_directories: int = 3
    num_decoy_files: int = 5
    timeout_seconds: int = 60


class ChaosEnvironment:
    """
    Volatile file system environment.

    Creates a directory structure with:
    - Secret key at known location
    - Decoy files to confuse agents
    - Permissions that randomly flip between READABLE and LOCKED

    Examples:
        >>> env = ChaosEnvironment(root_dir="/tmp/chaos")
        >>> env.setup()
        >>>
        >>> # Execute steps
        >>> for step in range(10):
        ...     env.tick()  # Maybe change permissions
        ...     if env.is_locked():
        ...         print(f"Step {step}: Files are LOCKED")
        ...     else:
        ...         print(f"Step {step}: Files are READABLE")
    """

    def __init__(
        self, root_dir: Optional[str] = None, chaos_interval: int = 3, lock_probability: float = 0.5
    ):
        """
        Initialize chaos environment.

        Args:
            root_dir: Root directory (creates temp if None)
            chaos_interval: Steps between chaos ticks
            lock_probability: P(lock) on chaos tick
        """
        if root_dir is None:
            root_dir = tempfile.mkdtemp(prefix="chaos_scriptorium_")

        self.root_dir = root_dir
        self.chaos_interval = chaos_interval
        self.lock_probability = lock_probability

        self.step_count = 0
        self.locked = False

        # Paths
        self.vault_dir = os.path.join(root_dir, "data", "vault")
        self.key_path = os.path.join(self.vault_dir, "key.txt")
        self.secret_key = f"SECRET_KEY_{random.randint(1000, 9999)}"

    def setup(self):
        """Create directory structure and secret key"""
        # Create directories
        os.makedirs(self.vault_dir, exist_ok=True)

        # Write secret key
        with open(self.key_path, "w") as f:
            f.write(self.secret_key)

        # Create decoy files
        for i in range(5):
            decoy_path = os.path.join(self.root_dir, "data", f"decoy_{i}.txt")
            with open(decoy_path, "w") as f:
                f.write(f"DECOY_KEY_{random.randint(1000, 9999)}")

        # Initial state: unlocked
        self.locked = False
        self._set_permissions(readable=True)

    def tick(self):
        """
        Advance one step. Maybe trigger chaos.
        """
        self.step_count += 1

        # Check if chaos should occur
        if self.step_count % self.chaos_interval == 0:
            self._trigger_chaos()

    def _trigger_chaos(self):
        """Randomly change permissions"""
        if random.random() < self.lock_probability:
            # Lock files
            self.locked = True
            self._set_permissions(readable=False)
        else:
            # Unlock files
            self.locked = False
            self._set_permissions(readable=True)

    def _set_permissions(self, readable: bool):
        """Set file permissions"""
        if readable:
            # Make readable
            os.chmod(self.vault_dir, 0o755)
            os.chmod(self.key_path, 0o644)
        else:
            # Make locked (no read permission)
            os.chmod(self.vault_dir, 0o000)
            os.chmod(self.key_path, 0o000)

    def is_locked(self) -> bool:
        """Check if files are currently locked"""
        return self.locked

    def reset(self):
        """Reset environment state"""
        self.step_count = 0
        self.locked = False
        try:
            self._set_permissions(readable=True)
        except PermissionError:
            # If locked, can't chmod - just reset internal state
            pass

    def cleanup(self):
        """Remove temporary directory"""
        try:
            shutil.rmtree(self.root_dir)
        except:
            pass


# Tool implementations for Chaos Scriptorium


class ShellTool(ToolLens):
    """
    Execute shell commands.

    Performance under lock:
    - Unlocked: 95% success
    - Locked: 40% success (struggles with permissions)
    """

    def __init__(self, env: ChaosEnvironment):
        super().__init__(
            name="shell_exec",
            input_schema={
                "type": "object",
                "required": ["command"],
                "properties": {"command": {"type": "string"}},
            },
            output_schema={"type": "string"},
        )
        self.env = env

    def get(self, state: dict) -> ExecutionResult:
        self.call_count += 1
        command = state.get("command", "")

        # Simulate higher failure rate when locked
        if self.env.is_locked() and random.random() < 0.6:
            self.failure_count += 1
            return ExecutionResult(
                success=False, value=None, error="Permission denied", prediction_error=0.9
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.env.root_dir,
            )

            success = result.returncode == 0
            if not success:
                self.failure_count += 1

            return ExecutionResult(
                success=success,
                value=result.stdout if success else None,
                error=result.stderr if not success else None,
                prediction_error=0.05 if success else 0.8,
            )
        except Exception as e:
            self.failure_count += 1
            return ExecutionResult(success=False, value=None, error=str(e), prediction_error=0.95)

    def set(self, state: dict, observation: str) -> dict:
        return {**state, "shell_output": observation}


class PythonTool(ToolLens):
    """
    Execute Python code.

    Performance under lock:
    - Unlocked: 90% success
    - Locked: 80% success (better than shell)
    """

    def __init__(self, env: ChaosEnvironment):
        super().__init__(
            name="python_exec",
            input_schema={
                "type": "object",
                "required": ["code"],
                "properties": {"code": {"type": "string"}},
            },
            output_schema={"type": "string"},
        )
        self.env = env

    def get(self, state: dict) -> ExecutionResult:
        self.call_count += 1
        code = state.get("code", "")

        # Python is more resilient to locks
        if self.env.is_locked() and random.random() < 0.2:
            self.failure_count += 1
            return ExecutionResult(
                success=False, value=None, error="Access error", prediction_error=0.7
            )

        try:
            # Execute in restricted namespace
            namespace = {"__builtins__": __builtins__, "os": os, "open": open, "Path": Path}
            exec(code, namespace)
            result = namespace.get("result", "Executed")

            return ExecutionResult(
                success=True, value=str(result), error=None, prediction_error=0.1
            )
        except Exception as e:
            self.failure_count += 1
            return ExecutionResult(success=False, value=None, error=str(e), prediction_error=0.8)

    def set(self, state: dict, observation: str) -> dict:
        return {**state, "python_output": observation}


class FileReadTool(ToolLens):
    """
    Direct file reading.

    Performance under lock:
    - Unlocked: 100% success
    - Locked: 0% success (completely fails)
    """

    def __init__(self, env: ChaosEnvironment):
        super().__init__(
            name="file_read",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
            output_schema={"type": "string"},
        )
        self.env = env

    def get(self, state: dict) -> ExecutionResult:
        self.call_count += 1
        path = state.get("path", "")

        # Completely fails when locked
        if self.env.is_locked():
            self.failure_count += 1
            return ExecutionResult(
                success=False, value=None, error="File locked", prediction_error=1.0
            )

        try:
            content = Path(path).read_text()
            return ExecutionResult(success=True, value=content, error=None, prediction_error=0.0)
        except Exception as e:
            self.failure_count += 1
            return ExecutionResult(success=False, value=None, error=str(e), prediction_error=0.95)

    def set(self, state: dict, observation: str) -> dict:
        return {**state, "file_content": observation}


class ChaosScriptoriumBenchmark:
    """
    Full benchmark runner for Chaos Scriptorium.

    Examples:
        >>> from lrs import create_lrs_agent
        >>> from langchain_anthropic import ChatAnthropic
        >>>
        >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        >>>
        >>> benchmark = ChaosScriptoriumBenchmark(llm=llm)
        >>> results = benchmark.run(num_trials=10)
        >>>
        >>> print(f"Success rate: {results['success_rate']:.1%}")
        >>> print(f"Avg steps: {results['avg_steps']:.1f}")
    """

    def __init__(self, llm: Any, config: Optional[ChaosConfig] = None):
        """
        Initialize benchmark.

        Args:
            llm: Language model for agent
            config: Optional chaos configuration
        """
        self.llm = llm
        self.config = config or ChaosConfig()

    def run_single_trial(self, max_steps: int = 20) -> Dict[str, Any]:
        """
        Run single trial.

        Args:
            max_steps: Maximum execution steps

        Returns:
            Trial results dict
        """
        # Create environment
        env = ChaosEnvironment(
            chaos_interval=self.config.chaos_interval, lock_probability=self.config.lock_probability
        )
        env.setup()

        # Create tools
        tools = [ShellTool(env), PythonTool(env), FileReadTool(env)]

        # Create registry
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)

        # Create LRS agent
        from lrs import create_lrs_agent

        tracker = LRSStateTracker()
        agent = create_lrs_agent(llm=self.llm, tools=tools, tracker=tracker)

        # Run agent
        start_time = time.time()

        state = {
            "messages": [{"role": "user", "content": f"Find the secret key at {env.key_path}"}],
            "belief_state": {"goal": "find_key", "target_path": env.key_path},
            "max_iterations": max_steps,
        }

        try:
            result = agent.invoke(state)
            execution_time = time.time() - start_time

            # Check if key was found
            tool_history = result.get("tool_history", [])
            found_key = False

            for entry in tool_history:
                if entry.get("success") and env.secret_key in str(entry.get("result", "")):
                    found_key = True
                    break

            # Count steps and adaptations
            steps = len(tool_history)
            adaptations = result.get("adaptation_count", 0)

            # Get precision trajectory
            precision_trajectory = tracker.get_precision_trajectory("execution")

            return {
                "success": found_key,
                "steps": steps,
                "adaptations": adaptations,
                "execution_time": execution_time,
                "precision_trajectory": precision_trajectory,
                "final_precision": result.get("precision", {}),
            }

        except Exception as e:
            return {
                "success": False,
                "steps": 0,
                "adaptations": 0,
                "execution_time": time.time() - start_time,
                "error": str(e),
            }

        finally:
            env.cleanup()

    def run(self, num_trials: int = 100) -> Dict[str, Any]:
        """
        Run full benchmark with multiple trials.

        Args:
            num_trials: Number of trials to run

        Returns:
            Aggregate results
        """
        print(f"Running Chaos Scriptorium benchmark ({num_trials} trials)...")

        results = []
        for i in range(num_trials):
            if (i + 1) % 10 == 0:
                print(f"  Trial {i + 1}/{num_trials}...")

            trial_result = self.run_single_trial()
            results.append(trial_result)

        # Aggregate statistics
        successes = [r for r in results if r["success"]]
        success_rate = len(successes) / len(results)

        avg_steps = sum(r["steps"] for r in successes) / len(successes) if successes else 0
        avg_adaptations = (
            sum(r["adaptations"] for r in successes) / len(successes) if successes else 0
        )
        avg_time = sum(r["execution_time"] for r in results) / len(results)

        return {
            "success_rate": success_rate,
            "total_trials": num_trials,
            "successes": len(successes),
            "failures": num_trials - len(successes),
            "avg_steps": avg_steps,
            "avg_adaptations": avg_adaptations,
            "avg_execution_time": avg_time,
            "all_results": results,
        }


def run_chaos_benchmark(
    llm: Any, num_trials: int = 100, output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run Chaos Scriptorium benchmark.

    Args:
        llm: Language model
        num_trials: Number of trials
        output_file: Optional JSON output file

    Returns:
        Benchmark results

    Examples:
        >>> from langchain_anthropic import ChatAnthropic
        >>>
        >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        >>> results = run_chaos_benchmark(llm, num_trials=50)
        >>>
        >>> print(f"LRS Success Rate: {results['success_rate']:.1%}")
    """
    benchmark = ChaosScriptoriumBenchmark(llm)
    results = benchmark.run(num_trials)

    if output_file:
        import json

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("CHAOS SCRIPTORIUM RESULTS")
    print("=" * 60)
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Total Trials: {results['total_trials']}")
    print(f"Successes: {results['successes']}")
    print(f"Failures: {results['failures']}")
    print(f"Avg Steps (success): {results['avg_steps']:.1f}")
    print(f"Avg Adaptations: {results['avg_adaptations']:.1f}")
    print(f"Avg Execution Time: {results['avg_execution_time']:.2f}s")
    print("=" * 60)

    return results


# Allow running as script
if __name__ == "__main__":
    import sys

    # Check for LLM
    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    except:
        print("Error: Install langchain-anthropic to run benchmark")
        sys.exit(1)

    # Run benchmark
    results = run_chaos_benchmark(llm, num_trials=10)
