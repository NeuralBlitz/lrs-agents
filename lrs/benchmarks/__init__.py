"""
Benchmark suite for LRS-Agents.

This module provides:
- Chaos Scriptorium: Volatile environment benchmark
- GAIA: General AI Assistants benchmark
- Performance comparison utilities
"""

from lrs.benchmarks.chaos_scriptorium import (
    ChaosEnvironment,
    ChaosScriptoriumBenchmark,
    run_chaos_benchmark
)
from lrs.benchmarks.gaia_benchmark import (
    GAIATask,
    GAIAToolkit,
    GAIABenchmark
)

__all__ = [
    "ChaosEnvironment",
    "ChaosScriptoriumBenchmark",
    "run_chaos_benchmark",
    "GAIATask",
    "GAIAToolkit",
    "GAIABenchmark",
]
