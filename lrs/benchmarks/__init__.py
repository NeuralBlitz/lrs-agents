"""Benchmark suites for LRS-Agents."""

from lrs.benchmarks.chaos_scriptorium import (
    ChaosScriptoriumBenchmark,
    run_benchmark,
)
from lrs.benchmarks.gaia_benchmark import (
    GAIABenchmark,
    run_gaia_benchmark,
)

__all__ = [
    "ChaosScriptoriumBenchmark",
    "run_benchmark",
    "GAIABenchmark",
    "run_gaia_benchmark",
]
