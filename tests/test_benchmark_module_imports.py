"""Regression checks for benchmark package imports and analysis utilities."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "lrs.benchmarking.comprehensive_benchmark_runner",
        "lrs.benchmarking.lightweight_benchmarks",
        "lrs.enterprise.performance_optimization",
        "lrs.opencode.lrs_opencode_integration",
    ],
)
def test_installed_package_qualified_benchmark_modules_import(module_name):
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name


def test_chaos_performance_calculates_from_observed_results():
    runner = importlib.import_module("lrs.benchmarking.comprehensive_benchmark_runner")
    results = {
        "success_rate": 0.5,
        "all_results": [
            {
                "success": True,
                "execution_time": 1.0,
                "adaptations": 1,
                "final_precision": {"execution": 0.6},
            },
            {
                "success": False,
                "execution_time": 3.0,
                "adaptations": 0,
                "final_precision": {"execution": 0.4},
            },
        ],
    }

    metrics = runner.analyze_chaos_performance(results)

    assert metrics["resilience_score"] == pytest.approx(0.5)
    assert metrics["avg_success_time"] == pytest.approx(1.0)
    assert metrics["avg_failure_time"] == pytest.approx(3.0)
    assert metrics["adaptation_effectiveness"] == pytest.approx(0.5)


def test_gaia_performance_groups_task_difficulty():
    runner = importlib.import_module("lrs.benchmarking.comprehensive_benchmark_runner")
    metrics = runner.analyze_gaia_performance(
        {
            "success_rate": 0.5,
            "all_results": [
                {"success": True, "execution_time": 1.0, "precision_update": 0.1, "difficulty": "easy"},
                {"success": False, "execution_time": 3.0, "precision_update": 0.3, "difficulty": "easy"},
            ],
        }
    )
    assert metrics["reasoning_complexity_score"] == pytest.approx(0.5)
    assert metrics["difficulty_adaptation"]["easy"] == pytest.approx(0.5)
    assert metrics["learning_efficiency"] == pytest.approx(0.5)


def test_json_serialize_helper_preserves_primitives():
    runner = importlib.import_module("lrs.benchmarking.comprehensive_benchmark_runner")
    assert runner.make_json_serializable({"ok": True, "values": [1, None]}) == {
        "ok": True,
        "values": [1, None],
    }
