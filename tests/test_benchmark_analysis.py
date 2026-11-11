"""Regression coverage for the package's lightweight benchmark analysis helpers."""

import pytest

from lrs.benchmarking.comprehensive_benchmark_runner import (
    analyze_chaos_performance,
    analyze_gaia_performance,
    analyze_overall_performance,
    make_json_serializable,
)


def test_chaos_summary_uses_real_trials():
    result = {
        "success_rate": 0.5,
        "all_results": [
            {
                "success": True,
                "execution_time": 2.0,
                "final_precision": {"execution": 0.8},
                "adaptations": 1,
            },
            {
                "success": False,
                "execution_time": 4.0,
                "final_precision": {"execution": 0.4},
                "adaptations": 0,
            },
        ],
    }
    summary = analyze_chaos_performance(result)
    assert summary["resilience_score"] == pytest.approx(0.5)
    assert summary["avg_success_time"] == pytest.approx(2.0)
    assert summary["avg_failure_time"] == pytest.approx(4.0)
    assert summary["adaptation_effectiveness"] == pytest.approx(0.5)
    assert summary["precision_stability"] > 0


def test_chaos_summary_handles_empty_trials():
    result = {"success_rate": 0.0, "all_results": []}
    summary = analyze_chaos_performance(result)
    assert summary["adaptation_effectiveness"] == 0.0
    assert summary["avg_success_time"] == 0.0
    assert summary["precision_stability"] == 0.0


def test_gaia_summary_tracks_difficulty_and_learning():
    result = {
        "success_rate": 0.5,
        "all_results": [
            {
                "difficulty": "easy",
                "success": True,
                "precision_update": 0.1,
                "execution_time": 1.0,
            },
            {
                "difficulty": "hard",
                "success": False,
                "precision_update": 0.5,
                "execution_time": 3.0,
            },
        ],
    }
    summary = analyze_gaia_performance(result)
    assert summary["reasoning_complexity_score"] == 0.5
    assert summary["difficulty_adaptation"] == {"easy": 1.0, "hard": 0.0}
    assert summary["avg_precision_update"] == pytest.approx(0.3)
    assert summary["learning_efficiency"] == pytest.approx(0.5)
    assert summary["execution_time_variance"] > 0


def test_gaia_summary_handles_empty_trials():
    summary = analyze_gaia_performance({"success_rate": 0.0, "all_results": []})
    assert summary["learning_efficiency"] == 0.0
    assert summary["execution_time_variance"] == 0.0
    assert summary["difficulty_adaptation"] == {}


def test_overall_summary_uses_actual_benchmark_measurements():
    results = {
        "chaos_scriptorium": {
            "results": {"success_rate": 0.8},
            "performance_metrics": {
                "precision_stability": 0.7,
                "adaptation_effectiveness": 0.6,
            },
        },
        "gaia": {
            "results": {"success_rate": 0.6},
            "performance_metrics": {
                "avg_precision_update": 0.5,
                "learning_efficiency": 0.4,
            },
        },
        "code_analysis_performance": {"avg_analysis_time": 2.0},
    }
    summary = analyze_overall_performance(results)
    assert summary["overall_success_rate"] == pytest.approx(0.72)
    assert summary["avg_precision"] == pytest.approx(0.6)
    assert summary["adaptation_effectiveness"] == pytest.approx(0.5)
    assert summary["benchmark_efficiency"] == pytest.approx(24.11 / 2.0)
    assert summary["phase2_targets_met"]["success_rate_target"] is True


def test_json_serialization_preserves_simple_data():
    payload = {"value": 3.0, "nested": [None, True, {"value": 2}], "other": (1, 2)}
    result = make_json_serializable(payload)
    assert result["value"] == 3.0
    assert result["nested"] == [None, True, {"value": 2}]
    assert result["other"] == "(1, 2)"
