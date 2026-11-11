"""Regression tests for safe benchmark arithmetic and scoped exceptions."""

import subprocess

import pytest

from lrs.benchmarking.lightweight_benchmarks import (
    CalculatorTool,
    FileReadTool,
    LightweightChaosEnvironment,
    _evaluate_arithmetic,
)
from lrs.benchmarks.gaia_benchmark import GAIABenchmark


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(7 - 2) / 2", 2.5),
        ("2 ** 4", 16),
        ("-2 + 5", 3),
        ("9 // 4", 2),
        ("9 % 4", 1),
    ],
)
def test_calculator_evaluates_bounded_arithmetic(expression, expected):
    tool = CalculatorTool()
    result = tool.execute(expression)
    assert result["success"] is True
    assert result["result"] == pytest.approx(expected)
    assert tool.call_count == 1
    assert tool.success_count == 1


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo no')",
        "(1).__class__",
        "abs(-3)",
        "2 ** 100",
        "1 / 0",
        "1 +",
        "x + 2",
        "1" * 300,
        "True",
    ],
)
def test_calculator_rejects_executable_or_unbounded_expressions(expression):
    tool = CalculatorTool()
    result = tool.execute(expression)
    assert result["success"] is False
    assert result["prediction_error"] == pytest.approx(0.8)
    assert tool.call_count == 1
    assert tool.success_count == 0


def test_calculator_does_not_suppress_keyboard_interrupt(monkeypatch):
    def interruption(_expression):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "lrs.benchmarking.lightweight_benchmarks._evaluate_arithmetic", interruption
    )
    with pytest.raises(KeyboardInterrupt):
        CalculatorTool().execute("1 + 1")


def test_calculator_direct_helper_only_returns_numbers():
    assert _evaluate_arithmetic("3 - 5") == -2
    with pytest.raises(ValueError):
        _evaluate_arithmetic("[1, 2, 3]")


def test_file_read_missing_file_is_normal_failure(tmp_path):
    tool = FileReadTool()
    result = tool.execute(str(tmp_path / "does-not-exist.txt"))
    assert result["success"] is False
    assert tool.call_count == 1
    assert tool.success_count == 0


def test_file_read_existing_file_succeeds(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("safe data", encoding="utf-8")
    tool = FileReadTool()
    result = tool.execute(str(file))
    assert result["success"] is True
    assert result["content"] == "safe data"


def test_cleanup_handles_os_errors_but_not_keyboard_interrupt(tmp_path, monkeypatch):
    env = LightweightChaosEnvironment(root_dir=str(tmp_path))

    def fail_os(_path):
        raise PermissionError("expected failure")

    monkeypatch.setattr("lrs.benchmarking.lightweight_benchmarks.shutil.rmtree", fail_os)
    env.cleanup()

    def interrupt(_path):
        raise KeyboardInterrupt()

    monkeypatch.setattr("lrs.benchmarking.lightweight_benchmarks.shutil.rmtree", interrupt)
    with pytest.raises(KeyboardInterrupt):
        env.cleanup()


@pytest.mark.parametrize(
    ("predicted", "expected", "matches"),
    [
        ("100.5", "100", True),
        ("105", "100", False),
        ("1e-9", "0", True),
        ("1e-7", "0", False),
        ("42", "42", True),
        ("not a number", "10", False),
    ],
)
def test_gaia_numeric_answer_tolerance_handles_zero(predicted, expected, matches):
    benchmark = object.__new__(GAIABenchmark)
    assert benchmark._check_answer(predicted, expected) is matches
