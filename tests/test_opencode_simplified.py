"""OpenCode adapter checks that do not invoke a real external process."""

import subprocess
from types import SimpleNamespace

import pytest

from lrs.opencode.simplified_integration import OpenCodeTool, SimplifiedLRSAgent


def _new_tool_without_probe():
    tool = object.__new__(OpenCodeTool)
    tool.opencode_path = "opencode"
    return tool


def test_missing_opencode_binary_is_reported(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("binary absent")

    monkeypatch.setattr(subprocess, "run", unavailable)
    tool = OpenCodeTool()

    assert tool.opencode_path is None
    result = tool.get({"current_task": "list files"})
    assert result["success"] is False
    assert result["prediction_error"] == 1.0


def test_probe_can_fall_back_to_local_binary(monkeypatch):
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=1 if cmd[0] == "opencode" else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    tool = OpenCodeTool()

    assert tool.opencode_path == "./node_modules/.bin/opencode"
    assert calls == [
        ["opencode", "--version"],
        ["./node_modules/.bin/opencode", "--version"],
    ]


def test_probe_timeout_does_not_block_second_candidate(monkeypatch):
    def fake_run(cmd, **_kwargs):
        if cmd[0] == "opencode":
            raise subprocess.TimeoutExpired(cmd, timeout=5)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert OpenCodeTool().opencode_path == "./node_modules/.bin/opencode"


def test_keyboard_interrupt_during_probe_propagates(monkeypatch):
    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        OpenCodeTool()


@pytest.mark.parametrize(
    "task, state, expected",
    [
        ("list files", {}, ["opencode", "ls"]),
        ("find usage", {"search_pattern": "TODO"}, ["opencode", "grep", "TODO"]),
        ("read document", {"target_file": "doc.txt"}, ["opencode", "read", "doc.txt"]),
        ("execute task", {"command": "ls"}, ["opencode", "bash", "ls"]),
        ("explain Python", {}, ["opencode", "question", "explain Python"]),
    ],
)
def test_task_mapping_is_argument_based(task, state, expected):
    tool = _new_tool_without_probe()
    assert tool._task_to_command(task, state) == expected


def test_get_invokes_argument_list_without_shell(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tool = OpenCodeTool()
    result = tool.get({"current_task": "list files", "working_directory": "/tmp"})

    assert result["success"] is True
    assert result["prediction_error"] == 0.0
    assert result["stdout"] == "ok"
    assert calls[-1][0] == ["opencode", "ls"]
    assert calls[-1][1]["cwd"] == "/tmp"
    assert calls[-1][1].get("shell") is not True


def test_get_handles_subprocess_timeout(monkeypatch):
    def fake_run(cmd, **_kwargs):
        if cmd[-1] == "--version":
            return SimpleNamespace(returncode=0)
        raise subprocess.TimeoutExpired(cmd, timeout=15)

    monkeypatch.setattr(subprocess, "run", fake_run)
    tool = OpenCodeTool()
    result = tool.get({"current_task": "list files"})

    assert result["success"] is False
    assert result["prediction_error"] == 0.9
    assert result["error"] == "Command timed out"


def test_missing_task_is_not_executed(monkeypatch):
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    tool = OpenCodeTool()
    result = tool.get({})

    assert result["success"] is False
    assert result["prediction_error"] == 0.8
    assert calls == [["opencode", "--version"]]


def test_agent_updates_precision_on_failure_then_success(capsys):
    class FailingTool:
        name = "failure"

        def get(self, _state):
            return {"success": False, "prediction_error": 0.9, "error": "failed"}

    class SucceedingTool:
        name = "success"

        def get(self, _state):
            return {"success": True, "prediction_error": 0.0, "value": 3}

        def set(self, state, value):
            return {**state, "result": value["value"]}

    agent = SimplifiedLRSAgent([FailingTool(), SucceedingTool()])
    result = agent.execute_task("compute")
    capsys.readouterr()

    assert result["success"] is True
    assert result["final_state"]["result"] == 3
    assert result["final_state"]["adaptation_count"] == 1
    assert result["final_state"]["precision"] == pytest.approx(0.8 * 0.9 * 1.05)
