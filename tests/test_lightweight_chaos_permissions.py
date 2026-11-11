"""Regression tests for lightweight Chaos Scriptorium lock behavior."""

from pathlib import Path

import pytest

from lrs.benchmarking.lightweight_benchmarks import LightweightChaosEnvironment


@pytest.mark.parametrize("interval", [0, -1])
def test_rejects_nonpositive_chaos_intervals(tmp_path, interval):
    with pytest.raises(ValueError, match="chaos_interval"):
        LightweightChaosEnvironment(root_dir=str(tmp_path), chaos_interval=interval)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_lock_probability(tmp_path, probability):
    with pytest.raises(ValueError, match="lock_probability"):
        LightweightChaosEnvironment(root_dir=str(tmp_path), lock_probability=probability)


def test_lock_and_unlock_preserve_directory_traversal(tmp_path, monkeypatch):
    env = LightweightChaosEnvironment(root_dir=str(tmp_path))
    calls = []
    monkeypatch.setattr(
        "lrs.benchmarking.lightweight_benchmarks.os.chmod",
        lambda path, mode: calls.append((path, mode)),
    )
    env.setup()
    calls.clear()

    env._set_permissions(readable=False)
    assert calls == [(env.vault_dir, 0o755), (env.key_path, 0o000)]

    calls.clear()
    env._set_permissions(readable=True)
    assert calls == [(env.vault_dir, 0o755), (env.key_path, 0o644)]


def test_tick_locks_on_configured_interval_and_reset_unlocks(tmp_path, monkeypatch):
    root = tmp_path / "chaos"
    env = LightweightChaosEnvironment(
        root_dir=str(root), chaos_interval=2, lock_probability=1.0
    )
    calls = []
    monkeypatch.setattr(
        "lrs.benchmarking.lightweight_benchmarks.os.chmod",
        lambda path, mode: calls.append((path, mode)),
    )
    env.setup()

    env.tick()
    assert env.step_count == 1
    assert env.is_locked() is False

    env.tick()
    assert env.step_count == 2
    assert env.is_locked() is True
    assert calls[-1] == (env.key_path, 0o000)

    env.reset()
    assert env.step_count == 0
    assert env.is_locked() is False
    assert calls[-1] == (env.key_path, 0o644)


def test_cleanup_reopens_locked_file_before_removing_directory(tmp_path, monkeypatch):
    root = tmp_path / "chaos"
    env = LightweightChaosEnvironment(root_dir=str(root))
    env.setup()
    env.locked = True
    events = []

    def fake_permissions(readable):
        events.append(("permissions", readable))

    def fake_delete(path):
        events.append(("rmtree", path))

    monkeypatch.setattr(env, "_set_permissions", fake_permissions)
    monkeypatch.setattr(
        "lrs.benchmarking.lightweight_benchmarks.shutil.rmtree", fake_delete
    )
    env.cleanup()

    assert events == [("permissions", True), ("rmtree", str(root))]
