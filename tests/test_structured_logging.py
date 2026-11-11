"""Behavioral tests for structured JSONL logging and handler lifecycle."""

import json
import logging
import uuid

import pytest

from lrs.monitoring.structured_logging import LRSLogger, create_logger_for_agent


@pytest.fixture
def event_log(tmp_path):
    agent_id = "test_agent_" + uuid.uuid4().hex
    path = tmp_path / "nested" / "agent.jsonl"
    logger = LRSLogger(agent_id=agent_id, log_file=str(path), console=False)
    yield logger, path
    for handler in list(logger.logger.handlers):
        logger.logger.removeHandler(handler)
        handler.close()


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_precision_update_is_structured_and_rounded(event_log):
    logger, path = event_log
    logger.log_precision_update(
        level="execution",
        old_value=0.8,
        new_value=0.4,
        prediction_error=0.95555,
        propagated=True,
    )
    events = _events(path)
    assert len(events) == 1
    event = events[0]
    assert event["agent_id"] == logger.agent_id
    assert event["session_id"] == logger.session_id
    assert event["event_type"] == "precision_update"
    assert event["data"]["delta"] == pytest.approx(-0.4)
    assert event["data"]["prediction_error"] == pytest.approx(0.9556)
    assert event["data"]["propagated"] is True


def test_policy_selection_records_scores(event_log):
    logger, path = event_log
    logger.log_policy_selection(
        policies=[["tool_a"], ["tool_b"]],
        selected_index=1,
        G_values=[2.5, -0.75],
        precision=0.65,
    )
    event = _events(path)[0]
    assert event["event_type"] == "policy_selection"
    assert event["data"]["num_policies"] == 2
    assert event["data"]["selected_index"] == 1
    assert event["data"]["selected_G"] == pytest.approx(-0.75)


def test_failed_tool_uses_warning_and_records_error(event_log):
    logger, path = event_log
    captured = []

    class CollectingHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.levelno)

    observer = CollectingHandler()
    logger.logger.addHandler(observer)
    try:
        logger.log_tool_execution(
            "fetch", success=False, execution_time=0.25,
            prediction_error=0.9, error_message="timeout"
        )
    finally:
        logger.logger.removeHandler(observer)
        observer.close()
    event = _events(path)[0]
    assert event["data"]["execution_time_ms"] == pytest.approx(250.0)
    assert event["data"]["error"] == "timeout"
    assert logging.WARNING in captured


def test_adaptation_and_error_events(event_log):
    logger, path = event_log
    logger.log_adaptation_event(
        "prediction_error", {"planning": 0.85}, {"planning": 0.65}, "replan"
    )
    logger.log_error("RuntimeError", "lost communication", "trace omitted")
    events = _events(path)
    assert [event["event_type"] for event in events] == ["adaptation", "error"]
    assert events[0]["data"]["action"] == "replan"
    assert events[1]["data"]["stack_trace"] == "trace omitted"


def test_performance_metrics_avoid_zero_division(event_log):
    logger, path = event_log
    logger.log_performance_metrics(
        total_steps=0, success_rate=0.0, avg_precision=0.5,
        adaptation_count=0, execution_time=0.0
    )
    event = _events(path)[0]
    assert event["data"]["steps_per_second"] == 0


def test_reconfiguring_agent_logger_closes_old_file_handler(tmp_path):
    agent_id = "reused_agent_" + uuid.uuid4().hex
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first = create_logger_for_agent(agent_id, log_file=str(first_path), console=False)
    first_handler = first.logger.handlers[0]
    first.log_error("test", "first")
    second = create_logger_for_agent(agent_id, log_file=str(second_path), console=False)
    try:
        assert first_handler.stream is None
        assert first_handler not in second.logger.handlers
        second.log_error("test", "second")
        assert [e["data"]["message"] for e in _events(first_path)] == ["first"]
        assert [e["data"]["message"] for e in _events(second_path)] == ["second"]
    finally:
        for handler in list(second.logger.handlers):
            second.logger.removeHandler(handler)
            handler.close()
