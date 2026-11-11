"""Regression tests for the NeuralBlitz in-process message bus."""

import asyncio
from datetime import datetime, timedelta

import pytest

from lrs.neuralblitz_integration.messaging import (
    Message,
    MessageType,
    UnifiedMessageBus,
)


def test_message_round_trip_preserves_input_data():
    original = Message(
        type=MessageType.SYNC_REQUEST,
        source="agent_a",
        destination="agent_b",
        payload={"request": "state"},
        correlation_id="trace-1",
    )
    data = original.to_dict()
    snapshot = dict(data)

    restored = Message.from_dict(data)

    assert data == snapshot  # from_dict must not change the caller's dict
    assert data["type"] == "sync_request"
    assert isinstance(data["timestamp"], str)
    assert restored.to_dict() == snapshot
    assert restored.type is MessageType.SYNC_REQUEST


def test_message_from_dict_accepts_already_parsed_fields():
    original = Message(type=MessageType.HEARTBEAT)
    data = original.to_dict()
    data["type"] = MessageType.HEARTBEAT
    data["timestamp"] = original.timestamp

    restored = Message.from_dict(data)

    assert restored.type is MessageType.HEARTBEAT
    assert restored.timestamp == original.timestamp


def test_expiration_and_retry_bounds():
    old = datetime.utcnow() - timedelta(minutes=2)
    expired = Message(timestamp=old, ttl=30)
    assert expired.is_expired() is True
    assert Message(ttl=None).is_expired() is False

    retryable = Message(max_retries=2, retry_count=1)
    exhausted = Message(max_retries=2, retry_count=2)
    assert retryable.can_retry() is True
    assert exhausted.can_retry() is False


@pytest.mark.asyncio
async def test_expired_message_is_not_queued():
    bus = UnifiedMessageBus()
    expired = Message(timestamp=datetime.utcnow() - timedelta(minutes=2), ttl=1)

    assert await bus.publish(expired) is False
    assert bus.get_stats()["messages_sent"] == 0
    assert bus.get_stats()["messages_failed"] == 1
    assert bus.get_stats()["queue_size"] == 0


@pytest.mark.asyncio
async def test_subscription_and_unsubscription_are_idempotent():
    bus = UnifiedMessageBus()

    def callback(message):
        return None

    await bus.subscribe(MessageType.HEARTBEAT, callback)
    await bus.subscribe(MessageType.HEARTBEAT, callback)
    assert bus.get_stats()["subscribers"] == 1

    await bus.unsubscribe(MessageType.ERROR, callback)
    await bus.unsubscribe(MessageType.HEARTBEAT, callback)
    await bus.unsubscribe(MessageType.HEARTBEAT, callback)
    assert bus.get_stats()["subscribers"] == 0


@pytest.mark.asyncio
async def test_async_delivery_updates_stats_and_queue_join():
    bus = UnifiedMessageBus()
    received = []

    async def callback(message):
        received.append(message.payload["value"])

    await bus.subscribe(MessageType.SYNC_REQUEST, callback)
    await bus.start()
    try:
        assert await bus.publish(
            Message(type=MessageType.SYNC_REQUEST, payload={"value": 42})
        )
        await asyncio.wait_for(bus._message_queue.join(), timeout=3)
    finally:
        await bus.stop()

    assert received == [42]
    assert bus.get_stats()["messages_sent"] == 1
    assert bus.get_stats()["messages_received"] == 1
    assert bus.get_stats()["messages_failed"] == 0
    assert bus.get_stats()["running"] is False


@pytest.mark.asyncio
async def test_callback_failure_does_not_stop_other_subscribers():
    bus = UnifiedMessageBus()
    delivered = []

    def bad_callback(_message):
        raise RuntimeError("expected callback failure")

    async def good_callback(message):
        delivered.append(message.id)

    await bus.subscribe(MessageType.HEARTBEAT, bad_callback)
    await bus.subscribe(MessageType.HEARTBEAT, good_callback)
    message = Message()
    await bus._process_message(message)

    assert delivered == [message.id]
    assert bus.get_stats()["messages_received"] == 1
    assert bus.get_stats()["messages_failed"] == 1


@pytest.mark.asyncio
async def test_sync_subscriber_is_dispatched():
    bus = UnifiedMessageBus()
    received = []

    def callback(message):
        received.append(message.source)

    await bus.subscribe(MessageType.ERROR, callback)
    await bus._process_message(Message(type=MessageType.ERROR, source="agent"))
    assert received == ["agent"]


@pytest.mark.asyncio
async def test_history_limits_and_clear():
    bus = UnifiedMessageBus()
    first = Message(source="first")
    second = Message(source="second")

    assert await bus.publish(first) is True
    assert await bus.publish(second) is True
    assert bus.get_message_history(limit=1) == [second]
    assert bus.get_message_history(limit=0) == []
    with pytest.raises(ValueError, match="nonnegative"):
        bus.get_message_history(limit=-1)

    await bus.clear_history()
    assert bus.get_message_history() == []
    assert bus.get_stats()["message_history_size"] == 0


@pytest.mark.asyncio
async def test_message_factory_helpers():
    bus = UnifiedMessageBus()
    heartbeat = await bus.create_heartbeat("agent_a")
    error = await bus.create_error("agent_a", "timeout", correlation_id="r1")

    assert heartbeat.type is MessageType.HEARTBEAT
    assert heartbeat.payload == {"status": "alive"}
    assert error.type is MessageType.ERROR
    assert error.payload == {"error": "timeout"}
    assert error.correlation_id == "r1"


@pytest.mark.asyncio
async def test_start_stop_can_be_called_twice():
    bus = UnifiedMessageBus()
    await bus.start()
    await bus.start()
    assert bus.get_stats()["running"] is True
    await bus.stop()
    await bus.stop()
    assert bus.get_stats()["running"] is False
