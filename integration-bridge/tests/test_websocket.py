"""
Tests for the WebSocket communication layer.
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import websockets

from opencode_lrs_bridge.websocket.manager import (
    WebSocketBridge,
    ConnectionManager,
    EventProcessor,
)
from tests.conftest import WebSocketTestCase


class TestConnectionManager(WebSocketTestCase):
    """Test connection management."""

    def setup_method(self):
        """Set up test method."""
        super().setup_method()
        self.connection_manager = ConnectionManager(self.config)

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful WebSocket connection."""
        success = await self.connection_manager.connect(
            self.websocket, self.connection_id, self.user_info
        )

        assert success is True
        assert self.connection_id in self.connection_manager.active_connections

        # Check if welcome message was sent
        self.websocket.send_text.assert_called()
        call_args = self.websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message["type"] == "connection_established"
        assert message["data"]["connection_id"] == self.connection_id

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test WebSocket disconnection."""
        # Connect first
        await self.connection_manager.connect(
            self.websocket, self.connection_id, self.user_info
        )

        # Disconnect
        await self.connection_manager.disconnect(self.connection_id)

        assert self.connection_id not in self.connection_manager.active_connections

        # Check if heartbeat task was cancelled
        # This would need more sophisticated mocking

    @pytest.mark.asyncio
    async def test_subscribe_to_agent(self):
        """Test agent subscription."""
        # Connect first
        await self.connection_manager.connect(
            self.websocket, self.connection_id, self.user_info
        )

        # Subscribe to agent
        agent_id = "test_agent_001"
        success = await self.connection_manager.subscribe_to_agent(
            self.connection_id, agent_id
        )

        assert success is True
        assert (
            agent_id
            in self.connection_manager.connection_subscriptions[self.connection_id]
        )
        assert self.connection_id in self.connection_manager.agent_subscribers[agent_id]

    @pytest.mark.asyncio
    async def test_unsubscribe_from_agent(self):
        """Test agent unsubscription."""
        # Connect and subscribe first
        await self.connection_manager.connect(
            self.websocket, self.connection_id, self.user_info
        )

        agent_id = "test_agent_001"
        await self.connection_manager.subscribe_to_agent(self.connection_id, agent_id)

        # Unsubscribe
        await self.connection_manager.unsubscribe_from_agent(
            self.connection_id, agent_id
        )

        assert (
            agent_id
            not in self.connection_manager.connection_subscriptions[self.connection_id]
        )
        assert (
            self.connection_id
            not in self.connection_manager.agent_subscribers[agent_id]
        )

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message to connection."""
        # Connect first
        await self.connection_manager.connect(
            self.websocket, self.connection_id, self.user_info
        )

        # Send message
        message_data = {"type": "test", "data": {"message": "Hello"}}
        success = await self.connection_manager.send_message(
            self.connection_id, message_data
        )

        assert success is True
        self.websocket.send_text.assert_called_with(json.dumps(message_data))

    @pytest.mark.asyncio
    async def test_send_message_to_nonexistent_connection(self):
        """Test sending message to non-existent connection."""
        message_data = {"type": "test", "data": {"message": "Hello"}}
        success = await self.connection_manager.send_message(
            "nonexistent_connection", message_data
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(self):
        """Test broadcasting message to agent subscribers."""
        # Set up multiple connections
        connection_ids = ["conn1", "conn2", "conn3"]
        websockets = []

        for conn_id in connection_ids:
            ws = Mock()
            ws.send_text = AsyncMock()
            websockets.append(ws)

            await self.connection_manager.connect(ws, conn_id, self.user_info)

        agent_id = "test_agent_001"
        for conn_id in connection_ids:
            await self.connection_manager.subscribe_to_agent(conn_id, agent_id)

        # Broadcast message
        message_data = {"type": "agent_update", "data": {"status": "active"}}
        await self.connection_manager.broadcast_to_subscribers(agent_id, message_data)

        # All subscribers should receive the message
        for ws in websockets:
            ws.send_text.assert_called_with(json.dumps(message_data))

    @pytest.mark.asyncio
    async def test_heartbeat_functionality(self):
        """Test heartbeat functionality."""
        # Connect
        await self.connection_manager.connect(
            self.websocket, self.connection_id, self.user_info
        )

        # Wait for heartbeat interval (reduced for testing)
        await asyncio.sleep(0.1)

        # Check if heartbeat message was sent
        heartbeat_calls = [
            call
            for call in self.websocket.send_text.call_args_list
            if json.loads(call[0][0])["type"] == "heartbeat"
        ]

        assert len(heartbeat_calls) >= 1


class TestEventProcessor(WebSocketTestCase):
    """Test event processing."""

    def setup_method(self):
        """Set up test method."""
        super().setup_method()
        self.connection_manager = Mock()
        self.event_processor = EventProcessor(self.config, self.connection_manager)

    @pytest.mark.asyncio
    async def test_process_subscribe_agent_message(self):
        """Test processing agent subscription message."""
        message = {"type": "subscribe_agent", "data": {"agent_id": "test_agent_001"}}

        await self.event_processor.process_message(self.connection_id, message)

        # Check if subscription was called
        self.connection_manager.subscribe_to_agent.assert_called_once_with(
            self.connection_id, "test_agent_001"
        )

        # Check if response was sent
        self.connection_manager.send_message.assert_called()
        call_args = self.connection_manager.send_message.call_args[0]
        response_message = call_args[1]
        assert response_message["type"] == "subscription_response"
        assert response_message["data"]["agent_id"] == "test_agent_001"

    @pytest.mark.asyncio
    async def test_process_unsubscribe_agent_message(self):
        """Test processing agent unsubscription message."""
        message = {"type": "unsubscribe_agent", "data": {"agent_id": "test_agent_001"}}

        await self.event_processor.process_message(self.connection_id, message)

        # Check if unsubscription was called
        self.connection_manager.unsubscribe_from_agent.assert_called_once_with(
            self.connection_id, "test_agent_001"
        )

    @pytest.mark.asyncio
    async def test_process_ping_message(self):
        """Test processing ping message."""
        message = {"type": "ping", "data": {}}

        await self.event_processor.process_message(self.connection_id, message)

        # Check if pong response was sent
        self.connection_manager.send_message.assert_called()
        call_args = self.connection_manager.send_message.call_args[0]
        response_message = call_args[1]
        assert response_message["type"] == "pong"
        assert "timestamp" in response_message["data"]

    @pytest.mark.asyncio
    async def test_process_invalid_message_type(self):
        """Test processing message with invalid type."""
        message = {"type": "invalid_type", "data": {}}

        await self.event_processor.process_message(self.connection_id, message)

        # Check if error response was sent
        self.connection_manager.send_message.assert_called()
        call_args = self.connection_manager.send_message.call_args[0]
        response_message = call_args[1]
        assert response_message["type"] == "error"
        assert "Unknown message type" in response_message["data"]["error"]

    @pytest.mark.asyncio
    async def test_process_message_without_type(self):
        """Test processing message without type."""
        message = {"data": {}}

        await self.event_processor.process_message(self.connection_id, message)

        # Check if error response was sent
        self.connection_manager.send_message.assert_called()
        call_args = self.connection_manager.send_message.call_args[0]
        response_message = call_args[1]
        assert response_message["type"] == "error"
        assert "Message type required" in response_message["data"]["error"]


class TestWebSocketBridge(WebSocketTestCase):
    """Test WebSocket bridge functionality."""

    def setup_method(self):
        """Set up test method."""
        super().setup_method()
        self.bridge = WebSocketBridge(self.config)

    @pytest.mark.asyncio
    @patch("opencode_lrs_bridge.websocket.manager.websockets.connect")
    async def test_connect_to_lrs_websocket(self, mock_connect):
        """Test connecting to LRS WebSocket."""
        # Setup mock
        mock_websocket = AsyncMock()
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_websocket)
        mock_websocket.__aiter__ = AsyncMock(return_value=iter([]))

        # Test connection
        await self.bridge.connect_to_lrs_websocket()

        # Check if connection was attempted
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    @patch("opencode_lrs_bridge.websocket.manager.websockets.connect")
    async def test_connect_to_opencode_websocket(self, mock_connect):
        """Test connecting to opencode WebSocket."""
        # Setup mock
        mock_websocket = AsyncMock()
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_websocket)
        mock_websocket.__aiter__ = AsyncMock(return_value=iter([]))

        # Test connection
        await self.bridge.connect_to_opencode_websocket()

        # Check if connection was attempted
        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_lrs_message(self):
        """Test handling LRS message."""
        agent_id = "test_agent_001"
        message_data = {
            "type": "agent_state_update",
            "agent_id": agent_id,
            "data": {"status": "active", "current_task": "Test task"},
        }

        # Simulate message from LRS
        await self.bridge._handle_lrs_message(message_data)

        # This would test that the message was broadcast to subscribers
        # Implementation depends on how broadcasting is mocked

    @pytest.mark.asyncio
    async def test_handle_opencode_message(self):
        """Test handling opencode message."""
        message_data = {
            "type": "agent_update",
            "agent_id": "test_agent_001",
            "data": {"status": "active", "task": "Test task"},
        }

        # Simulate message from opencode
        await self.bridge._handle_opencode_message(message_data)

        # This would test that the message was broadcast to subscribers
        # Implementation depends on how broadcasting is mocked

    @pytest.mark.asyncio
    async def test_broadcast_agent_update(self):
        """Test broadcasting agent update."""
        from opencode_lrs_bridge.models.schemas import AgentState, AgentStatus

        agent_state = AgentState(
            agent_id="test_agent_001",
            agent_type="hybrid",
            status=AgentStatus.ACTIVE,
            current_task="Test task",
        )

        # Mock connection manager
        self.bridge.connection_manager = Mock()
        self.bridge.connection_manager.broadcast_to_subscribers = AsyncMock()

        # Test broadcast
        await self.bridge.broadcast_agent_update(agent_state)

        # Check if broadcast was called
        self.bridge.connection_manager.broadcast_to_subscribers.assert_called_once()

        # Check message content
        call_args = self.bridge.connection_manager.broadcast_to_subscribers.call_args[0]
        broadcasted_agent_id = call_args[0]
        broadcasted_message = call_args[1]

        assert broadcasted_agent_id == "test_agent_001"
        assert broadcasted_message["type"] == "agent_state_update"
        assert broadcasted_message["data"]["agent_id"] == "test_agent_001"
        assert broadcasted_message["data"]["status"] == "active"
        assert broadcasted_message["source"] == "integration_bridge"

    @pytest.mark.asyncio
    async def test_broadcast_tool_execution(self):
        """Test broadcasting tool execution result."""
        execution_result = {
            "execution_id": "exec_001",
            "tool_name": "search",
            "agent_id": "test_agent_001",
            "status": "completed",
            "result": {"items": ["result1", "result2"]},
            "execution_time": 1.5,
        }

        # Mock connection manager
        self.bridge.connection_manager = Mock()
        self.bridge.connection_manager.broadcast_to_subscribers = AsyncMock()

        # Test broadcast
        await self.bridge.broadcast_tool_execution(execution_result)

        # Check if broadcast was called
        self.bridge.connection_manager.broadcast_to_subscribers.assert_called_once()

        # Check message content
        call_args = self.bridge.connection_manager.broadcast_to_subscribers.call_args[0]
        broadcasted_agent_id = call_args[0]
        broadcasted_message = call_args[1]

        assert broadcasted_agent_id == "test_agent_001"
        assert broadcasted_message["type"] == "tool_execution_update"
        assert broadcasted_message["data"]["tool_name"] == "search"
        assert broadcasted_message["source"] == "integration_bridge"
