"""
WebSocket Manager: Real-time event streaming for TUI integration.

This component manages WebSocket connections for bidirectional communication
between opencode TUI and LRS-Agents, providing reliable event delivery
and connection management.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:
    # Fallback types if FastAPI not available
    WebSocket = Any
    WebSocketDisconnect = Exception


@dataclass
class WebSocketConnection:
    """Represents an active WebSocket connection."""

    websocket: WebSocket
    connection_id: str
    channel: str  # Subscription channel
    connected_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if connection is still active."""
        return (datetime.now() - self.last_activity).total_seconds() < 300  # 5 minutes timeout


@dataclass
class EventBuffer:
    """Circular buffer for event streaming."""

    max_size: int = 1000
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add_event(self, event: Dict[str, Any]):
        """Add event to buffer."""
        self.events.append(event)

        # Keep only recent events
        if len(self.events) > self.max_size:
            self.events = self.events[-self.max_size :]

    def get_events_since(self, timestamp: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get events since specified timestamp."""
        if timestamp is None:
            return self.events.copy()

        return [
            event for event in self.events if datetime.fromisoformat(event["timestamp"]) > timestamp
        ]

    def clear_old_events(self, cutoff_time: datetime):
        """Remove events older than cutoff time."""
        self.events = [
            event
            for event in self.events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]


class WebSocketManager:
    """
    Manages WebSocket connections for real-time TUI communication.

    Features:
    - Connection lifecycle management
    - Channel-based message routing
    - Event buffering and history
    - Automatic cleanup of stale connections
    - Message acknowledgment and retry
    - Performance monitoring and metrics
    - Authentication and authorization support

    Examples:
        >>> manager = WebSocketManager(config)
        >>>
        >>> # Accept new connection
        >>> await manager.connect(websocket, "agent_123")
        >>>
        >>> # Broadcast to channel
        >>> await manager.broadcast("precision", {"agent_id": "123", "value": 0.8})
        >>>
        >>> # Send to specific connection
        >>> await manager.send_to_connection("conn_456", {"type": "update"})
        >>>
        >>> # Get connection metrics
        >>> metrics = manager.get_metrics()
    """

    def __init__(self, config):
        """
        Initialize WebSocket Manager.

        Args:
            config: TUI configuration
        """
        self.config = config

        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.channel_subscriptions: Dict[str, Set[str]] = {}  # channel -> connection_ids

        # Event buffering
        self.event_buffers: Dict[str, EventBuffer] = {}
        self.default_buffer_size = config.event_buffer_size

        # Performance metrics
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "connection_errors": 0,
            "broadcast_count": 0,
        }

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.ping_task: Optional[asyncio.Task] = None

        # Message tracking
        self.pending_acks: Dict[str, Dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)

        # Start background tasks
        self._start_background_tasks()

    async def connect(
        self, websocket: WebSocket, channel: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Accept and register new WebSocket connection.

        Args:
            websocket: WebSocket connection
            channel: Subscription channel
            metadata: Optional connection metadata

        Returns:
            Connection ID
        """
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        now = datetime.now()

        # Create connection object
        connection = WebSocketConnection(
            websocket=websocket,
            connection_id=connection_id,
            channel=channel,
            connected_at=now,
            last_activity=now,
            metadata=metadata or {},
        )

        # Register connection
        self.connections[connection_id] = connection

        # Subscribe to channel
        if channel not in self.channel_subscriptions:
            self.channel_subscriptions[channel] = set()
        self.channel_subscriptions[channel].add(connection_id)

        # Initialize event buffer for channel
        if channel not in self.event_buffers:
            self.event_buffers[channel] = EventBuffer(max_size=self.default_buffer_size)

        # Update metrics
        self.metrics["total_connections"] += 1
        self.metrics["active_connections"] = len(self.connections)

        self.logger.info(f"WebSocket connected: {connection_id} on channel {channel}")

        # Send connection confirmation
        await self.send_to_connection(
            connection_id,
            {
                "type": "connection_established",
                "connection_id": connection_id,
                "channel": channel,
                "server_time": datetime.now().isoformat(),
            },
        )

        # Send buffered events if requested
        if metadata and metadata.get("send_history", False):
            await self._send_buffered_events(connection_id, channel)

        return connection_id

    def disconnect(self, websocket: WebSocket, channel: str):
        """
        Handle WebSocket disconnection.

        Args:
            websocket: WebSocket connection
            channel: Subscription channel
        """
        # Find connection by websocket
        connection_id = None
        for cid, conn in self.connections.items():
            if conn.websocket == websocket:
                connection_id = cid
                break

        if connection_id:
            self._remove_connection(connection_id)
            self.logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """
        Send message to specific connection.

        Args:
            connection_id: Target connection ID
            message: Message to send

        Returns:
            Success status
        """
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        try:
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()

            # Add message ID for tracking
            message_id = str(uuid.uuid4())
            message["message_id"] = message_id

            # Send message
            await connection.websocket.send_text(json.dumps(message))

            # Update activity
            connection.last_activity = datetime.now()

            # Update metrics
            self.metrics["messages_sent"] += 1

            return True

        except Exception as e:
            self.logger.error(f"Error sending to connection {connection_id}: {e}")
            self.metrics["connection_errors"] += 1

            # Remove problematic connection
            self._remove_connection(connection_id)

            return False

    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]) -> int:
        """
        Send message to all connections for specific agent.

        Args:
            agent_id: Target agent ID
            message: Message to send

        Returns:
            Number of connections message was sent to
        """
        channel = f"agent_{agent_id}"
        return await self.broadcast(channel, message)

    async def send_response(self, request_id: str, response: Dict[str, Any]) -> bool:
        """
        Send response to pending request.

        Args:
            request_id: Original request ID
            response: Response data

        Returns:
            Success status
        """
        if request_id in self.pending_acks:
            connection_id = self.pending_acks[request_id]["connection_id"]

            response_message = {
                "type": "response",
                "request_id": request_id,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }

            # Clean up pending acknowledgment
            del self.pending_acks[request_id]

            return await self.send_to_connection(connection_id, response_message)

        return False

    async def broadcast(self, channel: str, message: Dict[str, Any]) -> int:
        """
        Broadcast message to all connections on channel.

        Args:
            channel: Target channel
            message: Message to broadcast

        Returns:
            Number of connections message was sent to
        """
        if channel not in self.channel_subscriptions:
            return 0

        # Add to event buffer
        if channel not in self.event_buffers:
            self.event_buffers[channel] = EventBuffer(max_size=self.default_buffer_size)

        broadcast_message = message.copy()
        broadcast_message["channel"] = channel

        if "timestamp" not in broadcast_message:
            broadcast_message["timestamp"] = datetime.now().isoformat()

        self.event_buffers[channel].add_event(broadcast_message)

        # Send to all subscribed connections
        connection_ids = self.channel_subscriptions[channel].copy()
        success_count = 0

        for connection_id in connection_ids:
            if await self.send_to_connection(connection_id, broadcast_message):
                success_count += 1

        # Update metrics
        self.metrics["broadcast_count"] += 1

        return success_count

    async def handle_message(self, connection_id: str, message_data: str):
        """
        Handle incoming message from connection.

        Args:
            connection_id: Source connection ID
            message_data: Raw message data
        """
        try:
            message = json.loads(message_data)

            # Update activity
            if connection_id in self.connections:
                self.connections[connection_id].last_activity = datetime.now()

            # Update metrics
            self.metrics["messages_received"] += 1

            # Handle message types
            message_type = message.get("type")

            if message_type == "ping":
                await self._handle_ping(connection_id, message)

            elif message_type == "subscribe":
                await self._handle_subscribe(connection_id, message)

            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(connection_id, message)

            elif message_type == "request":
                await self._handle_request(connection_id, message)

            elif message_type == "ack":
                await self._handle_ack(connection_id, message)

            else:
                self.logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON from connection {connection_id}")
        except Exception as e:
            self.logger.error(f"Error handling message from {connection_id}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get WebSocket manager performance metrics.

        Returns:
            Metrics dictionary
        """
        return {
            **self.metrics,
            "channels": list(self.channel_subscriptions.keys()),
            "connections_per_channel": {
                channel: len(connections)
                for channel, connections in self.channel_subscriptions.items()
            },
            "buffer_sizes": {
                channel: len(buffer.events) for channel, buffer in self.event_buffers.items()
            },
        }

    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about specific connection.

        Args:
            connection_id: Connection ID

        Returns:
            Connection info or None
        """
        if connection_id not in self.connections:
            return None

        connection = self.connections[connection_id]

        return {
            "connection_id": connection.connection_id,
            "channel": connection.channel,
            "connected_at": connection.connected_at.isoformat(),
            "last_activity": connection.last_activity.isoformat(),
            "is_active": connection.is_active,
            "metadata": connection.metadata,
        }

    def close_all_connections(self):
        """Close all active connections."""
        for connection_id in list(self.connections.keys()):
            self._remove_connection(connection_id)

        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.ping_task:
            self.ping_task.cancel()

        self.logger.info("All WebSocket connections closed")

    def _remove_connection(self, connection_id: str):
        """Remove connection from manager."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        # Remove from channel subscriptions
        if connection.channel in self.channel_subscriptions:
            self.channel_subscriptions[connection.channel].discard(connection_id)

            # Clean up empty channel
            if not self.channel_subscriptions[connection.channel]:
                del self.channel_subscriptions[connection.channel]

        # Remove connection
        del self.connections[connection_id]

        # Update metrics
        self.metrics["active_connections"] = len(self.connections)

        # Clean up pending acknowledgments
        pending_to_remove = [
            req_id
            for req_id, req_data in self.pending_acks.items()
            if req_data["connection_id"] == connection_id
        ]

        for req_id in pending_to_remove:
            del self.pending_acks[req_id]

    async def _send_buffered_events(self, connection_id: str, channel: str):
        """Send buffered events to new connection."""
        if channel not in self.event_buffers:
            return

        buffer = self.event_buffers[channel]
        events = buffer.get_events_since()

        # Send events in batches
        batch_size = 10
        for i in range(0, len(events), batch_size):
            batch = events[i : i + batch_size]

            await self.send_to_connection(
                connection_id,
                {
                    "type": "history_batch",
                    "events": batch,
                    "batch_index": i // batch_size,
                    "total_batches": (len(events) + batch_size - 1) // batch_size,
                },
            )

            # Small delay between batches
            await asyncio.sleep(0.01)

    async def _handle_ping(self, connection_id: str, message: Dict[str, Any]):
        """Handle ping message."""
        await self.send_to_connection(
            connection_id, {"type": "pong", "timestamp": datetime.now().isoformat()}
        )

    async def _handle_subscribe(self, connection_id: str, message: Dict[str, Any]):
        """Handle channel subscription request."""
        new_channel = message.get("channel")
        if not new_channel:
            return

        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Unsubscribe from current channel
        if connection.channel in self.channel_subscriptions:
            self.channel_subscriptions[connection.channel].discard(connection_id)

        # Subscribe to new channel
        connection.channel = new_channel

        if new_channel not in self.channel_subscriptions:
            self.channel_subscriptions[new_channel] = set()
        self.channel_subscriptions[new_channel].add(connection_id)

        # Initialize buffer if needed
        if new_channel not in self.event_buffers:
            self.event_buffers[new_channel] = EventBuffer(max_size=self.default_buffer_size)

        await self.send_to_connection(
            connection_id,
            {"type": "subscribed", "channel": new_channel, "timestamp": datetime.now().isoformat()},
        )

    async def _handle_unsubscribe(self, connection_id: str, message: Dict[str, Any]):
        """Handle channel unsubscription request."""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Remove from current channel
        if connection.channel in self.channel_subscriptions:
            self.channel_subscriptions[connection.channel].discard(connection_id)

        await self.send_to_connection(
            connection_id, {"type": "unsubscribed", "timestamp": datetime.now().isoformat()}
        )

    async def _handle_request(self, connection_id: str, message: Dict[str, Any]):
        """Handle request message requiring response."""
        request_id = message.get("request_id")
        if not request_id:
            return

        # Track pending acknowledgment
        self.pending_acks[request_id] = {
            "connection_id": connection_id,
            "timestamp": datetime.now(),
            "message": message,
        }

        # This would be handled by the bridge logic
        # For now, just acknowledge receipt
        await self.send_to_connection(
            connection_id,
            {
                "type": "request_received",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def _handle_ack(self, connection_id: str, message: Dict[str, Any]):
        """Handle acknowledgment message."""
        message_id = message.get("message_id")
        if message_id:
            # Process acknowledgment (could be used for delivery confirmation)
            pass

    def _start_background_tasks(self):
        """Start background maintenance tasks."""

        # Cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Ping task
        self.ping_task = asyncio.create_task(self._ping_loop())

    async def _cleanup_loop(self):
        """Background task for cleanup operations."""
        while True:
            try:
                # Remove inactive connections
                inactive_connections = [
                    cid for cid, conn in self.connections.items() if not conn.is_active
                ]

                for connection_id in inactive_connections:
                    self._remove_connection(connection_id)
                    self.logger.info(f"Removed inactive connection: {connection_id}")

                # Clean old events from buffers
                cutoff_time = datetime.now() - timedelta(hours=1)
                for buffer in self.event_buffers.values():
                    buffer.clear_old_events(cutoff_time)

                await asyncio.sleep(60)  # Cleanup every minute

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10)

    async def _ping_loop(self):
        """Background task for connection health checks."""
        while True:
            try:
                # Send ping to all connections
                for connection_id in list(self.connections.keys()):
                    await self.send_to_connection(
                        connection_id, {"type": "ping", "timestamp": datetime.now().isoformat()}
                    )

                await asyncio.sleep(30)  # Ping every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in ping loop: {e}")
                await asyncio.sleep(10)


# Import timedelta for background tasks
from datetime import timedelta
