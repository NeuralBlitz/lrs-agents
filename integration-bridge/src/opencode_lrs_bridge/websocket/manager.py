"""
WebSocket communication for real-time updates between opencode and LRS-Agents.
"""

import asyncio
import json
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
import websockets
from fastapi import WebSocket, WebSocketDisconnect, Depends
import structlog

from ..config.settings import IntegrationBridgeConfig
from ..models.schemas import WebSocketMessage, EventData, AgentState
from ..auth.middleware import AuthenticationMiddleware

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message routing."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, Set[str]] = {}
        self.agent_subscribers: Dict[str, Set[str]] = {}
        self.event_history: deque = deque(maxlen=1000)
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.websocket.message_queue_size
        )

    async def connect(
        self, websocket: WebSocket, connection_id: str, user_info: Dict[str, Any]
    ) -> bool:
        """Accept WebSocket connection."""
        try:
            await websocket.accept()

            # Store connection
            self.active_connections[connection_id] = websocket
            self.connection_subscriptions[connection_id] = set()

            # Start heartbeat for this connection
            self.heartbeat_tasks[connection_id] = asyncio.create_task(
                self._heartbeat_loop(connection_id)
            )

            # Send welcome message
            await self.send_message(
                connection_id,
                {
                    "type": "connection_established",
                    "data": {
                        "connection_id": connection_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                },
            )

            logger.info("WebSocket connection established", connection_id=connection_id)
            return True

        except Exception as e:
            logger.error("Failed to establish WebSocket connection", error=str(e))
            return False

    async def disconnect(self, connection_id: str):
        """Disconnect WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        if connection_id in self.connection_subscriptions:
            del self.connection_subscriptions[connection_id]

        if connection_id in self.heartbeat_tasks:
            self.heartbeat_tasks[connection_id].cancel()
            del self.heartbeat_tasks[connection_id]

        # Remove from agent subscriptions
        for agent_id, subscribers in self.agent_subscribers.items():
            subscribers.discard(connection_id)

        logger.info("WebSocket connection closed", connection_id=connection_id)

    async def subscribe_to_agent(self, connection_id: str, agent_id: str):
        """Subscribe connection to agent updates."""
        if connection_id not in self.connection_subscriptions:
            return False

        self.connection_subscriptions[connection_id].add(agent_id)

        if agent_id not in self.agent_subscribers:
            self.agent_subscribers[agent_id] = set()
        self.agent_subscribers[agent_id].add(connection_id)

        # Send current agent state if available
        # This would be populated by the agent manager

        logger.info(
            "Subscribed to agent updates",
            connection_id=connection_id,
            agent_id=agent_id,
        )
        return True

    async def unsubscribe_from_agent(self, connection_id: str, agent_id: str):
        """Unsubscribe connection from agent updates."""
        if connection_id in self.connection_subscriptions:
            self.connection_subscriptions[connection_id].discard(agent_id)

        if agent_id in self.agent_subscribers:
            self.agent_subscribers[agent_id].discard(connection_id)

        logger.info(
            "Unsubscribed from agent updates",
            connection_id=connection_id,
            agent_id=agent_id,
        )

    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.active_connections:
            return False

        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(
                "Failed to send WebSocket message",
                connection_id=connection_id,
                error=str(e),
            )
            await self.disconnect(connection_id)
            return False

    async def broadcast_to_subscribers(self, agent_id: str, message: Dict[str, Any]):
        """Broadcast message to all subscribers of an agent."""
        if agent_id not in self.agent_subscribers:
            return

        subscribers = self.agent_subscribers[agent_id].copy()

        # Send to all subscribers
        tasks = []
        for connection_id in subscribers:
            if connection_id in self.active_connections:
                task = asyncio.create_task(self.send_message(connection_id, message))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        connection_ids = list(self.active_connections.keys())

        tasks = []
        for connection_id in connection_ids:
            task = asyncio.create_task(self.send_message(connection_id, message))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _heartbeat_loop(self, connection_id: str):
        """Send periodic heartbeat messages."""
        while connection_id in self.active_connections:
            try:
                await asyncio.sleep(self.config.websocket.heartbeat_interval)

                if connection_id in self.active_connections:
                    heartbeat_message = {
                        "type": "heartbeat",
                        "data": {
                            "timestamp": datetime.utcnow().isoformat(),
                            "connection_id": connection_id,
                        },
                    }

                    if not await self.send_message(connection_id, heartbeat_message):
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Heartbeat error", connection_id=connection_id, error=str(e)
                )
                break


class EventProcessor:
    """Processes and routes WebSocket events."""

    def __init__(
        self, config: IntegrationBridgeConfig, connection_manager: ConnectionManager
    ):
        self.config = config
        self.connection_manager = connection_manager
        self.event_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default event handlers."""
        self.event_handlers.update(
            {
                "subscribe_agent": self._handle_subscribe_agent,
                "unsubscribe_agent": self._handle_unsubscribe_agent,
                "get_agent_state": self._handle_get_agent_state,
                "execute_tool": self._handle_execute_tool,
                "ping": self._handle_ping,
            }
        )

    async def process_message(self, connection_id: str, message: Dict[str, Any]):
        """Process incoming WebSocket message."""
        try:
            message_type = message.get("type")
            if not message_type:
                await self.connection_manager.send_message(
                    connection_id,
                    {"type": "error", "data": {"error": "Message type required"}},
                )
                return

            handler = self.event_handlers.get(message_type)
            if not handler:
                await self.connection_manager.send_message(
                    connection_id,
                    {
                        "type": "error",
                        "data": {"error": f"Unknown message type: {message_type}"},
                    },
                )
                return

            await handler(connection_id, message.get("data", {}))

        except Exception as e:
            logger.error(
                "Error processing WebSocket message",
                connection_id=connection_id,
                error=str(e),
            )
            await self.connection_manager.send_message(
                connection_id,
                {"type": "error", "data": {"error": "Internal server error"}},
            )

    async def _handle_subscribe_agent(self, connection_id: str, data: Dict[str, Any]):
        """Handle agent subscription request."""
        agent_id = data.get("agent_id")
        if not agent_id:
            await self.connection_manager.send_message(
                connection_id, {"type": "error", "data": {"error": "agent_id required"}}
            )
            return

        success = await self.connection_manager.subscribe_to_agent(
            connection_id, agent_id
        )

        await self.connection_manager.send_message(
            connection_id,
            {
                "type": "subscription_response",
                "data": {"agent_id": agent_id, "subscribed": success},
            },
        )

    async def _handle_unsubscribe_agent(self, connection_id: str, data: Dict[str, Any]):
        """Handle agent unsubscription request."""
        agent_id = data.get("agent_id")
        if not agent_id:
            await self.connection_manager.send_message(
                connection_id, {"type": "error", "data": {"error": "agent_id required"}}
            )
            return

        await self.connection_manager.unsubscribe_from_agent(connection_id, agent_id)

        await self.connection_manager.send_message(
            connection_id,
            {"type": "unsubscription_response", "data": {"agent_id": agent_id}},
        )

    async def _handle_get_agent_state(self, connection_id: str, data: Dict[str, Any]):
        """Handle get agent state request."""
        agent_id = data.get("agent_id")
        if not agent_id:
            await self.connection_manager.send_message(
                connection_id, {"type": "error", "data": {"error": "agent_id required"}}
            )
            return

        # This would query the agent manager for current state
        await self.connection_manager.send_message(
            connection_id,
            {
                "type": "agent_state_response",
                "data": {
                    "agent_id": agent_id,
                    "state": None,  # Would be populated by agent manager
                },
            },
        )

    async def _handle_execute_tool(self, connection_id: str, data: Dict[str, Any]):
        """Handle tool execution request."""
        # This would delegate to the tool executor
        await self.connection_manager.send_message(
            connection_id,
            {
                "type": "tool_execution_response",
                "data": {"execution_id": "temp_id", "status": "pending"},
            },
        )

    async def _handle_ping(self, connection_id: str, data: Dict[str, Any]):
        """Handle ping request."""
        await self.connection_manager.send_message(
            connection_id,
            {"type": "pong", "data": {"timestamp": datetime.utcnow().isoformat()}},
        )


class WebSocketBridge:
    """Main WebSocket bridge for real-time communication."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.connection_manager = ConnectionManager(config)
        self.event_processor = EventProcessor(config, self.connection_manager)
        self.auth_middleware = AuthenticationMiddleware(config)

        # Connect to external WebSocket servers
        self.lrs_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.opencode_connections: Dict[str, websockets.WebSocketServerProtocol] = {}

    async def handle_websocket_connection(
        self, websocket: WebSocket, connection_id: str, token: Optional[str] = None
    ):
        """Handle incoming WebSocket connection."""
        try:
            # Authenticate connection
            if token:
                user_info = await self._authenticate_websocket(token)
            else:
                # For now, allow unauthenticated connections in development
                user_info = {"user_id": "anonymous", "permissions": ["read_state"]}

            # Establish connection
            success = await self.connection_manager.connect(
                websocket, connection_id, user_info
            )
            if not success:
                return

            # Listen for messages
            while True:
                try:
                    message = await websocket.receive_text()
                    message_data = json.loads(message)
                    await self.event_processor.process_message(
                        connection_id, message_data
                    )

                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await self.connection_manager.send_message(
                        connection_id,
                        {"type": "error", "data": {"error": "Invalid JSON format"}},
                    )
                except Exception as e:
                    logger.error("WebSocket message handling error", error=str(e))

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error("WebSocket connection error", error=str(e))
        finally:
            await self.connection_manager.disconnect(connection_id)

    async def _authenticate_websocket(self, token: str) -> Dict[str, Any]:
        """Authenticate WebSocket connection using JWT token."""
        try:
            # This would use the token manager to validate JWT
            return {
                "user_id": "authenticated_user",
                "permissions": ["read_state", "write_state"],
            }
        except Exception:
            raise Exception("Authentication failed")

    async def connect_to_lrs_websocket(self):
        """Connect to LRS-Agents WebSocket endpoints."""
        try:
            # Connect to agent state updates
            uri = f"ws://{self.config.lrs.base_url.replace('http://', '')}/ws/agents/state"
            async with websockets.connect(uri) as websocket:
                self.lrs_connections["agent_state"] = websocket

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_lrs_message(data)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON from LRS WebSocket")

        except Exception as e:
            logger.error("Failed to connect to LRS WebSocket", error=str(e))

    async def connect_to_opencode_websocket(self):
        """Connect to opencode WebSocket endpoints."""
        try:
            # Connect to opencode updates
            uri = f"ws://{self.config.opencode.base_url.replace('http://', '')}/ws/updates"
            async with websockets.connect(uri) as websocket:
                self.opencode_connections["updates"] = websocket

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_opencode_message(data)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON from opencode WebSocket")

        except Exception as e:
            logger.error("Failed to connect to opencode WebSocket", error=str(e))

    async def _handle_lrs_message(self, data: Dict[str, Any]):
        """Handle message from LRS-Agents."""
        message_type = data.get("type")

        if message_type == "agent_state_update":
            agent_id = data.get("agent_id")
            if agent_id:
                await self.connection_manager.broadcast_to_subscribers(
                    agent_id,
                    {
                        "type": "agent_state_update",
                        "data": data,
                        "source": "lrs_agents",
                    },
                )

        elif message_type == "precision_update":
            agent_id = data.get("agent_id")
            if agent_id:
                await self.connection_manager.broadcast_to_subscribers(
                    agent_id,
                    {"type": "precision_update", "data": data, "source": "lrs_agents"},
                )

        elif message_type == "tool_execution":
            agent_id = data.get("agent_id")
            if agent_id:
                await self.connection_manager.broadcast_to_subscribers(
                    agent_id,
                    {
                        "type": "tool_execution_update",
                        "data": data,
                        "source": "lrs_agents",
                    },
                )

    async def _handle_opencode_message(self, data: Dict[str, Any]):
        """Handle message from opencode."""
        message_type = data.get("type")

        if message_type == "agent_update":
            agent_id = data.get("agent_id")
            if agent_id:
                await self.connection_manager.broadcast_to_subscribers(
                    agent_id,
                    {"type": "agent_state_update", "data": data, "source": "opencode"},
                )

        elif message_type == "system_event":
            # Broadcast system events to all connections
            await self.connection_manager.broadcast_to_all(
                {"type": "system_event", "data": data, "source": "opencode"}
            )

    async def start_background_connections(self):
        """Start background WebSocket connections to external services."""
        # Start LRS connection
        asyncio.create_task(self.connect_to_lrs_websocket())

        # Start opencode connection
        asyncio.create_task(self.connect_to_opencode_websocket())

    async def broadcast_agent_update(self, agent_state: AgentState):
        """Broadcast agent state update to subscribers."""
        message = {
            "type": "agent_state_update",
            "data": {
                "agent_id": agent_state.agent_id,
                "status": agent_state.status.value,
                "current_task": agent_state.current_task,
                "precision_data": [p.dict() for p in agent_state.precision_data],
                "last_activity": agent_state.last_activity.isoformat(),
            },
            "source": "integration_bridge",
        }

        await self.connection_manager.broadcast_to_subscribers(
            agent_state.agent_id, message
        )

    async def broadcast_tool_execution(self, execution_result: Dict[str, Any]):
        """Broadcast tool execution result."""
        agent_id = execution_result.get("agent_id")
        if agent_id:
            message = {
                "type": "tool_execution_update",
                "data": execution_result,
                "source": "integration_bridge",
            }

            await self.connection_manager.broadcast_to_subscribers(agent_id, message)
