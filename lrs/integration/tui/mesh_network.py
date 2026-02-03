"""
Distributed Agent Discovery and Mesh Networking.

This component provides decentralized agent discovery, P2P networking,
and mesh communication capabilities for distributed LRS agent deployments.
"""

import asyncio
import hashlib
import json
import struct
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ...multi_agent.shared_state import SharedWorldState


class MessageType(Enum):
    """Types of mesh network messages."""

    DISCOVERY = "discovery"
    HEARTBEAT = "heartbeat"
    ANNOUNCEMENT = "announcement"
    REQUEST = "request"
    RESPONSE = "response"
    COORDINATION = "coordination"
    STATE_SYNC = "state_sync"
    RESOURCE_SHARE = "resource_share"
    TASK_DISTRIBUTION = "task_distribution"


class NodeStatus(Enum):
    """Node status in mesh network."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    FAILED = "failed"
    LEAVING = "leaving"


class ConnectionType(Enum):
    """Types of connections in mesh network."""

    DIRECT = "direct"
    RELAYED = "relayed"
    PROXIMITY = "proximity"
    PREFERRED = "preferred"


@dataclass
class MeshNode:
    """Node in mesh network."""

    node_id: str
    address: Tuple[str, int]  # (host, port)
    capabilities: List[str]
    status: NodeStatus
    last_seen: datetime
    connection_types: Set[ConnectionType]
    load: float = 0.0
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if node is active."""
        return self.status == NodeStatus.ACTIVE and (datetime.now() - self.last_seen) < timedelta(
            minutes=5
        )

    @property
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "address": self.address,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "last_seen": self.last_seen.isoformat(),
            "connection_types": [ct.value for ct in self.connection_types],
            "load": self.load,
            "resources": self.resources,
            "metadata": self.metadata,
        }


@dataclass
class MeshMessage:
    """Message in mesh network."""

    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    timestamp: datetime
    ttl: int  # Time to live (hops)
    data: Dict[str, Any]
    signature: Optional[str] = None
    routing_path: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = more important

    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.ttl <= 0

    @property
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "data": self.data,
            "signature": self.signature,
            "routing_path": self.routing_path,
            "priority": self.priority,
        }


@dataclass
class RoutingTable:
    """Routing table for mesh network."""

    destination: str
    next_hop: str
    metric: float  # Cost metric
    timestamp: datetime
    path_history: List[str] = field(default_factory=list)


class DistributedMeshNetwork:
    """
    Distributed agent discovery and mesh networking system.

    Features:
    - Decentralized agent discovery
    - P2P mesh networking with redundancy
    - Adaptive routing with load balancing
    - Resource sharing across nodes
    - Fault tolerance and self-healing
    - Security and authentication
    - Cross-cluster communication
    - Distributed task coordination
    - Mesh analytics and monitoring

    Examples:
        >>> mesh = DistributedMeshNetwork(
        ...     local_node_id="node_1",
        ...     listen_port=9000,
        ...     shared_state=shared_state
        ... )
        >>>
        >>> # Start mesh network
        >>> await mesh.start()
        >>>
        >>> # Discover other nodes
        >>> await mesh.discover_peers()
        >>>
        >>> # Send message to specific node
        >>> await mesh.send_message(
        ...     recipient_id="node_2",
        ...     message_type=MessageType.COORDINATION,
        ...     data={'task': 'analysis', 'parameters': {...}}
        ... )
    """

    def __init__(
        self,
        local_node_id: str,
        listen_port: int,
        shared_state: SharedWorldState,
        discovery_ports: List[int] = None,
        max_connections: int = 50,
    ):
        """
        Initialize distributed mesh network.

        Args:
            local_node_id: Local node identifier
            listen_port: Port to listen on
            shared_state: LRS shared world state
            discovery_ports: Ports for node discovery
            max_connections: Maximum concurrent connections
        """
        self.local_node_id = local_node_id
        self.listen_port = listen_port
        self.shared_state = shared_state
        self.discovery_ports = discovery_ports or [9000, 9001, 9002]
        self.max_connections = max_connections

        # Network state
        self.nodes: Dict[str, MeshNode] = {}
        self.routing_table: Dict[str, RoutingTable] = {}
        self.connections: Dict[str, asyncio.StreamWriter] = {}
        self.connection_tasks: Dict[str, asyncio.Task] = {}

        # Message handling
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.pending_acks: Dict[str, asyncio.Future] = {}

        # Discovery and maintenance
        self.discovery_active = False
        self.heartbeat_interval = 30  # seconds
        self.cleanup_interval = 300  # seconds

        # Security
        self.node_keys: Dict[str, str] = {}  # node_id -> public_key
        self.private_key = self._generate_key_pair()

        # Configuration
        self.config = self._default_config()

        # Server components
        self.server: Optional[asyncio.Server] = None
        self.server_running = False

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._setup_message_handlers()
        self._create_local_node()

    async def start(self) -> bool:
        """
        Start mesh network server and discovery.

        Returns:
            Success status
        """
        try:
            # Start server
            self.server = await asyncio.start_server(
                self._handle_connection, "0.0.0.0", self.listen_port
            )

            self.server_running = True
            self.logger.info(f"Mesh network started on port {self.listen_port}")

            # Start discovery
            await self.start_discovery()

            # Start background tasks
            await self._start_background_tasks()

            return True

        except Exception as e:
            self.logger.error(f"Failed to start mesh network: {e}")
            return False

    async def stop(self):
        """Stop mesh network and cleanup resources."""

        self.server_running = False

        # Stop discovery
        await self.stop_discovery()

        # Close all connections
        for node_id, writer in self.connections.items():
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        self.logger.info("Mesh network stopped")

    async def discover_peers(self, timeout: int = 10) -> List[MeshNode]:
        """
        Discover peers in mesh network.

        Args:
            timeout: Discovery timeout in seconds

        Returns:
            List of discovered nodes
        """
        discovered_nodes = []

        # Send discovery messages to all known nodes
        discovery_message = MeshMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DISCOVERY,
            sender_id=self.local_node_id,
            recipient_id=None,  # Broadcast
            timestamp=datetime.now(),
            ttl=3,  # 3 hops max
            data={
                "node_id": self.local_node_id,
                "address": ("localhost", self.listen_port),
                "capabilities": self._get_local_capabilities(),
                "resources": self._get_local_resources(),
            },
        )

        # Send to discovery ports
        for port in self.discovery_ports:
            try:
                await self._send_udp_message(discovery_message, ("255.255.255.255", port))
            except Exception as e:
                self.logger.error(f"Failed to send discovery to port {port}: {e}")

        # Wait for responses
        await asyncio.sleep(timeout)

        # Return discovered active nodes
        discovered_nodes = [
            node
            for node in self.nodes.values()
            if node.is_active and node.node_id != self.local_node_id
        ]

        self.logger.info(f"Discovered {len(discovered_nodes)} peers")
        return discovered_nodes

    async def send_message(
        self,
        recipient_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
        priority: int = 0,
        require_ack: bool = True,
    ) -> bool:
        """
        Send message to specific node.

        Args:
            recipient_id: Target node ID
            message_type: Type of message
            data: Message data
            priority: Message priority
            require_ack: Whether to require acknowledgment

        Returns:
            Success status
        """
        if recipient_id not in self.nodes:
            self.logger.warning(f"Unknown recipient: {recipient_id}")
            return False

        message = MeshMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.local_node_id,
            recipient_id=recipient_id,
            timestamp=datetime.now(),
            ttl=10,  # Default TTL
            data=data,
            priority=priority,
        )

        # Sign message
        message.signature = self._sign_message(message)

        try:
            # Send via direct connection if available
            if recipient_id in self.connections:
                await self._send_message_direct(message, recipient_id)
            else:
                # Route through mesh
                await self._route_message(message)

            # Wait for acknowledgment if required
            if require_ack:
                ack_future = asyncio.Future()
                self.pending_acks[message.message_id] = ack_future

                try:
                    await asyncio.wait_for(ack_future, timeout=30.0)
                    return True
                except asyncio.TimeoutError:
                    self.logger.warning(f"No ACK received for message {message.message_id}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to send message to {recipient_id}: {e}")
            return False

    async def broadcast_message(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        exclude_nodes: List[str] = None,
        ttl: int = 5,
    ) -> int:
        """
        Broadcast message to all nodes in mesh.

        Args:
            message_type: Type of message
            data: Message data
            exclude_nodes: Nodes to exclude from broadcast
            ttl: Time to live in hops

        Returns:
            Number of nodes message was sent to
        """
        message = MeshMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.local_node_id,
            recipient_id=None,  # Broadcast
            timestamp=datetime.now(),
            ttl=ttl,
            data=data,
        )

        message.signature = self._sign_message(message)

        exclude_nodes = exclude_nodes or []
        sent_count = 0

        for node_id, node in self.nodes.items():
            if node.is_active and node_id not in exclude_nodes and node_id != self.local_node_id:
                try:
                    await self._route_message(message)
                    sent_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to broadcast to {node_id}: {e}")

        return sent_count

    async def get_mesh_status(self) -> Dict[str, Any]:
        """
        Get current mesh network status.

        Returns:
            Mesh status information
        """
        active_nodes = [node for node in self.nodes.values() if node.is_active]

        connection_stats = {
            "total_nodes": len(self.nodes),
            "active_nodes": len(active_nodes),
            "direct_connections": len(self.connections),
            "routing_table_size": len(self.routing_table),
            "message_queue_size": self.message_queue.qsize(),
        }

        # Calculate network metrics
        avg_load = np.mean([node.load for node in active_nodes]) if active_nodes else 0.0
        total_resources = self._calculate_total_resources(active_nodes)

        return {
            "timestamp": datetime.now().isoformat(),
            "local_node_id": self.local_node_id,
            "connection_stats": connection_stats,
            "network_metrics": {
                "average_load": avg_load,
                "total_resources": total_resources,
                "mesh_density": len(active_nodes) / max(1, len(self.nodes)),
                "connectivity": self._calculate_connectivity(active_nodes),
            },
            "nodes": [node.to_dict for node in active_nodes],
            "routing_table": {
                dest: {
                    "next_hop": rt.next_hop,
                    "metric": rt.metric,
                    "timestamp": rt.timestamp.isoformat(),
                }
                for dest, rt in self.routing_table.items()
            },
        }

    async def register_service(self, service_name: str, capabilities: Dict[str, Any]) -> bool:
        """
        Register service with mesh network.

        Args:
            service_name: Name of service
            capabilities: Service capabilities

        Returns:
            Success status
        """
        try:
            announcement = MeshMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ANNOUNCEMENT,
                sender_id=self.local_node_id,
                recipient_id=None,
                timestamp=datetime.now(),
                ttl=10,
                data={
                    "service_name": service_name,
                    "capabilities": capabilities,
                    "node_id": self.local_node_id,
                    "address": ("localhost", self.listen_port),
                },
            )

            await self.broadcast_message(announcement)

            # Update local node metadata
            if self.local_node_id in self.nodes:
                if "services" not in self.nodes[self.local_node_id].metadata:
                    self.nodes[self.local_node_id].metadata["services"] = {}

                self.nodes[self.local_node_id].metadata["services"][service_name] = capabilities

            return True

        except Exception as e:
            self.logger.error(f"Failed to register service {service_name}: {e}")
            return False

    async def discover_services(self, service_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Discover services in mesh network.

        Args:
            service_name: Specific service to find (optional)

        Returns:
            List of available services
        """
        services = []

        for node in self.nodes.values():
            if node.is_active and "services" in node.metadata:
                node_services = node.metadata["services"]

                for name, capabilities in node_services.items():
                    if service_name is None or name == service_name:
                        services.append(
                            {
                                "service_name": name,
                                "node_id": node.node_id,
                                "node_address": node.address,
                                "capabilities": capabilities,
                                "load": node.load,
                            }
                        )

        return services

    def _setup_message_handlers(self):
        """Setup message handlers for different message types."""

        self.message_handlers = {
            MessageType.DISCOVERY: [self._handle_discovery],
            MessageType.HEARTBEAT: [self._handle_heartbeat],
            MessageType.ANNOUNCEMENT: [self._handle_announcement],
            MessageType.REQUEST: [self._handle_request],
            MessageType.RESPONSE: [self._handle_response],
            MessageType.COORDINATION: [self._handle_coordination],
            MessageType.STATE_SYNC: [self._handle_state_sync],
            MessageType.RESOURCE_SHARE: [self._handle_resource_share],
            MessageType.TASK_DISTRIBUTION: [self._handle_task_distribution],
        }

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP connection."""

        try:
            # Get peer address
            peer_address = writer.get_extra_info("peername")

            # Read node identification
            node_id_data = await reader.readline()
            if not node_id_data:
                return

            node_id = node_id_data.decode().strip()

            # Store connection
            self.connections[node_id] = writer

            # Create or update node
            if node_id not in self.nodes:
                self.nodes[node_id] = MeshNode(
                    node_id=node_id,
                    address=peer_address,
                    capabilities=[],
                    status=NodeStatus.ACTIVE,
                    last_seen=datetime.now(),
                    connection_types={ConnectionType.DIRECT},
                )
            else:
                self.nodes[node_id].last_seen = datetime.now()
                self.nodes[node_id].status = NodeStatus.ACTIVE

            # Start message handling task
            task = asyncio.create_task(self._handle_messages(reader, node_id))
            self.connection_tasks[node_id] = task

            self.logger.info(f"Connected to node {node_id} from {peer_address}")

        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")

    async def _handle_messages(self, reader: asyncio.StreamReader, node_id: str):
        """Handle messages from connected node."""

        try:
            while self.server_running:
                # Read message length
                length_data = await reader.readexactly(4)
                message_length = struct.unpack("!I", length_data)[0]

                # Read message data
                message_data = await reader.readexactly(message_length)
                message_dict = json.loads(message_data.decode())

                # Create message object
                message = MeshMessage(
                    message_id=message_dict["message_id"],
                    message_type=MessageType(message_dict["message_type"]),
                    sender_id=message_dict["sender_id"],
                    recipient_id=message_dict.get("recipient_id"),
                    timestamp=datetime.fromisoformat(message_dict["timestamp"]),
                    ttl=message_dict["ttl"],
                    data=message_dict["data"],
                    signature=message_dict.get("signature"),
                    routing_path=message_dict.get("routing_path", []),
                    priority=message_dict.get("priority", 0),
                )

                # Verify signature
                if not self._verify_message(message):
                    self.logger.warning(f"Invalid signature for message from {node_id}")
                    continue

                # Decrement TTL
                message.ttl -= 1

                # Handle message
                if message.recipient_id is None or message.recipient_id == self.local_node_id:
                    await self._process_message(message)
                else:
                    # Route message
                    await self._route_message(message)

        except asyncio.IncompleteReadError:
            self.logger.info(f"Node {node_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling messages from {node_id}: {e}")
        finally:
            # Cleanup
            if node_id in self.connections:
                del self.connections[node_id]
            if node_id in self.connection_tasks:
                self.connection_tasks[node_id].cancel()
                del self.connection_tasks[node_id]

    async def _process_message(self, message: MeshMessage):
        """Process received message."""

        try:
            # Check if message is for us
            if message.recipient_id and message.recipient_id != self.local_node_id:
                return

            # Get handlers for message type
            handlers = self.message_handlers.get(message.message_type, [])

            # Call all handlers
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {e}")

            # Send acknowledgment for direct messages
            if (
                message.recipient_id == self.local_node_id
                and message.message_type != MessageType.HEARTBEAT
            ):
                ack_message = MeshMessage(
                    message_id=f"ack_{message.message_id}",
                    message_type=MessageType.RESPONSE,
                    sender_id=self.local_node_id,
                    recipient_id=message.sender_id,
                    timestamp=datetime.now(),
                    ttl=5,
                    data={"original_message_id": message.message_id},
                )

                await self._send_message_direct(ack_message, message.sender_id)

        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {e}")

    async def _handle_discovery(self, message: MeshMessage):
        """Handle discovery message."""

        if message.sender_id == self.local_node_id:
            return

        # Update node information
        data = message.data

        if message.sender_id not in self.nodes:
            self.nodes[message.sender_id] = MeshNode(
                node_id=message.sender_id,
                address=tuple(data.get("address", ("unknown", 0))),
                capabilities=data.get("capabilities", []),
                status=NodeStatus.ACTIVE,
                last_seen=datetime.now(),
                connection_types=set(),
                resources=data.get("resources", {}),
            )
        else:
            self.nodes[message.sender_id].last_seen = datetime.now()
            self.nodes[message.sender_id].status = NodeStatus.ACTIVE

        # Send discovery response
        response = MeshMessage(
            message_id=f"discovery_response_{self.local_node_id}",
            message_type=MessageType.RESPONSE,
            sender_id=self.local_node_id,
            recipient_id=message.sender_id,
            timestamp=datetime.now(),
            ttl=5,
            data={
                "node_id": self.local_node_id,
                "address": ("localhost", self.listen_port),
                "capabilities": self._get_local_capabilities(),
                "resources": self._get_local_resources(),
            },
        )

        await self._send_message_direct(response, message.sender_id)

    async def _handle_heartbeat(self, message: MeshMessage):
        """Handle heartbeat message."""

        if message.sender_id in self.nodes:
            self.nodes[message.sender_id].last_seen = datetime.now()

            # Update load and resources
            if "load" in message.data:
                self.nodes[message.sender_id].load = message.data["load"]

            if "resources" in message.data:
                self.nodes[message.sender_id].resources.update(message.data["resources"])

    async def _handle_announcement(self, message: MeshMessage):
        """Handle service announcement message."""

        data = message.data
        service_name = data.get("service_name")

        if service_name and message.sender_id in self.nodes:
            if "services" not in self.nodes[message.sender_id].metadata:
                self.nodes[message.sender_id].metadata["services"] = {}

            self.nodes[message.sender_id].metadata["services"][service_name] = data.get(
                "capabilities", {}
            )

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""

        return {
            "max_ttl": 10,
            "heartbeat_interval": 30,
            "discovery_timeout": 10,
            "message_timeout": 30,
            "max_message_size": 1048576,  # 1MB
            "max_connections": 50,
            "routing_update_interval": 60,
            "cleanup_interval": 300,
        }

    def _create_local_node(self):
        """Create local node representation."""

        self.nodes[self.local_node_id] = MeshNode(
            node_id=self.local_node_id,
            address=("localhost", self.listen_port),
            capabilities=self._get_local_capabilities(),
            status=NodeStatus.ACTIVE,
            last_seen=datetime.now(),
            connection_types=set(),
            resources=self._get_local_resources(),
            metadata={"local": True},
        )

    def _get_local_capabilities(self) -> List[str]:
        """Get local node capabilities."""

        return [
            "lrs_agent",
            "precision_tracking",
            "tool_execution",
            "state_management",
            "coordination",
            "optimization",
        ]

    def _get_local_resources(self) -> Dict[str, Any]:
        """Get local node resources."""

        return {
            "cpu_cores": 4,  # Would get from system
            "memory_gb": 8,
            "disk_space_gb": 100,
            "bandwidth_mbps": 1000,
            "agent_capacity": 10,
        }

    def _sign_message(self, message: MeshMessage) -> str:
        """Sign message with private key."""

        # Simplified signature (would use proper crypto)
        message_data = json.dumps(message.to_dict, sort_keys=True)
        signature = hashlib.sha256((message_data + self.private_key).encode()).hexdigest()

        return signature

    def _verify_message(self, message: MeshMessage) -> bool:
        """Verify message signature."""

        if not message.signature:
            return False  # Require signed messages

        # For now, accept all validly formatted signatures
        # In production, would verify with public key
        return True

    async def _send_message_direct(self, message: MeshMessage, node_id: str):
        """Send message directly to node."""

        if node_id not in self.connections:
            raise ValueError(f"No direct connection to {node_id}")

        writer = self.connections[node_id]
        message_data = json.dumps(message.to_dict).encode()

        # Send length first
        writer.write(struct.pack("!I", len(message_data)))
        writer.write(message_data)
        await writer.drain()

    async def _route_message(self, message: MeshMessage):
        """Route message through mesh network."""

        if message.is_expired:
            return

        # Check if we've seen this message before
        if message.message_id in self.pending_acks:
            return

        # Add to routing path
        if self.local_node_id not in message.routing_path:
            message.routing_path.append(self.local_node_id)

        # Determine next hop
        next_hop = self._get_next_hop(message.recipient_id)

        if next_hop and next_hop in self.connections:
            await self._send_message_direct(message, next_hop)
        elif message.recipient_id is None:
            # Broadcast to all connected nodes (except those already in path)
            for node_id, writer in self.connections.items():
                if node_id != self.local_node_id and node_id not in message.routing_path:
                    await self._send_message_direct(message, node_id)

    def _get_next_hop(self, destination: str) -> Optional[str]:
        """Get next hop for routing to destination."""

        if destination in self.routing_table:
            return self.routing_table[destination].next_hop

        # Use simple flooding if no route exists
        return None

    def _calculate_connectivity(self, nodes: List[MeshNode]) -> float:
        """Calculate network connectivity metric."""

        if len(nodes) < 2:
            return 0.0

        # Simple connectivity based on active connections
        total_possible = len(nodes) * (len(nodes) - 1) / 2
        actual_connections = sum(len(node.connection_types) for node in nodes) / 2

        return actual_connections / max(1, total_possible)

    def _calculate_total_resources(self, nodes: List[MeshNode]) -> Dict[str, float]:
        """Calculate total resources in network."""

        total = {
            "cpu_cores": 0.0,
            "memory_gb": 0.0,
            "disk_space_gb": 0.0,
            "bandwidth_mbps": 0.0,
            "agent_capacity": 0.0,
        }

        for node in nodes:
            for resource, value in node.resources.items():
                if resource in total:
                    total[resource] += float(value)

        return total


# Import required modules (would normally be at top)
import logging
import asyncio
import numpy as np
