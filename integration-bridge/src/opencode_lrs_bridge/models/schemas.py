"""
Data models for opencode ↔ LRS-Agents integration.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from ..config.settings import OpenCodeConfig


class AgentType(str, Enum):
    """Agent type enumeration."""

    LRS = "lrs"
    COLLABORATIVE = "collaborative"
    OPENCODE = "opencode"
    HYBRID = "hybrid"


class AgentStatus(str, Enum):
    """Agent status enumeration."""

    IDLE = "idle"
    ACTIVE = "active"
    THINKING = "thinking"
    EXECUTING = "executing"
    ERROR = "error"
    COMPLETED = "completed"


class ToolExecutionStatus(str, Enum):
    """Tool execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class PrecisionData(BaseModel):
    """Precision tracking data."""

    tool_name: str
    alpha: float
    beta: float
    precision: float
    confidence: float
    prediction_error: Optional[float] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """Agent state information."""

    agent_id: str
    agent_type: AgentType
    status: AgentStatus
    current_task: Optional[str] = None
    precision_data: List[PrecisionData] = []
    belief_state: Dict[str, Any] = {}
    tool_history: List[Dict[str, Any]] = []
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ToolExecutionRequest(BaseModel):
    """Tool execution request."""

    tool_name: str
    parameters: Dict[str, Any] = {}
    timeout: float = 30.0
    agent_id: Optional[str] = None
    context: Dict[str, Any] = {}


class ToolExecutionResult(BaseModel):
    """Tool execution result."""

    execution_id: str
    tool_name: str
    status: ToolExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    prediction_error: Optional[float] = None
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentCreateRequest(BaseModel):
    """Agent creation request."""

    agent_id: str
    agent_type: AgentType
    config: Dict[str, Any] = Field(default_factory=lambda: {"opencode": {}})
    tools: List[str] = []
    preferences: Dict[str, Any] = {}
    opencode_session_id: Optional[str] = None
    request: Dict[str, Any] = Field(default_factory=dict)


class AgentUpdateRequest(BaseModel):
    """Agent update request."""

    status: Optional[AgentStatus] = None
    config: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    belief_state: Optional[Dict[str, Any]] = None


class WebSocketMessage(BaseModel):
    """WebSocket message format."""

    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None


class SystemInfo(BaseModel):
    """System information."""

    version: str
    environment: str
    uptime: float
    active_agents: int
    total_executions: int
    system_load: Dict[str, float]
    memory_usage: Dict[str, Any]


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}


class EventData(BaseModel):
    """Event data for streaming."""

    event_type: str
    agent_id: Optional[str] = None
    data: Dict[str, Any]
    severity: str = "info"
    category: str = "general"


class MetricData(BaseModel):
    """Metrics data for monitoring."""

    metric_name: str
    value: float
    labels: Dict[str, str] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConflictResolutionStrategy(str, Enum):
    """State conflict resolution strategies."""

    TUI_WINS = "tui_wins"
    LRS_WINS = "lrs_wins"
    TIMESTAMP = "timestamp"
    MERGE = "merge"


class StateSyncRequest(BaseModel):
    """State synchronization request."""

    agent_id: str
    state: Dict[str, Any]
    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.TIMESTAMP
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StateSyncResponse(BaseModel):
    """State synchronization response."""

    success: bool
    resolved_state: Dict[str, Any]
    conflicts: List[str] = []
    strategy_used: ConflictResolutionStrategy
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IntegrationBridgeMetrics(BaseModel):
    """Integration bridge specific metrics."""

    total_api_requests: int
    active_websocket_connections: int
    agents_managed: int
    tools_executed: int
    avg_response_time: float
    error_rate: float
    uptime: float
    last_updated: datetime = Field(default_factory=datetime.utcnow)
