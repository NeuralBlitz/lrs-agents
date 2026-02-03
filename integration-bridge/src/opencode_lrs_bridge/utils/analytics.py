"""
Analytics and monitoring integration for the integration bridge.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics
from dataclasses import dataclass, asdict
import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

from ..config.settings import IntegrationBridgeConfig

logger = structlog.get_logger(__name__)

# Custom Prometheus metrics registry
REGISTRY = CollectorRegistry()

# Define custom metrics
BRIDGE_REQUESTS_TOTAL = Counter(
    'bridge_requests_total',
    'Total number of requests processed by the bridge',
    ['method', 'endpoint', 'status', 'source_system'],
    registry=REGISTRY
)

BRIDGE_REQUEST_DURATION = Histogram(
    'bridge_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint', 'source_system'],
    registry=REGISTRY
)

AGENT_OPERATIONS_TOTAL = Counter(
    'agent_operations_total',
    'Total number of agent operations',
    ['operation', 'agent_type', 'status'],
    registry=REGISTRY
)

TOOL_EXECUTIONS_TOTAL = Counter(
    'tool_executions_total',
    'Total number of tool executions',
    ['tool_name', 'system', 'status'],
    registry=REGISTRY

TOOL_EXECUTION_DURATION = Histogram(
    'tool_execution_duration_seconds',
    'Tool execution duration in seconds',
    ['tool_name', 'system'],
    registry=REGISTRY
)

WEBSOCKET_CONNECTIONS = Gauge(
    'websocket_connections_current',
    'Current number of WebSocket connections',
    registry=REGISTRY
)

STATE_SYNCHRONIZATIONS_TOTAL = Counter(
    'state_synchronizations_total',
    'Total number of state synchronizations',
    ['strategy', 'conflict_count', 'status'],
    registry=REGISTRY
)

SECURITY_EVENTS_TOTAL = Counter(
    'security_events_total',
    'Total number of security events',
    ['event_type', 'severity'],
    registry=REGISTRY
)

SYSTEM_METRICS = Gauge(
    'bridge_system_metrics',
    'System metrics for the bridge',
    ['metric_name'],
    registry=REGISTRY
)


@dataclass
class AnalyticsEvent:
    """Analytics event data structure."""
    
    timestamp: datetime
    event_type: str
    source_system: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    duration: Optional[float] = None
    status: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EventAggregator:
    """Aggregates analytics events for metrics calculation."""
    
    def __init__(self, window_size_minutes: int = 60):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.events: deque = deque(maxlen=10000)
        self.aggregated_metrics: Dict[str, Any] = {}
        self.last_aggregation = datetime.utcnow()
    
    def add_event(self, event: AnalyticsEvent):
        """Add an event to the aggregator."""
        self.events.append(event)
        
        # Trigger aggregation if needed
        if datetime.utcnow() - self.last_aggregation > timedelta(minutes=5):
            asyncio.create_task(self._aggregate_events())
    
    async def _aggregate_events(self):
        """Aggregate events and update metrics."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - self.window_size
        
        # Filter events within window
        recent_events = [
            event for event in self.events
            if event.timestamp > cutoff_time
        ]
        
        if not recent_events:
            return
        
        # Calculate metrics
        self.aggregated_metrics = {
            "total_events": len(recent_events),
            "events_per_minute": len(recent_events) / 60,
            "unique_agents": len(set(e.agent_id for e in recent_events if e.agent_id)),
            "unique_users": len(set(e.user_id for e in recent_events if e.user_id)),
            "error_rate": self._calculate_error_rate(recent_events),
            "avg_duration": self._calculate_avg_duration(recent_events),
            "popular_tools": self._get_popular_tools(recent_events),
            "active_agents": self._get_active_agents(recent_events),
            "security_events": self._get_security_events(recent_events),
            "last_updated": current_time.isoformat()
        }
        
        # Update Prometheus gauges
        SYSTEM_METRICS.labels(metric_name='events_per_minute').set(self.aggregated_metrics['events_per_minute'])
        SYSTEM_METRICS.labels(metric_name='unique_agents').set(self.aggregated_metrics['unique_agents'])
        SYSTEM_METRICS.labels(metric_name='unique_users').set(self.aggregated_metrics['unique_users'])
        SYSTEM_METRICS.labels(metric_name='error_rate').set(self.aggregated_metrics['error_rate'])
        
        self.last_aggregation = current_time
        logger.info("Analytics aggregated", **self.aggregated_metrics)
    
    def _calculate_error_rate(self, events: List[AnalyticsEvent]) -> float:
        """Calculate error rate from events."""
        error_events = [e for e in events if e.status in ['error', 'failed', 'timeout']]
        return len(error_events) / len(events) if events else 0.0
    
    def _calculate_avg_duration(self, events: List[AnalyticsEvent]) -> float:
        """Calculate average duration from events."""
        durations = [e.duration for e in events if e.duration is not None]
        return statistics.mean(durations) if durations else 0.0
    
    def _get_popular_tools(self, events: List[AnalyticsEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular tools from events."""
        tool_counts = defaultdict(int)
        for event in events:
            if event.metadata and 'tool_name' in event.metadata:
                tool_counts[event.metadata['tool_name']] += 1
        
        return [
            {"tool_name": tool, "count": count}
            for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _get_active_agents(self, events: List[AnalyticsEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most active agents from events."""
        agent_counts = defaultdict(int)
        for event in events:
            if event.agent_id:
                agent_counts[event.agent_id] += 1
        
        return [
            {"agent_id": agent_id, "activity_count": count}
            for agent_id, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _get_security_events(self, events: List[AnalyticsEvent]) -> Dict[str, int]:
        """Get security events breakdown."""
        security_events = defaultdict(int)
        for event in events:
            if event.event_type.startswith('security_'):
                security_events[event.event_type] += 1
        
        return dict(security_events)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current aggregated metrics."""
        return self.aggregated_metrics.copy()


class AnomalyDetector:
    """Detects anomalies in system behavior."""
    
    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.baselines: Dict[str, float] = {}
        self.alert_thresholds: Dict[str, float] = {
            "error_rate": 0.1,  # 10% error rate
            "response_time": 5.0,  # 5 seconds
            "connection_count": 1000,  # 1000 connections
        }
        self.anomaly_history: deque = deque(maxlen=100)
    
    async def check_for_anomalies(self, metrics: Dict[str, Any]):
        """Check metrics for anomalies and trigger alerts."""
        anomalies = []
        
        # Check error rate
        error_rate = metrics.get('error_rate', 0.0)
        if error_rate > self.alert_thresholds['error_rate']:
            anomalies.append({
                "type": "high_error_rate",
                "value": error_rate,
                "threshold": self.alert_thresholds['error_rate'],
                "severity": "high" if error_rate > 0.2 else "medium"
            })
        
        # Check response time
        avg_duration = metrics.get('avg_duration', 0.0)
        if avg_duration > self.alert_thresholds['response_time']:
            anomalies.append({
                "type": "slow_response",
                "value": avg_duration,
                "threshold": self.alert_thresholds['response_time'],
                "severity": "medium"
            })
        
        # Check connection count
        connection_count = metrics.get('websocket_connections', 0)
        if connection_count > self.alert_thresholds['connection_count']:
            anomalies.append({
                "type": "high_connection_count",
                "value": connection_count,
                "threshold": self.alert_thresholds['connection_count'],
                "severity": "medium"
            })
        
        # Store anomalies
        for anomaly in anomalies:
            anomaly['timestamp'] = datetime.utcnow().isoformat()
            self.anomaly_history.append(anomaly)
            
            # Log anomaly
            logger.warning(
                "Anomaly detected",
                anomaly_type=anomaly['type'],
                value=anomaly['value'],
                threshold=anomaly['threshold'],
                severity=anomaly['severity']
            )
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent anomalies within specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            anomaly for anomaly in self.anomaly_history
            if datetime.fromisoformat(anomaly['timestamp']) > cutoff_time
        ]


class AnalyticsManager:
    """Main analytics manager for the integration bridge."""
    
    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.event_aggregator = EventAggregator()
        self.anomaly_detector = AnomalyDetector(config)
        self.custom_metrics: Dict[str, Any] = {}
        self.is_collecting = False
        
    async def track_event(
        self,
        event_type: str,
        source_system: str,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        duration: Optional[float] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track an analytics event."""
        if not self.config.monitoring.enable_logging:
            return
        
        event = AnalyticsEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            source_system=source_system,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            duration=duration,
            status=status,
            metadata=metadata or {}
        )
        
        self.event_aggregator.add_event(event)
        
        # Update Prometheus metrics
        await self._update_prometheus_metrics(event)
    
    async def _update_prometheus_metrics(self, event: AnalyticsEvent):
        """Update Prometheus metrics based on event."""
        try:
            if event.event_type == 'api_request':
                BRIDGE_REQUESTS_TOTAL.labels(
                    method=event.metadata.get('method', 'unknown'),
                    endpoint=event.metadata.get('endpoint', 'unknown'),
                    status=event.status or 'unknown',
                    source_system=event.source_system
                ).inc()
                
                if event.duration:
                    BRIDGE_REQUEST_DURATION.labels(
                        method=event.metadata.get('method', 'unknown'),
                        endpoint=event.metadata.get('endpoint', 'unknown'),
                        source_system=event.source_system
                    ).observe(event.duration)
            
            elif event.event_type == 'agent_operation':
                AGENT_OPERATIONS_TOTAL.labels(
                    operation=event.metadata.get('operation', 'unknown'),
                    agent_type=event.metadata.get('agent_type', 'unknown'),
                    status=event.status or 'unknown'
                ).inc()
            
            elif event.event_type == 'tool_execution':
                TOOL_EXECUTIONS_TOTAL.labels(
                    tool_name=event.metadata.get('tool_name', 'unknown'),
                    system=event.source_system,
                    status=event.status or 'unknown'
                ).inc()
                
                if event.duration:
                    TOOL_EXECUTION_DURATION.labels(
                        tool_name=event.metadata.get('tool_name', 'unknown'),
                        system=event.source_system
                    ).observe(event.duration)
            
            elif event.event_type == 'security_event':
                SECURITY_EVENTS_TOTAL.labels(
                    event_type=event.metadata.get('event_type', 'unknown'),
                    severity=event.metadata.get('severity', 'unknown')
                ).inc()
        
        except Exception as e:
            logger.error("Failed to update Prometheus metrics", error=str(e))
    
    async def start_metrics_collection(self):
        """Start background metrics collection."""
        if not self.config.monitoring.enable_metrics:
            return
        
        self.is_collecting = True
        
        # Start periodic aggregation and anomaly detection
        asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("Analytics metrics collection started")
    
    async def _metrics_collection_loop(self):
        """Background loop for metrics collection and anomaly detection."""
        while self.is_collecting:
            try:
                # Get current metrics
                current_metrics = self.event_aggregator.get_metrics()
                
                # Check for anomalies
                await self.anomaly_detector.check_for_anomalies(current_metrics)
                
                # Wait before next collection
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(60)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if not self.config.monitoring.enable_metrics:
            return ""
        
        try:
            return generate_latest(REGISTRY).decode('utf-8')
        except Exception as e:
            logger.error("Failed to generate Prometheus metrics", error=str(e))
            return ""
    
    def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for analytics."""
        return {
            "current_metrics": self.event_aggregator.get_metrics(),
            "recent_anomalies": self.anomaly_detector.get_recent_anomalies(24),
            "anomaly_summary": self._summarize_anomalies(),
            "system_health": self._calculate_system_health(),
            "trend_analysis": self._analyze_trends()
        }
    
    def _summarize_anomalies(self) -> Dict[str, Any]:
        """Summarize recent anomalies."""
        recent_anomalies = self.anomaly_detector.get_recent_anomalies(24)
        
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for anomaly in recent_anomalies:
            severity_counts[anomaly['severity']] += 1
            type_counts[anomaly['type']] += 1
        
        return {
            "total_anomalies": len(recent_anomalies),
            "severity_breakdown": dict(severity_counts),
            "type_breakdown": dict(type_counts),
            "high_severity_count": severity_counts.get('high', 0)
        }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        current_metrics = self.event_aggregator.get_metrics()
        recent_anomalies = self.anomaly_detector.get_recent_anomalies(24)
        
        health_score = 100  # Start with perfect health
        
        # Deduct for high error rate
        error_rate = current_metrics.get('error_rate', 0.0)
        if error_rate > 0.05:  # 5%
            health_score -= min(30, error_rate * 100)
        
        # Deduct for anomalies
        high_severity_anomalies = len([
            a for a in recent_anomalies 
            if a.get('severity') == 'high'
        ])
        health_score -= min(40, high_severity_anomalies * 10)
        
        # Deduct for slow response times
        avg_duration = current_metrics.get('avg_duration', 0.0)
        if avg_duration > 2.0:  # 2 seconds
            health_score -= min(20, avg_duration * 5)
        
        health_score = max(0, health_score)  # Ensure non-negative
        
        return {
            "overall_score": health_score,
            "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "unhealthy",
            "factors": {
                "error_rate_impact": min(30, error_rate * 100),
                "anomaly_impact": min(40, high_severity_anomalies * 10),
                "performance_impact": min(20, avg_duration * 5)
            }
        }
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in the data."""
        # This would implement more sophisticated trend analysis
        # For now, return placeholder data
        
        current_metrics = self.event_aggregator.get_metrics()
        
        return {
            "request_volume_trend": "stable",  # Could be: increasing, decreasing, stable
            "error_rate_trend": "decreasing",
            "performance_trend": "improving",
            "recommendations": [
                "Monitor agent 'agent_001' for potential optimization",
                "Consider scaling up during peak hours"
            ]
        }
    
    def record_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a custom metric."""
        key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
        self.custom_metrics[key] = {
            "name": name,
            "value": value,
            "labels": labels or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def stop_metrics_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        logger.info("Analytics metrics collection stopped")