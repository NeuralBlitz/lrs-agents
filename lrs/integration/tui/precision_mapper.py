"""
TUI Precision Mapper: Maps LRS precision data to TUI-friendly visualization formats.

This component transforms mathematical precision representations into
intuitive confidence indicators, alerts, and visual cues for the TUI.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class ConfidenceLevel(Enum):
    """Confidence levels for TUI visualization."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ConfidenceIndicator:
    """Visual confidence indicator for TUI."""

    level: ConfidenceLevel
    score: float  # 0-100
    color: str  # Hex color code
    icon: str  # Icon name
    description: str
    trend: Optional[str] = None  # "improving", "declining", "stable"


@dataclass
class TUIAlert:
    """TUI alert for precision changes."""

    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    agent_id: str
    precision_before: float
    precision_after: float
    timestamp: datetime
    suggested_actions: List[str]
    metadata: Dict[str, Any]


class TUIPrecisionMapper:
    """
    Maps LRS precision to TUI confidence indicators and alerts.

    Features:
    - Precision to confidence level mapping
    - Visual indicators with colors and icons
    - Trend analysis and prediction
    - Alert generation for significant changes
    - Actionable recommendations
    - Historical precision tracking

    Examples:
        >>> mapper = TUIPrecisionMapper()
        >>>
        >>> # Map precision to confidence
        >>> confidence = mapper.precision_to_confidence({'value': 0.85})
        >>> print(confidence.level)  # ConfidenceLevel.HIGH
        >>>
        >>> # Generate adaptation alert
        >>> alert = mapper.adaptation_to_tui_alert({
        ...     'precision_before': 0.7,
        ...     'precision_after': 0.3
        ... })
        >>> print(alert.severity)  # AlertSeverity.ERROR
        >>>
        >>> # Get visual indicators
        >>> indicators = mapper.get_confidence_indicators([
        ...     {'value': 0.9}, {'value': 0.6}, {'value': 0.2}
        ... ])
    """

    def __init__(self):
        """Initialize TUI Precision Mapper."""

        # Confidence level thresholds
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_HIGH: (0.9, 1.0),
            ConfidenceLevel.HIGH: (0.75, 0.9),
            ConfidenceLevel.MEDIUM: (0.5, 0.75),
            ConfidenceLevel.LOW: (0.25, 0.5),
            ConfidenceLevel.VERY_LOW: (0.0, 0.25),
        }

        # Visual mapping
        self.visual_mapping = {
            ConfidenceLevel.VERY_HIGH: {
                "color": "#10b981",  # Green
                "icon": "confidence-very-high",
                "description": "Very High Confidence - Agent is highly reliable",
            },
            ConfidenceLevel.HIGH: {
                "color": "#22c55e",  # Light Green
                "icon": "confidence-high",
                "description": "High Confidence - Agent is reliable",
            },
            ConfidenceLevel.MEDIUM: {
                "color": "#f59e0b",  # Orange
                "icon": "confidence-medium",
                "description": "Medium Confidence - Agent is moderately reliable",
            },
            ConfidenceLevel.LOW: {
                "color": "#f97316",  # Dark Orange
                "icon": "confidence-low",
                "description": "Low Confidence - Agent reliability is questionable",
            },
            ConfidenceLevel.VERY_LOW: {
                "color": "#ef4444",  # Red
                "icon": "confidence-very-low",
                "description": "Very Low Confidence - Agent is unreliable",
            },
        }

        # Alert configuration
        self.alert_thresholds = {
            "precision_drop": 0.3,  # Alert if precision drops by this amount
            "low_precision": 0.4,  # Alert if precision below this level
            "critical_precision": 0.2,  # Critical alert if precision below this level
        }

        # Historical precision tracking
        self.precision_history: Dict[str, List[Tuple[datetime, float]]] = {}

    def precision_to_confidence(self, precision_data: Dict[str, Any]) -> ConfidenceIndicator:
        """
        Map precision data to confidence indicator.

        Args:
            precision_data: Precision data from LRS

        Returns:
            Confidence indicator for TUI visualization
        """
        precision_value = precision_data.get("value", 0.5)

        # Determine confidence level
        confidence_level = self._get_confidence_level(precision_value)

        # Get visual mapping
        visual_info = self.visual_mapping[confidence_level]

        # Calculate score (0-100)
        confidence_score = precision_value * 100

        # Determine trend if historical data available
        trend = self._calculate_trend(precision_data.get("agent_id"), precision_value)

        return ConfidenceIndicator(
            level=confidence_level,
            score=round(confidence_score, 1),
            color=visual_info["color"],
            icon=visual_info["icon"],
            description=visual_info["description"],
            trend=trend,
        )

    def adaptation_to_tui_alert(self, adaptation_event: Dict[str, Any]) -> TUIAlert:
        """
        Convert adaptation event to TUI alert.

        Args:
            adaptation_event: Adaptation event data

        Returns:
            TUI alert for user notification
        """
        agent_id = adaptation_event.get("agent_id", "unknown")
        precision_before = adaptation_event.get("precision_before", 0.5)
        precision_after = adaptation_event.get("precision_after", 0.5)

        # Calculate precision change
        precision_change = precision_after - precision_before

        # Determine alert severity
        if precision_after < self.alert_thresholds["critical_precision"]:
            severity = AlertSeverity.CRITICAL
        elif precision_after < self.alert_thresholds["low_precision"]:
            severity = AlertSeverity.ERROR
        elif abs(precision_change) > self.alert_thresholds["precision_drop"]:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        # Generate alert content
        alert_type = "precision_drop" if precision_change < 0 else "precision_change"

        if precision_change < 0:
            title = f"Precision Drop for Agent {agent_id}"
            message = (
                f"Agent confidence decreased from {precision_before:.2f} to {precision_after:.2f}"
            )
        else:
            title = f"Precision Improvement for Agent {agent_id}"
            message = (
                f"Agent confidence increased from {precision_before:.2f} to {precision_after:.2f}"
            )

        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(
            precision_before, precision_after, severity
        )

        return TUIAlert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            agent_id=agent_id,
            precision_before=precision_before,
            precision_after=precision_after,
            timestamp=datetime.now(),
            suggested_actions=suggested_actions,
            metadata={
                "precision_change": precision_change,
                "adaptation_trigger": adaptation_event.get("trigger_tool"),
                "confidence_before": self.precision_to_confidence({"value": precision_before}),
                "confidence_after": self.precision_to_confidence({"value": precision_after}),
            },
        )

    def get_confidence_indicators(
        self, precision_values: List[float], agent_id: Optional[str] = None
    ) -> List[ConfidenceIndicator]:
        """
        Get confidence indicators for multiple precision values.

        Args:
            precision_values: List of precision values
            agent_id: Optional agent ID for historical tracking

        Returns:
            List of confidence indicators
        """
        indicators = []

        for precision in precision_values:
            indicator = self.precision_to_confidence({"value": precision})

            # Update historical tracking
            if agent_id:
                self._update_precision_history(agent_id, precision)

                # Recalculate trend with updated history
                indicator.trend = self._calculate_trend(agent_id, precision)

            indicators.append(indicator)

        return indicators

    def get_precision_summary(
        self, agent_id: str, precision_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get comprehensive precision summary for TUI.

        Args:
            agent_id: Agent identifier
            precision_data: Current precision data

        Returns:
            Precision summary for TUI display
        """
        confidence = self.precision_to_confidence(precision_data)

        # Get historical statistics
        history_stats = self._get_historical_stats(agent_id)

        # Predict future precision
        prediction = self._predict_precision_trend(agent_id)

        return {
            "agent_id": agent_id,
            "current_precision": precision_data.get("value", 0.5),
            "confidence": {
                "level": confidence.level.value,
                "score": confidence.score,
                "color": confidence.color,
                "icon": confidence.icon,
                "description": confidence.description,
                "trend": confidence.trend,
            },
            "historical": history_stats,
            "prediction": prediction,
            "alerts": self._check_for_alerts(agent_id, precision_data),
            "last_updated": datetime.now().isoformat(),
        }

    def get_dashboard_metrics(self, all_precisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get dashboard-level precision metrics.

        Args:
            all_precisions: All agent precision data

        Returns:
            Dashboard metrics
        """
        if not all_precisions:
            return {
                "total_agents": 0,
                "average_precision": 0,
                "confidence_distribution": {},
                "critical_agents": [],
                "system_health": "unknown",
            }

        # Calculate aggregate metrics
        precision_values = [data.get("value", 0.5) for data in all_precisions.values()]

        average_precision = sum(precision_values) / len(precision_values)

        # Confidence distribution
        confidence_distribution = {level.value: 0 for level in ConfidenceLevel}

        critical_agents = []

        for agent_id, data in all_precisions.items():
            confidence = self.precision_to_confidence(data)
            confidence_distribution[confidence.level.value] += 1

            if confidence.level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]:
                critical_agents.append(
                    {
                        "agent_id": agent_id,
                        "precision": data.get("value", 0.5),
                        "confidence": confidence.level.value,
                        "color": confidence.color,
                    }
                )

        # System health assessment
        if average_precision > 0.8:
            system_health = "excellent"
        elif average_precision > 0.6:
            system_health = "good"
        elif average_precision > 0.4:
            system_health = "fair"
        else:
            system_health = "poor"

        return {
            "total_agents": len(all_precisions),
            "average_precision": round(average_precision, 3),
            "system_health": system_health,
            "confidence_distribution": confidence_distribution,
            "critical_agents": critical_agents,
            "health_color": self._get_health_color(system_health),
            "last_updated": datetime.now().isoformat(),
        }

    def _get_confidence_level(self, precision_value: float) -> ConfidenceLevel:
        """Determine confidence level from precision value."""

        for level, (min_val, max_val) in self.confidence_thresholds.items():
            if min_val <= precision_value < max_val:
                return level

        return ConfidenceLevel.VERY_LOW  # Fallback

    def _calculate_trend(self, agent_id: Optional[str], current_precision: float) -> Optional[str]:
        """Calculate precision trend based on historical data."""

        if not agent_id or agent_id not in self.precision_history:
            return None

        history = self.precision_history[agent_id]

        # Need at least 3 data points for trend calculation
        if len(history) < 3:
            return None

        # Get last 3 points (excluding current)
        recent_points = history[-3:]

        # Simple linear regression to determine trend
        x_values = list(range(len(recent_points)))
        y_values = [point[1] for point in recent_points]

        n = len(recent_points)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        # Calculate slope
        if n * sum_x2 - sum_x**2 != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        else:
            slope = 0

        # Determine trend
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"

    def _update_precision_history(self, agent_id: str, precision_value: float):
        """Update precision history for agent."""

        if agent_id not in self.precision_history:
            self.precision_history[agent_id] = []

        # Add new data point
        self.precision_history[agent_id].append((datetime.now(), precision_value))

        # Keep only last 100 data points
        if len(self.precision_history[agent_id]) > 100:
            self.precision_history[agent_id] = self.precision_history[agent_id][-100:]

    def _get_historical_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get historical precision statistics."""

        if agent_id not in self.precision_history:
            return {"data_points": 0, "average": 0, "min": 0, "max": 0, "variance": 0}

        history = self.precision_history[agent_id]
        precision_values = [point[1] for point in history]

        if not precision_values:
            return {"data_points": 0, "average": 0, "min": 0, "max": 0, "variance": 0}

        average = sum(precision_values) / len(precision_values)
        min_val = min(precision_values)
        max_val = max(precision_values)

        # Calculate variance
        variance = sum((x - average) ** 2 for x in precision_values) / len(precision_values)

        return {
            "data_points": len(precision_values),
            "average": round(average, 3),
            "min": round(min_val, 3),
            "max": round(max_val, 3),
            "variance": round(variance, 3),
            "standard_deviation": round(math.sqrt(variance), 3),
        }

    def _predict_precision_trend(self, agent_id: str) -> Dict[str, Any]:
        """Predict precision trend for agent."""

        if agent_id not in self.precision_history:
            return {
                "trend": "unknown",
                "confidence": 0,
                "predicted_value": None,
                "time_horizon": "unknown",
            }

        history = self.precision_history[agent_id]

        if len(history) < 5:
            return {
                "trend": "insufficient_data",
                "confidence": 0,
                "predicted_value": None,
                "time_horizon": "unknown",
            }

        # Simple trend prediction based on last few points
        recent_values = [point[1] for point in history[-5:]]

        # Calculate trend
        if len(recent_values) >= 2:
            recent_trend = recent_values[-1] - recent_values[-2]
        else:
            recent_trend = 0

        # Predict next value
        predicted_value = recent_values[-1] + recent_trend

        # Determine confidence in prediction
        variance = sum(
            (x - sum(recent_values) / len(recent_values)) ** 2 for x in recent_values
        ) / len(recent_values)
        prediction_confidence = max(0, 1 - variance * 10)  # Higher variance = lower confidence

        # Determine trend direction
        if recent_trend > 0.05:
            trend = "improving"
        elif recent_trend < -0.05:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "confidence": round(prediction_confidence, 2),
            "predicted_value": round(predicted_value, 3),
            "time_horizon": "short_term",  # Based on recent data only
        }

    def _check_for_alerts(
        self, agent_id: str, precision_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for any alert conditions."""

        alerts = []
        precision_value = precision_data.get("value", 0.5)

        # Critical precision alert
        if precision_value < self.alert_thresholds["critical_precision"]:
            alerts.append(
                {
                    "type": "critical_precision",
                    "severity": "critical",
                    "message": f"Agent {agent_id} has critically low precision: {precision_value:.2f}",
                    "suggested_actions": [
                        "immediate_intervention",
                        "tool_reset",
                        "fallback_activation",
                    ],
                }
            )

        # Low precision alert
        elif precision_value < self.alert_thresholds["low_precision"]:
            alerts.append(
                {
                    "type": "low_precision",
                    "severity": "warning",
                    "message": f"Agent {agent_id} has low precision: {precision_value:.2f}",
                    "suggested_actions": ["monitor_closely", "consider_adaptation"],
                }
            )

        return alerts

    def _generate_suggested_actions(
        self, precision_before: float, precision_after: float, severity: AlertSeverity
    ) -> List[str]:
        """Generate suggested actions based on precision change."""

        actions = []

        if severity == AlertSeverity.CRITICAL:
            actions.extend(
                [
                    "Immediate intervention required",
                    "Reset agent precision parameters",
                    "Activate fallback mechanisms",
                    "Review tool selection",
                ]
            )
        elif severity == AlertSeverity.ERROR:
            actions.extend(
                ["Monitor agent closely", "Consider tool alternatives", "Adjust learning rates"]
            )
        elif severity == AlertSeverity.WARNING:
            actions.extend(
                [
                    "Continue monitoring",
                    "Review recent tool executions",
                    "Check for environmental changes",
                ]
            )
        else:
            actions.extend(["Continue normal operation", "Document improvement pattern"])

        return actions

    def _get_health_color(self, system_health: str) -> str:
        """Get color for system health indicator."""

        health_colors = {
            "excellent": "#10b981",  # Green
            "good": "#22c55e",  # Light Green
            "fair": "#f59e0b",  # Orange
            "poor": "#ef4444",  # Red
            "unknown": "#6b7280",  # Gray
        }

        return health_colors.get(system_health, "#6b7280")
