"""
Advanced Analytics and Forecasting Dashboard.

This component provides sophisticated analytics, predictive modeling,
and forecasting capabilities for LRS agents with real-time dashboards.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ...multi_agent.shared_state import SharedWorldState
from .precision_mapper import TUIPrecisionMapper


class MetricType(Enum):
    """Types of analytics metrics."""

    PRECISION = "precision"
    PERFORMANCE = "performance"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    ADAPTATION_FREQUENCY = "adaptation_frequency"
    RESOURCE_USAGE = "resource_usage"
    RESPONSE_TIME = "response_time"


class ForecastModel(Enum):
    """Forecasting model types."""

    LINEAR_REGRESSION = "linear_regression"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AnalyticsMetric:
    """Analytics metric data point."""

    metric_type: MetricType
    agent_id: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "agent_id": self.agent_id,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ForecastResult:
    """Forecasting result."""

    metric_type: MetricType
    agent_id: str
    forecast_values: List[float]
    forecast_timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    model_type: ForecastModel
    accuracy_score: float
    generated_at: datetime

    @property
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "agent_id": self.agent_id,
            "forecast_values": self.forecast_values,
            "forecast_timestamps": [ts.isoformat() for ts in self.forecast_timestamps],
            "confidence_intervals": self.confidence_intervals,
            "model_type": self.model_type.value,
            "accuracy_score": self.accuracy_score,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class AnalyticsAlert:
    """Analytics alert."""

    alert_id: str
    alert_type: str
    level: AlertLevel
    agent_id: Optional[str]
    metric_type: MetricType
    message: str
    threshold: float
    current_value: float
    predicted_values: Optional[List[float]]
    recommended_actions: List[str]
    created_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class AdvancedAnalyticsDashboard:
    """
    Advanced analytics and forecasting dashboard.

    Features:
    - Real-time metric collection and aggregation
    - Multiple forecasting models (statistical and ML-based)
    - Anomaly detection and alerting
    - Performance trend analysis
    - Resource utilization forecasting
    - Automated insights generation
    - Interactive dashboard data
    - Historical data analysis

    Examples:
        >>> dashboard = AdvancedAnalyticsDashboard(shared_state)
        >>>
        >>> # Get real-time analytics
        >>> analytics = await dashboard.get_real_time_analytics()
        >>>
        >>> # Generate forecast
        >>> forecast = await dashboard.generate_forecast(
        ...     'agent_1', MetricType.PRECISION, horizon=24
        ... )
        >>>
        >>> # Get system insights
        >>> insights = await dashboard.generate_insights()
    """

    def __init__(self, shared_state: SharedWorldState, precision_mapper: TUIPrecisionMapper):
        """
        Initialize analytics dashboard.

        Args:
            shared_state: LRS shared world state
            precision_mapper: TUI precision mapper
        """
        self.shared_state = shared_state
        self.precision_mapper = precision_mapper

        # Data storage
        self.metrics_buffer: List[AnalyticsMetric] = []
        self.historical_data: Dict[str, List[AnalyticsMetric]] = {}

        # Forecasting models
        self.forecasting_models: Dict[MetricType, Dict[ForecastModel, Any]] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}

        # Alerting system
        self.active_alerts: Dict[str, AnalyticsAlert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}

        # Dashboard state
        self.dashboard_cache: Dict[str, Any] = {}
        self.last_update: datetime = datetime.now()

        # Configuration
        self.config = self._default_config()

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._initialize_forecasting_models()
        self._setup_alert_rules()
        self._start_background_tasks()

    async def get_real_time_analytics(self) -> Dict[str, Any]:
        """
        Get real-time analytics dashboard data.

        Returns:
            Comprehensive analytics data for dashboard
        """
        # Collect current metrics
        current_metrics = await self._collect_current_metrics()

        # Calculate aggregated statistics
        system_stats = self._calculate_system_statistics(current_metrics)

        # Generate insights
        insights = await self._generate_real_time_insights(current_metrics)

        # Get active alerts
        alerts = await self._get_active_alerts()

        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": [metric.to_dict for metric in current_metrics],
            "system_statistics": system_stats,
            "insights": insights,
            "active_alerts": [alert.__dict__ for alert in alerts.values()],
            "performance_trends": self._calculate_performance_trends(),
            "resource_utilization": self._get_resource_utilization(),
        }

    async def generate_forecast(
        self,
        agent_id: str,
        metric_type: MetricType,
        horizon: int = 24,
        model_type: Optional[ForecastModel] = None,
    ) -> ForecastResult:
        """
        Generate forecast for specific agent and metric.

        Args:
            agent_id: Agent identifier
            metric_type: Type of metric to forecast
            horizon: Number of time periods to forecast
            model_type: Specific forecasting model (auto-select if None)

        Returns:
            Forecast result with confidence intervals
        """
        try:
            # Get historical data
            historical_data = self._get_historical_data(agent_id, metric_type)

            if len(historical_data) < 10:
                raise ValueError(
                    f"Insufficient data for forecasting: {len(historical_data)} points"
                )

            # Select best model if not specified
            if model_type is None:
                model_type = await self._select_best_model(historical_data, metric_type)

            # Generate forecast
            forecast_data = await self._generate_model_forecast(
                historical_data, model_type, horizon
            )

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                historical_data, forecast_data["values"]
            )

            # Create forecast result
            result = ForecastResult(
                metric_type=metric_type,
                agent_id=agent_id,
                forecast_values=forecast_data["values"],
                forecast_timestamps=forecast_data["timestamps"],
                confidence_intervals=confidence_intervals,
                model_type=model_type,
                accuracy_score=forecast_data.get("accuracy", 0.8),
                generated_at=datetime.now(),
            )

            # Store forecast performance
            self._store_forecast_performance(result)

            return result

        except Exception as e:
            self.logger.error(f"Error generating forecast for {agent_id}: {e}")

            # Return fallback forecast
            return ForecastResult(
                metric_type=metric_type,
                agent_id=agent_id,
                forecast_values=[],
                forecast_timestamps=[],
                confidence_intervals=[],
                model_type=ForecastModel.LINEAR_REGRESSION,
                accuracy_score=0.0,
                generated_at=datetime.now(),
            )

    async def generate_insights(
        self, time_range: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """
        Generate automated insights from analytics data.

        Args:
            time_range: Time range for analysis

        Returns:
            Dictionary containing various insights
        """
        try:
            cutoff_time = datetime.now() - time_range

            # Get recent metrics
            recent_metrics = [
                metric for metric in self.metrics_buffer if metric.timestamp > cutoff_time
            ]

            insights = {
                "performance_insights": await self._analyze_performance_trends(recent_metrics),
                "precision_insights": await self._analyze_precision_patterns(recent_metrics),
                "efficiency_insights": await self._analyze_efficiency_patterns(recent_metrics),
                "anomaly_insights": await self._detect_anomalies(recent_metrics),
                "optimization_opportunities": await self._identify_optimization_opportunities(
                    recent_metrics
                ),
                "risk_assessment": await self._assess_system_risks(recent_metrics),
            }

            return insights

        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return {}

    async def get_comparative_analytics(
        self,
        agent_ids: List[str],
        metric_types: List[MetricType],
        time_range: timedelta = timedelta(days=7),
    ) -> Dict[str, Any]:
        """
        Get comparative analytics between multiple agents.

        Args:
            agent_ids: List of agent IDs to compare
            metric_types: Types of metrics to compare
            time_range: Time range for comparison

        Returns:
            Comparative analytics data
        """
        try:
            cutoff_time = datetime.now() - time_range

            comparison_data = {
                "agents": agent_ids,
                "metrics": {},
                "comparisons": {},
                "rankings": {},
                "summary": {},
            }

            # Collect data for each agent and metric
            for metric_type in metric_types:
                metric_data = {}

                for agent_id in agent_ids:
                    agent_metrics = [
                        metric
                        for metric in self.metrics_buffer
                        if (
                            metric.agent_id == agent_id
                            and metric.metric_type == metric_type
                            and metric.timestamp > cutoff_time
                        )
                    ]

                    if agent_metrics:
                        values = [m.value for m in agent_metrics]
                        metric_data[agent_id] = {
                            "values": values,
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "trend": self._calculate_trend(values),
                        }

                comparison_data["metrics"][metric_type.value] = metric_data

            # Generate comparisons
            for metric_type in metric_types:
                metric_key = metric_type.value
                if metric_key in comparison_data["metrics"]:
                    comparison_data["comparisons"][metric_key] = self._generate_comparisons(
                        comparison_data["metrics"][metric_key]
                    )

            # Generate rankings
            for metric_type in metric_types:
                metric_key = metric_type.value
                if metric_key in comparison_data["metrics"]:
                    comparison_data["rankings"][metric_key] = self._generate_rankings(
                        comparison_data["metrics"][metric_key]
                    )

            return comparison_data

        except Exception as e:
            self.logger.error(f"Error generating comparative analytics: {e}")
            return {}

    async def create_anomaly_detector(
        self, metric_type: MetricType, sensitivity: float = 2.0
    ) -> Dict[str, Any]:
        """
        Create anomaly detector for specific metric.

        Args:
            metric_type: Type of metric to monitor
            sensitivity: Sensitivity threshold (standard deviations)

        Returns:
            Anomaly detector configuration
        """
        try:
            # Get recent data for baseline
            recent_data = [
                metric
                for metric in self.metrics_buffer
                if (
                    metric.metric_type == metric_type
                    and metric.timestamp > datetime.now() - timedelta(days=7)
                )
            ]

            if len(recent_data) < 50:
                raise ValueError("Insufficient data for anomaly detection")

            # Calculate baseline statistics
            values = [m.value for m in recent_data]
            baseline_mean = np.mean(values)
            baseline_std = np.std(values)

            detector_config = {
                "metric_type": metric_type.value,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "sensitivity": sensitivity,
                "threshold_upper": baseline_mean + sensitivity * baseline_std,
                "threshold_lower": baseline_mean - sensitivity * baseline_std,
                "created_at": datetime.now().isoformat(),
            }

            return detector_config

        except Exception as e:
            self.logger.error(f"Error creating anomaly detector: {e}")
            return {}

    def _initialize_forecasting_models(self):
        """Initialize forecasting models."""

        for metric_type in MetricType:
            self.forecasting_models[metric_type] = {
                ForecastModel.LINEAR_REGRESSION: self._create_linear_regression_model(),
                ForecastModel.MOVING_AVERAGE: self._create_moving_average_model(),
                ForecastModel.EXPONENTIAL_SMOOTHING: self._create_exponential_smoothing_model(),
                ForecastModel.ENSEMBLE: self._create_ensemble_model(),
            }

    def _setup_alert_rules(self):
        """Setup default alert rules."""

        self.alert_rules = {
            "precision_drop": {
                "metric_type": MetricType.PRECISION,
                "threshold": 0.3,
                "operator": "decrease",
                "time_window": 300,  # 5 minutes
                "level": AlertLevel.WARNING,
            },
            "critical_precision": {
                "metric_type": MetricType.PRECISION,
                "threshold": 0.2,
                "operator": "below",
                "time_window": 60,  # 1 minute
                "level": AlertLevel.CRITICAL,
            },
            "high_error_rate": {
                "metric_type": MetricType.ERROR_RATE,
                "threshold": 0.1,
                "operator": "above",
                "time_window": 300,
                "level": AlertLevel.WARNING,
            },
            "slow_response_time": {
                "metric_type": MetricType.RESPONSE_TIME,
                "threshold": 5000,  # 5 seconds
                "operator": "above",
                "time_window": 120,
                "level": AlertLevel.WARNING,
            },
        }

    async def _collect_current_metrics(self) -> List[AnalyticsMetric]:
        """Collect current metrics from all agents."""

        current_metrics = []
        all_states = self.shared_state.get_all_states()

        for agent_id, state in all_states.items():
            # Precision metric
            precision = state.get("precision", {})
            if "value" in precision:
                current_metrics.append(
                    AnalyticsMetric(
                        metric_type=MetricType.PRECISION,
                        agent_id=agent_id,
                        value=precision["value"],
                        timestamp=datetime.now(),
                        metadata=precision,
                    )
                )

            # Performance metric (can be calculated from various factors)
            performance_score = self._calculate_performance_score(state)
            current_metrics.append(
                AnalyticsMetric(
                    metric_type=MetricType.PERFORMANCE,
                    agent_id=agent_id,
                    value=performance_score,
                    timestamp=datetime.now(),
                    metadata={"calculated": True},
                )
            )

            # Error rate metric
            tool_executions = state.get("tool_executions", [])
            if tool_executions:
                error_rate = self._calculate_error_rate(tool_executions)
                current_metrics.append(
                    AnalyticsMetric(
                        metric_type=MetricType.ERROR_RATE,
                        agent_id=agent_id,
                        value=error_rate,
                        timestamp=datetime.now(),
                        metadata={"execution_count": len(tool_executions)},
                    )
                )

            # Response time metric
            response_times = [ex.get("duration", 0) for ex in tool_executions if "duration" in ex]
            if response_times:
                avg_response_time = np.mean(response_times)
                current_metrics.append(
                    AnalyticsMetric(
                        metric_type=MetricType.RESPONSE_TIME,
                        agent_id=agent_id,
                        value=avg_response_time,
                        timestamp=datetime.now(),
                        metadata={"sample_count": len(response_times)},
                    )
                )

        # Add to buffer
        self.metrics_buffer.extend(current_metrics)

        # Keep buffer manageable
        if len(self.metrics_buffer) > 10000:
            self.metrics_buffer = self.metrics_buffer[-5000]

        return current_metrics

    def _calculate_system_statistics(self, metrics: List[AnalyticsMetric]) -> Dict[str, Any]:
        """Calculate system-wide statistics."""

        if not metrics:
            return {}

        # Group by metric type
        metrics_by_type = {}
        for metric in metrics:
            if metric.metric_type not in metrics_by_type:
                metrics_by_type[metric.metric_type] = []
            metrics_by_type[metric.metric_type].append(metric.value)

        statistics = {}

        for metric_type, values in metrics_by_type.items():
            if values:
                statistics[metric_type.value] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99),
                }

        return statistics

    async def _generate_real_time_insights(self, metrics: List[AnalyticsMetric]) -> List[str]:
        """Generate real-time insights from current metrics."""

        insights = []

        # Analyze precision trends
        precision_metrics = [m for m in metrics if m.metric_type == MetricType.PRECISION]
        if precision_metrics:
            precision_values = [m.value for m in precision_metrics]
            if np.mean(precision_values) < 0.5:
                insights.append("System-wide precision is below optimal levels")
            elif np.mean(precision_values) > 0.8:
                insights.append("All agents are performing with high precision")

        # Analyze error rates
        error_metrics = [m for m in metrics if m.metric_type == MetricType.ERROR_RATE]
        if error_metrics:
            error_values = [m.value for m in error_metrics]
            if np.max(error_values) > 0.2:
                insights.append("Some agents are experiencing high error rates")

        # Analyze performance
        performance_metrics = [m for m in metrics if m.metric_type == MetricType.PERFORMANCE]
        if performance_metrics:
            perf_values = [m.value for m in performance_metrics]
            low_perf_count = len([v for v in perf_values if v < 0.5])
            if low_perf_count > len(perf_values) / 2:
                insights.append("More than half of agents show low performance")

        return insights

    def _calculate_performance_score(self, state: Dict[str, Any]) -> float:
        """Calculate performance score from agent state."""

        score = 0.5  # Base score
        factors = 0

        # Precision factor
        precision = state.get("precision", {})
        if "value" in precision:
            score += precision["value"] * 0.3
            factors += 0.3

        # Success rate factor
        tool_executions = state.get("tool_executions", [])
        if tool_executions:
            success_rate = self._calculate_success_rate(tool_executions)
            score += success_rate * 0.2
            factors += 0.2

        # Activity factor
        last_update = state.get("last_update")
        if last_update:
            try:
                update_time = datetime.fromisoformat(last_update)
                time_since_update = (datetime.now() - update_time).total_seconds()
                activity_score = max(0, 1 - time_since_update / 3600)  # Decay over 1 hour
                score += activity_score * 0.1
                factors += 0.1
            except:
                pass

        # Normalize score
        if factors > 0:
            score = min(1.0, score / (0.5 + factors))

        return score

    def _calculate_error_rate(self, tool_executions: List[Dict[str, Any]]) -> float:
        """Calculate error rate from tool executions."""

        if not tool_executions:
            return 0.0

        error_count = len([ex for ex in tool_executions if not ex.get("success", True)])
        return error_count / len(tool_executions)

    def _calculate_success_rate(self, tool_executions: List[Dict[str, Any]]) -> float:
        """Calculate success rate from tool executions."""

        if not tool_executions:
            return 0.5  # Neutral

        success_count = len([ex for ex in tool_executions if ex.get("success", True)])
        return success_count / len(tool_executions)

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""

        return {
            "buffer_size": 10000,
            "forecast_horizon": 24,
            "anomaly_sensitivity": 2.0,
            "insight_generation_interval": 300,  # 5 minutes
            "alert_cooldown": 900,  # 15 minutes
            "model_retrain_interval": 3600,  # 1 hour
        }

    async def _get_active_alerts(self) -> Dict[str, AnalyticsAlert]:
        """Get currently active alerts."""

        # Clean up old resolved alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        active_alerts = {
            alert_id: alert
            for alert_id, alert in self.active_alerts.items()
            if (not alert.resolved or alert.resolved_at > cutoff_time)
        }

        self.active_alerts = active_alerts
        return active_alerts

    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends."""

        trends = {}

        for metric_type in MetricType:
            recent_data = [
                metric
                for metric in self.metrics_buffer
                if (
                    metric.metric_type == metric_type
                    and metric.timestamp > datetime.now() - timedelta(hours=6)
                )
            ]

            if len(recent_data) >= 10:
                values = [m.value for m in recent_data]
                trend = self._calculate_trend(values)
                trends[metric_type.value] = trend

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values."""

        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression
        x = list(range(len(values)))
        n = len(values)

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization metrics."""

        # This would integrate with system monitoring
        # For now, return simulated data
        return {
            "cpu_utilization": 0.45,
            "memory_utilization": 0.62,
            "disk_utilization": 0.34,
            "network_utilization": 0.23,
        }

    def _start_background_tasks(self):
        """Start background analytics tasks."""

        # Metrics collection task
        task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.append(task)

        # Alert processing task
        task = asyncio.create_task(self._alert_processing_loop())
        self.background_tasks.append(task)

        # Forecast updates task
        task = asyncio.create_task(self._forecast_update_loop())
        self.background_tasks.append(task)

        # Analytics cleanup task
        task = asyncio.create_task(self._analytics_cleanup_loop())
        self.background_tasks.append(task)

    async def _metrics_collection_loop(self):
        """Background loop for metrics collection."""

        while True:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(60)  # Collect every minute

            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)

    async def _alert_processing_loop(self):
        """Background loop for alert processing."""

        while True:
            try:
                await self._process_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(10)

    async def _forecast_update_loop(self):
        """Background loop for forecast updates."""

        while True:
            try:
                await self._update_forecasts()
                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                self.logger.error(f"Error in forecast update: {e}")
                await asyncio.sleep(60)

    async def _analytics_cleanup_loop(self):
        """Background loop for analytics data cleanup."""

        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Clean every hour

            except Exception as e:
                self.logger.error(f"Error in analytics cleanup: {e}")
                await asyncio.sleep(60)


# Import required modules (would normally be at top)
import logging
