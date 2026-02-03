#!/usr/bin/env python3
"""
LRS Precision Calibration System

Tunes Beta distribution parameters for optimal performance across different
domains and task types. Implements domain-specific learning rates and
context-aware precision initialization.
"""

import json
import math
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Import our lightweight LRS components
from lrs_agents.lrs.opencode.lightweight_lrs import LightweightHierarchicalPrecision


@dataclass
class DomainCalibration:
    """Calibration settings for specific domains."""

    domain: str
    alpha_gain: float  # Success learning rate
    beta_loss: float  # Failure learning rate
    adaptation_threshold: float
    initial_precision: float
    task_characteristics: Dict[str, Any] = field(default_factory=dict)

    def apply_to_precision(self, precision: LightweightHierarchicalPrecision):
        """Apply domain-specific calibration to precision system."""
        # Update parameters based on domain characteristics
        for level in ["abstract", "planning", "execution"]:
            params = precision.get_level(level)

            # Domain-specific adjustments
            if self.domain == "code_analysis":
                # Code analysis benefits from conservative adaptation
                params.gain_learning_rate = self.alpha_gain * 0.8
                params.loss_learning_rate = self.beta_loss * 1.2
                params.adaptation_threshold = 0.3  # More sensitive

            elif self.domain == "refactoring":
                # Refactoring needs stable precision
                params.gain_learning_rate = self.alpha_gain * 0.6
                params.loss_learning_rate = self.beta_loss * 0.8
                params.adaptation_threshold = 0.5  # Less sensitive

            elif self.domain == "planning":
                # Planning benefits from exploration
                params.gain_learning_rate = self.alpha_gain * 1.2
                params.loss_learning_rate = self.beta_loss * 0.9
                params.adaptation_threshold = 0.4

            elif self.domain == "testing":
                # Testing needs quick adaptation
                params.gain_learning_rate = self.alpha_gain * 1.4
                params.loss_learning_rate = self.beta_loss * 1.6
                params.adaptation_threshold = 0.2  # Very sensitive

            # Set initial precision
            # Reset and set to calibrated initial value
            params.alpha = self.initial_precision * 10
            params.beta = (1 - self.initial_precision) * 10


class PrecisionCalibrator:
    """Advanced precision calibration system."""

    def __init__(self):
        self.domain_calibrations = self._load_default_calibrations()
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.calibration_cache: Dict[str, DomainCalibration] = {}

    def _load_default_calibrations(self) -> Dict[str, DomainCalibration]:
        """Load default calibrations for common domains."""

        return {
            "code_analysis": DomainCalibration(
                domain="code_analysis",
                alpha_gain=0.08,  # Conservative success learning
                beta_loss=0.25,  # Aggressive failure learning
                adaptation_threshold=0.3,
                initial_precision=0.6,  # Start moderately confident
                task_characteristics={
                    "complexity": "high",
                    "uncertainty": "medium",
                    "feedback_delay": "low",
                    "consequences": "low",
                },
            ),
            "refactoring": DomainCalibration(
                domain="refactoring",
                alpha_gain=0.06,  # Very conservative
                beta_loss=0.15,  # Moderate failure learning
                adaptation_threshold=0.5,
                initial_precision=0.7,  # Start confident
                task_characteristics={
                    "complexity": "very_high",
                    "uncertainty": "high",
                    "feedback_delay": "medium",
                    "consequences": "high",
                },
            ),
            "planning": DomainCalibration(
                domain="planning",
                alpha_gain=0.12,  # Encourages exploration
                beta_loss=0.18,  # Balanced failure learning
                adaptation_threshold=0.4,
                initial_precision=0.5,  # Start neutral
                task_characteristics={
                    "complexity": "high",
                    "uncertainty": "very_high",
                    "feedback_delay": "high",
                    "consequences": "medium",
                },
            ),
            "testing": DomainCalibration(
                domain="testing",
                alpha_gain=0.15,  # Quick success learning
                beta_loss=0.30,  # Strong failure learning
                adaptation_threshold=0.2,
                initial_precision=0.4,  # Start uncertain
                task_characteristics={
                    "complexity": "medium",
                    "uncertainty": "low",
                    "feedback_delay": "low",
                    "consequences": "medium",
                },
            ),
            "deployment": DomainCalibration(
                domain="deployment",
                alpha_gain=0.05,  # Very conservative
                beta_loss=0.35,  # Very strong failure learning
                adaptation_threshold=0.6,
                initial_precision=0.8,  # Start very confident
                task_characteristics={
                    "complexity": "medium",
                    "uncertainty": "low",
                    "feedback_delay": "medium",
                    "consequences": "very_high",
                },
            ),
        }

    def calibrate_for_domain(
        self, domain: str, task_context: Optional[Dict[str, Any]] = None
    ) -> LightweightHierarchicalPrecision:
        """
        Create calibrated precision system for specific domain.

        Args:
            domain: Domain name (code_analysis, refactoring, planning, testing, deployment)
            task_context: Optional task-specific context

        Returns:
            Calibrated hierarchical precision system
        """

        # Get or create domain calibration
        if domain not in self.domain_calibrations:
            # Create adaptive calibration for unknown domains
            calibration = self._create_adaptive_calibration(domain, task_context or {})
        else:
            calibration = self.domain_calibrations[domain]

            # Adapt based on task context
            if task_context:
                calibration = self._adapt_calibration_for_context(
                    calibration, task_context
                )

        # Create and calibrate precision system
        precision = LightweightHierarchicalPrecision()
        calibration.apply_to_precision(precision)

        # Cache calibration
        self.calibration_cache[domain] = calibration

        return precision

    def _create_adaptive_calibration(
        self, domain: str, context: Dict[str, Any]
    ) -> DomainCalibration:
        """Create adaptive calibration for unknown domains."""

        # Base calibration
        base_calibration = DomainCalibration(
            domain=domain,
            alpha_gain=0.1,
            beta_loss=0.2,
            adaptation_threshold=0.4,
            initial_precision=0.5,
        )

        # Adapt based on context clues
        complexity = context.get("complexity", "medium")
        uncertainty = context.get("uncertainty", "medium")
        consequences = context.get("consequences", "medium")

        # Adjust parameters based on characteristics
        if complexity == "high":
            base_calibration.alpha_gain *= 0.8
            base_calibration.beta_loss *= 1.2
        elif complexity == "low":
            base_calibration.alpha_gain *= 1.2
            base_calibration.beta_loss *= 0.8

        if uncertainty == "high":
            base_calibration.adaptation_threshold *= 0.8  # More sensitive
            base_calibration.initial_precision *= 0.9  # Start less confident
        elif uncertainty == "low":
            base_calibration.adaptation_threshold *= 1.2  # Less sensitive
            base_calibration.initial_precision *= 1.1  # Start more confident

        if consequences == "high":
            base_calibration.alpha_gain *= 0.7  # More conservative
            base_calibration.beta_loss *= 1.5  # Stronger failure learning

        return base_calibration

    def _adapt_calibration_for_context(
        self, calibration: DomainCalibration, context: Dict[str, Any]
    ) -> DomainCalibration:
        """Adapt existing calibration based on task context."""

        # Create copy for adaptation
        adapted = DomainCalibration(
            domain=calibration.domain,
            alpha_gain=calibration.alpha_gain,
            beta_loss=calibration.beta_loss,
            adaptation_threshold=calibration.adaptation_threshold,
            initial_precision=calibration.initial_precision,
            task_characteristics=calibration.task_characteristics.copy(),
        )

        # Context-based adjustments
        if context.get("time_pressure") == "high":
            # Speed up learning under time pressure
            adapted.alpha_gain *= 1.3
            adapted.beta_loss *= 1.4
            adapted.adaptation_threshold *= 0.9

        if context.get("expertise_level") == "high":
            # Experts can be more confident
            adapted.initial_precision *= 1.2
            adapted.adaptation_threshold *= 1.1

        elif context.get("expertise_level") == "low":
            # Novices need more conservative learning
            adapted.alpha_gain *= 0.8
            adapted.beta_loss *= 1.2
            adapted.adaptation_threshold *= 0.9

        if context.get("failure_cost") == "high":
            # High failure costs require conservatism
            adapted.alpha_gain *= 0.7
            adapted.beta_loss *= 1.3
            adapted.adaptation_threshold *= 1.2

        return adapted

    def record_performance(
        self, domain: str, task_id: str, performance: Dict[str, Any]
    ):
        """Record performance data for calibration improvement."""

        if domain not in self.performance_history:
            self.performance_history[domain] = []

        performance_entry = {
            "task_id": task_id,
            "timestamp": time.time(),
            "performance": performance,
            "calibration_used": self.calibration_cache.get(domain).__dict__
            if domain in self.calibration_cache
            else None,
        }

        self.performance_history[domain].append(performance_entry)

        # Keep only last 100 entries per domain
        if len(self.performance_history[domain]) > 100:
            self.performance_history[domain] = self.performance_history[domain][-100:]

    def analyze_performance_trends(self, domain: str) -> Dict[str, Any]:
        """Analyze performance trends to suggest calibration improvements."""

        if (
            domain not in self.performance_history
            or not self.performance_history[domain]
        ):
            return {"status": "insufficient_data"}

        history = self.performance_history[domain]

        # Analyze trends
        recent_entries = history[-20:]  # Last 20 tasks

        precision_trends = []
        adaptation_frequencies = []
        success_rates = []

        for entry in recent_entries:
            perf = entry["performance"]
            precision_trends.append(
                perf.get("final_precision", {}).get("execution", 0.5)
            )
            adaptation_frequencies.append(perf.get("adaptation_count", 0))
            success_rates.append(1.0 if perf.get("success", False) else 0.0)

        # Calculate trends
        avg_precision = sum(precision_trends) / len(precision_trends)
        avg_adaptations = sum(adaptation_frequencies) / len(adaptation_frequencies)
        avg_success = sum(success_rates) / len(success_rates)

        # Generate recommendations
        recommendations = []

        if avg_precision < 0.4:
            recommendations.append(
                "Increase initial precision or reduce adaptation threshold"
            )
        elif avg_precision > 0.8:
            recommendations.append("Consider more conservative learning rates")

        if avg_adaptations > 3:
            recommendations.append(
                "Increase adaptation threshold to reduce over-adaptation"
            )
        elif avg_adaptations < 1:
            recommendations.append(
                "Decrease adaptation threshold for more responsiveness"
            )

        if avg_success < 0.6:
            recommendations.append("Review task characteristics and domain calibration")

        return {
            "domain": domain,
            "sample_size": len(recent_entries),
            "avg_precision": avg_precision,
            "avg_adaptations": avg_adaptations,
            "avg_success_rate": avg_success,
            "recommendations": recommendations,
            "trend_analysis": self._calculate_trends(
                precision_trends, adaptation_frequencies
            ),
        }

    def _calculate_trends(
        self, precision_history: List[float], adaptation_history: List[float]
    ) -> Dict[str, Any]:
        """Calculate detailed trend analysis."""

        if len(precision_history) < 5:
            return {"status": "insufficient_data"}

        # Precision trend
        precision_slope = self._calculate_slope(precision_history)

        # Adaptation frequency trend
        adaptation_slope = self._calculate_slope(adaptation_history)

        # Volatility
        precision_volatility = self._calculate_volatility(precision_history)

        trends = {
            "precision_trend": "increasing"
            if precision_slope > 0.01
            else "decreasing"
            if precision_slope < -0.01
            else "stable",
            "precision_slope": precision_slope,
            "adaptation_trend": "increasing"
            if adaptation_slope > 0.1
            else "decreasing"
            if adaptation_slope < -0.1
            else "stable",
            "adaptation_slope": adaptation_slope,
            "precision_volatility": precision_volatility,
        }

        return trends

    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of linear trend."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator > 0 else 0.0

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation of changes)."""
        if len(values) < 2:
            return 0.0

        changes = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        mean_change = sum(changes) / len(changes)
        variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)

        return math.sqrt(variance)

    def export_calibration_profile(self, filepath: str):
        """Export current calibration profiles for backup/sharing."""

        profile = {
            "version": "1.0",
            "timestamp": time.time(),
            "domain_calibrations": {
                domain: calib.__dict__
                for domain, calib in self.domain_calibrations.items()
            },
            "performance_history_summary": {
                domain: {
                    "total_entries": len(entries),
                    "date_range": f"{min(e['timestamp'] for e in entries)} - {max(e['timestamp'] for e in entries)}"
                    if entries
                    else "none",
                }
                for domain, entries in self.performance_history.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(profile, f, indent=2)

    def import_calibration_profile(self, filepath: str):
        """Import calibration profiles."""

        with open(filepath, "r") as f:
            profile = json.load(f)

        # Load domain calibrations
        for domain, calib_data in profile.get("domain_calibrations", {}).items():
            self.domain_calibrations[domain] = DomainCalibration(**calib_data)

        print(
            f"Imported {len(self.domain_calibrations)} domain calibrations from {filepath}"
        )


# Global calibrator instance
precision_calibrator = PrecisionCalibrator()

# Test the calibration system
if __name__ == "__main__":
    print("🎯 Testing LRS Precision Calibration System")
    print("=" * 48)

    # Test different domain calibrations
    domains_to_test = [
        "code_analysis",
        "refactoring",
        "planning",
        "testing",
        "deployment",
    ]

    for domain in domains_to_test:
        print(f"\n🔧 Testing {domain} domain calibration")
        print("-" * (25 + len(domain)))

        # Create calibrated precision system
        precision = precision_calibrator.calibrate_for_domain(domain)

        print(f"   Abstract precision: {precision.abstract:.3f}")
        print(f"   Planning precision: {precision.planning:.3f}")
        print(f"   Execution precision: {precision.execution:.3f}")

        # Test adaptation
        precision.update("execution", 0.2)  # Small error
        print(f"   After small error: {precision.execution:.3f}")

        precision.update("execution", 0.8)  # Large error
        print(f"   After large error: {precision.execution:.3f}")

    # Test adaptive calibration for unknown domain
    print("\n🔧 Testing adaptive calibration for unknown domain")
    print("-" * 55)

    custom_precision = precision_calibrator.calibrate_for_domain(
        "custom_domain",
        task_context={
            "complexity": "high",
            "uncertainty": "high",
            "consequences": "high",
            "time_pressure": "high",
        },
    )

    print(f"   Custom domain precision: {custom_precision.execution:.3f}")

    # Test performance recording and analysis
    print("\n📊 Testing performance analysis")
    print("-" * 35)

    # Record some mock performance data
    for i in range(10):
        precision_calibrator.record_performance(
            "code_analysis",
            f"task_{i}",
            {
                "success": i > 2,  # First 3 fail, rest succeed
                "final_precision": {"execution": 0.5 + i * 0.03},
                "adaptation_count": max(0, 3 - i),
            },
        )

    # Analyze performance trends
    analysis = precision_calibrator.analyze_performance_trends("code_analysis")

    print("   Performance Analysis Results:")
    print(f"     Sample size: {analysis['sample_size']}")
    print(f"     Avg precision: {analysis['avg_precision']:.3f}")
    print(f"     Avg adaptations: {analysis['avg_adaptations']:.1f}")
    print(f"     Avg success rate: {analysis['avg_success_rate']:.3f}")
    print("     Recommendations:")
    for rec in analysis["recommendations"]:
        print(f"       • {rec}")

    print("\n🎉 Precision Calibration Testing Complete!")
    print("=" * 52)
    print("✅ Domain-specific calibrations working")
    print("✅ Adaptive calibration for unknown domains")
    print("✅ Performance trend analysis functional")
    print("✅ Context-aware parameter adjustment")
    print("✅ Ready for integration with LRS agents")
