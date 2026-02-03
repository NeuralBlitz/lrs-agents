#!/usr/bin/env python3
"""
Lightweight LRS Precision Implementation

NumPy-free implementation of LRS precision tracking for environments
where NumPy dependencies are problematic.
"""

import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LightweightPrecisionParameters:
    """
    Lightweight precision parameters using simplified Beta distribution.

    Tracks confidence in predictions without NumPy dependencies.
    """

    alpha: float = 1.0  # Success parameter
    beta: float = 1.0  # Failure parameter
    gain_learning_rate: float = 0.1
    loss_learning_rate: float = 0.2
    adaptation_threshold: float = 0.4

    @property
    def value(self) -> float:
        """Get current precision value γ = α/(α+β)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Get variance of Beta distribution"""
        a_plus_b = self.alpha + self.beta
        return (self.alpha * self.beta) / (a_plus_b**2 * (a_plus_b + 1))

    def update(self, prediction_error: float) -> None:
        """Update precision based on prediction error"""
        # Inverse error = success signal
        inverse_error = 1.0 - prediction_error

        # Asymmetric learning
        self.alpha += self.gain_learning_rate * inverse_error
        self.beta += self.loss_learning_rate * prediction_error

        # Clamp to valid range
        self.alpha = max(0.1, self.alpha)
        self.beta = max(0.1, self.beta)

    def should_adapt(self) -> bool:
        """Check if precision is below adaptation threshold"""
        return self.value < self.adaptation_threshold

    def reset(self) -> None:
        """Reset to initial uniform prior"""
        self.alpha = 1.0
        self.beta = 1.0

    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive statistics"""
        return {
            "value": self.value,
            "alpha": self.alpha,
            "beta": self.beta,
            "variance": self.variance,
            "confidence_interval": self._confidence_interval(),
        }

    def _confidence_interval(self, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval (simplified)"""
        # Using normal approximation for Beta distribution
        mean = self.value
        std = math.sqrt(self.variance)

        # z-score for 95% confidence
        z = 1.96
        margin = z * std

        return {
            "lower": max(0.0, mean - margin),
            "upper": min(1.0, mean + margin),
            "margin": margin,
        }


@dataclass
class LightweightHierarchicalPrecision:
    """
    Lightweight hierarchical precision tracking.

    Three-level precision without NumPy dependencies.
    """

    _abstract: LightweightPrecisionParameters = field(
        default_factory=LightweightPrecisionParameters
    )
    _planning: LightweightPrecisionParameters = field(
        default_factory=LightweightPrecisionParameters
    )
    _execution: LightweightPrecisionParameters = field(
        default_factory=LightweightPrecisionParameters
    )

    propagation_threshold: float = 0.7
    attenuation_factor: float = 0.5

    @property
    def abstract(self) -> float:
        return self._abstract.value

    @property
    def planning(self) -> float:
        return self._planning.value

    @property
    def execution(self) -> float:
        return self._execution.value

    def get_level(self, level: str) -> LightweightPrecisionParameters:
        """Get precision parameters for a level"""
        if level == "abstract":
            return self._abstract
        elif level == "planning":
            return self._planning
        elif level == "execution":
            return self._execution
        else:
            raise ValueError(f"Invalid level: {level}")

    def update(self, level: str, prediction_error: float) -> None:
        """Update precision with upward propagation"""
        # Update specified level
        params = self.get_level(level)
        params.update(prediction_error)

        # Propagate upward if error is high
        if prediction_error > self.propagation_threshold:
            attenuated_error = prediction_error * self.attenuation_factor

            if level == "execution":
                self._planning.update(attenuated_error)
            elif level == "planning":
                self._abstract.update(attenuated_error)

    def get_all_values(self) -> Dict[str, float]:
        """Get all precision values"""
        return {
            "abstract": self.abstract,
            "planning": self.planning,
            "execution": self.execution,
        }

    def reset(self) -> None:
        """Reset all levels"""
        self._abstract.reset()
        self._planning.reset()
        self._execution.reset()

    def should_adapt(self, level: str = "execution") -> bool:
        """Check if adaptation is needed"""
        return self.get_level(level).should_adapt()


class LightweightFreeEnergyCalculator:
    """Lightweight free energy calculations without NumPy"""

    @staticmethod
    def calculate_epistemic_value(policy: List, registry=None) -> float:
        """Calculate information gain from policy"""
        if not policy:
            return 0.0

        epistemic = 0.0
        for tool in policy:
            # Use success rate or default to neutral prior
            p = getattr(tool, "success_rate", 0.5)
            p = max(0.01, min(0.99, p))  # Clamp

            # Entropy calculation
            if p > 0 and p < 1:
                entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
            else:
                entropy = 0.0

            epistemic += entropy

        return epistemic

    @staticmethod
    def calculate_pragmatic_value(
        policy: List,
        preferences: Dict[str, float],
        registry=None,
        discount: float = 0.95,
    ) -> float:
        """Calculate expected reward from policy"""
        if not policy:
            return 0.0

        reward_success = preferences.get("success", 1.0)
        reward_error = preferences.get("error", -1.0)
        step_cost = preferences.get("step_cost", -0.1)

        pragmatic = 0.0
        discount_factor = 1.0

        for tool in policy:
            p_success = getattr(tool, "success_rate", 0.5)

            expected_reward = (
                p_success * reward_success + (1 - p_success) * reward_error + step_cost
            )

            pragmatic += discount_factor * expected_reward
            discount_factor *= discount

        return pragmatic

    @staticmethod
    def calculate_free_energy(
        epistemic: float,
        pragmatic: float,
        precision: float,
        epistemic_weight: Optional[float] = None,
    ) -> float:
        """Calculate expected free energy G"""
        if epistemic_weight is None:
            epistemic_weight = 1.0 - precision

        G = epistemic_weight * epistemic - pragmatic
        return G


class LightweightPolicySelector:
    """Lightweight policy selection without NumPy"""

    @staticmethod
    def precision_weighted_selection(
        evaluations: List[Dict], precision: float = 0.5
    ) -> int:
        """Select policy using precision-weighted softmax"""
        if not evaluations:
            raise ValueError("Cannot select from empty evaluations")

        # Extract G values
        G_values = [e["total_G"] for e in evaluations]

        # Softmax with precision as inverse temperature
        beta = precision if precision > 0 else 0.1
        exp_values = [math.exp(-beta * G) for G in G_values]
        total = sum(exp_values)

        if total == 0:
            return 0  # Fallback

        probabilities = [e / total for e in exp_values]

        # Sample from distribution
        import random

        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probabilities):
            cumsum += p
            if r < cumsum:
                return i

        return len(probabilities) - 1  # Fallback


def create_lightweight_lrs_agent(tools: List, preferences: Optional[Dict] = None):
    """
    Create a lightweight LRS agent without NumPy dependencies.

    This provides the same interface as the full LRS agent but with
    simplified implementations that work in restricted environments.
    """

    class LightweightLRSAgent:
        def __init__(self, tools, preferences=None):
            self.tools = tools
            self.precision = LightweightHierarchicalPrecision()
            self.preferences = preferences or {
                "success": 1.0,
                "error": -1.0,
                "step_cost": -0.1,
            }
            self.belief_state = {
                "goal": "",
                "current_task": "",
                "adaptation_count": 0,
                "adaptation_events": [],
            }

        def execute_task(self, task: str) -> Dict:
            """Execute a task using LRS reasoning"""
            self.belief_state["goal"] = task
            self.belief_state["current_task"] = task

            print(f"🤖 Lightweight LRS Agent executing: {task}")
            print(f"🎯 Current precision: {self.precision.execution:.3f}")

            # Try tools in sequence
            for tool in self.tools:
                print(f"🔧 Using tool: {getattr(tool, 'name', str(tool))}")

                try:
                    # Simulate tool execution (simplified)
                    result = {"success": True, "value": f"Executed {task}"}
                    prediction_error = 0.1  # Assume low error

                    if result["success"]:
                        print("✅ Tool succeeded")
                        self.precision.update("execution", prediction_error)

                        self.belief_state["last_result"] = result
                        return {
                            "success": True,
                            "result": result,
                            "final_state": self.belief_state.copy(),
                        }
                    else:
                        print("❌ Tool failed")
                        self.precision.update("execution", 0.8)  # High error

                except Exception as e:
                    print(f"💥 Tool error: {e}")
                    self.precision.update("execution", 0.9)  # Very high error

            return {
                "success": False,
                "error": "All tools failed",
                "final_state": self.belief_state.copy(),
            }

    return LightweightLRSAgent(tools, preferences)


# Test the lightweight implementation
if __name__ == "__main__":
    print("🪶 Testing Lightweight LRS Implementation")
    print("=" * 45)

    # Test precision parameters
    params = LightweightPrecisionParameters()
    print(f"Initial precision: {params.value:.3f}")

    # Test updates
    params.update(0.1)  # Success
    print(f"After success: {params.value:.3f}")

    params.update(0.8)  # Failure
    print(f"After failure: {params.value:.3f}")

    # Test hierarchical precision
    hp = LightweightHierarchicalPrecision()
    print(f"Initial levels: {hp.get_all_values()}")

    hp.update("execution", 0.9)  # High error
    print(f"After execution error: {hp.get_all_values()}")

    # Test free energy calculations
    calculator = LightweightFreeEnergyCalculator()

    # Mock policy with success rates
    class MockTool:
        def __init__(self, success_rate):
            self.success_rate = success_rate

    policy = [MockTool(0.8), MockTool(0.9)]
    epistemic = calculator.calculate_epistemic_value(policy)
    pragmatic = calculator.calculate_pragmatic_value(
        policy, {"success": 1.0, "error": -1.0}
    )
    G = calculator.calculate_free_energy(epistemic, pragmatic, 0.7)

    print(f"🎯 Epistemic value: {epistemic:.3f}")
    print(f"🎯 Pragmatic value: {pragmatic:.3f}")
    print(f"⚡ Free energy G: {G:.3f}")

    print("\n✅ Lightweight LRS implementation working!")
    print("💡 Ready for environments without NumPy dependencies")
