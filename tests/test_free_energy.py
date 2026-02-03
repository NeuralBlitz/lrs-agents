"""Tests for free energy calculations."""

import pytest
import numpy as np

from lrs.core.free_energy import (
    calculate_epistemic_value,
    calculate_pragmatic_value,
    calculate_expected_free_energy,
    # evaluate_policy,  # REMOVE THIS LINE
    precision_weighted_selection,
    PolicyEvaluation,
)
from lrs.core.lens import ToolLens, ExecutionResult


class MockTool(ToolLens):
    """Mock tool for testing"""

    def __init__(self, name="mock", success_rate=1.0):
        super().__init__(name, {}, {})
        self._success_rate = success_rate

    @property
    def success_rate(self):
        return self._success_rate

    def get(self, state):
        success = np.random.random() < self.success_rate
        return ExecutionResult(success, "result", None, 0.1 if success else 0.9)

    def set(self, state, observation):
        return state


class TestEpistemicValue:
    """Test epistemic value calculation"""

    def test_novel_tool_high_entropy(self):
        """Novel tools (no history) should have high epistemic value"""
        policy = [MockTool("novel_tool")]

        epistemic = calculate_epistemic_value(policy, {}, historical_stats=None)

        assert epistemic > 0.5  # High uncertainty

    def test_known_tool_low_entropy(self):
        """Known tools with consistent results should have low epistemic value"""
        policy = [MockTool("known_tool")]

        # Provide history showing high success rate
        historical_stats = {"known_tool": {"success_rate": 0.95, "error_variance": 0.01}}

        epistemic = calculate_epistemic_value(policy, {}, historical_stats)

        assert epistemic < 0.3  # Low uncertainty

    def test_uncertain_tool_medium_entropy(self):
        """Tools with 50/50 success rate should have high entropy"""
        policy = [MockTool("uncertain_tool")]

        historical_stats = {"uncertain_tool": {"success_rate": 0.5, "error_variance": 0.3}}

        epistemic = calculate_epistemic_value(policy, {}, historical_stats)

        assert epistemic > 0.5  # High entropy from uncertainty

    def test_multi_tool_policy(self):
        """Multi-tool policies should aggregate epistemic value"""
        policy = [MockTool("tool_a"), MockTool("tool_b")]

        epistemic = calculate_epistemic_value(policy, {}, None)

        # Should be higher than single tool
        single_epistemic = calculate_epistemic_value([MockTool("tool_a")], {}, None)
        assert epistemic >= single_epistemic


class TestPragmaticValue:
    """Test pragmatic value calculation"""

    def test_high_success_high_pragmatic(self):
        """High success probability should yield high pragmatic value"""
        policy = [MockTool("reliable_tool")]

        preferences = {"success": 5.0, "error": -3.0}

        historical_stats = {"reliable_tool": {"success_rate": 0.9}}

        pragmatic = calculate_pragmatic_value(policy, {}, preferences, historical_stats)

        assert pragmatic > 0  # Positive expected reward

    def test_low_success_low_pragmatic(self):
        """Low success probability should yield low/negative pragmatic value"""
        policy = [MockTool("unreliable_tool")]

        preferences = {"success": 5.0, "error": -3.0}

        historical_stats = {
            "unreliable_tool": {
                "success_rate": 0.2  # Usually fails
            }
        }

        pragmatic = calculate_pragmatic_value(policy, {}, preferences, historical_stats)

        assert pragmatic < 0  # Negative expected reward

    def test_temporal_discounting(self):
        """Later steps should be discounted"""
        policy = [MockTool(f"tool_{i}") for i in range(5)]

        preferences = {"success": 5.0}
        historical_stats = {f"tool_{i}": {"success_rate": 1.0} for i in range(5)}

        pragmatic = calculate_pragmatic_value(
            policy, {}, preferences, historical_stats, discount_factor=0.9
        )

        # Should be less than 5 steps * 5.0 reward due to discounting
        assert pragmatic < 25.0

    def test_step_cost(self):
        """Step costs should reduce pragmatic value"""
        policy = [MockTool("tool")]

        preferences = {"success": 5.0, "step_cost": -0.5}

        historical_stats = {"tool": {"success_rate": 1.0}}

        pragmatic = calculate_pragmatic_value(policy, {}, preferences, historical_stats)

        # Should include step cost
        assert pragmatic < 5.0


class TestExpectedFreeEnergy:
    """Test full G calculation"""

    def test_G_calculation(self):
        """G = Epistemic - Pragmatic"""
        policy = [MockTool("tool")]
        preferences = {"success": 5.0}

        epistemic = calculate_epistemic_value(policy, {}, None)
        pragmatic = calculate_pragmatic_value(policy, {}, preferences, None)

        G = calculate_expected_free_energy(policy, {}, preferences, None)

        # Should equal epistemic - pragmatic
        assert abs(G - (epistemic - pragmatic)) < 0.01

    def test_lower_G_is_better(self):
        """Lower G should indicate better policy"""
        good_policy = [MockTool("good_tool")]
        bad_policy = [MockTool("bad_tool")]

        preferences = {"success": 5.0, "error": -3.0}

        historical_stats = {
            "good_tool": {"success_rate": 0.9, "error_variance": 0.01},
            "bad_tool": {"success_rate": 0.3, "error_variance": 0.5},
        }

        G_good = calculate_expected_free_energy(good_policy, {}, preferences, historical_stats)
        G_bad = calculate_expected_free_energy(bad_policy, {}, preferences, historical_stats)

        assert G_good < G_bad

    def test_epistemic_weight(self):
        """Epistemic weight should affect G"""
        policy = [MockTool("tool")]
        preferences = {"success": 5.0}

        G_default = calculate_expected_free_energy(
            policy, {}, preferences, None, epistemic_weight=1.0
        )

        G_high_epistemic = calculate_expected_free_energy(
            policy, {}, preferences, None, epistemic_weight=2.0
        )

        # Higher epistemic weight → more emphasis on information gain
        assert G_high_epistemic != G_default


class TestPolicyEvaluation:
    """Test PolicyEvaluation dataclass"""

    def test_evaluate_policy(self):
        """Test full policy evaluation"""
        policy = [MockTool("tool")]
        preferences = {"success": 5.0}

        evaluation = evaluate_policy(policy, {}, preferences, None)

        assert isinstance(evaluation, PolicyEvaluation)
        assert evaluation.epistemic_value >= 0
        assert "tool_names" in evaluation.components

    def test_evaluation_components(self):
        """Evaluation should include detailed components"""
        policy = [MockTool("tool_a"), MockTool("tool_b")]
        evaluation = evaluate_policy(policy, {}, {"success": 5.0}, None)

        assert "epistemic" in evaluation.components
        assert "pragmatic" in evaluation.components
        assert "policy_length" in evaluation.components
        assert evaluation.components["policy_length"] == 2


class TestPrecisionWeightedSelection:
    """Test policy selection via precision-weighted softmax"""

    def test_high_precision_exploits(self):
        """High precision should select best policy deterministically"""
        # Create policies with different G values
        policies = [
            PolicyEvaluation(0.5, 5.0, -4.5, 0.9, {}),  # Best (lowest G)
            PolicyEvaluation(0.8, 3.0, -2.2, 0.6, {}),
            PolicyEvaluation(0.9, 2.0, -1.1, 0.5, {}),
        ]

        # High precision → deterministic selection
        np.random.seed(42)
        selections = [precision_weighted_selection(policies, precision=0.95) for _ in range(100)]

        # Should mostly select policy 0 (best G)
        assert selections.count(0) > 80

    def test_low_precision_explores(self):
        """Low precision should explore more uniformly"""
        policies = [
            PolicyEvaluation(0.5, 5.0, -4.5, 0.9, {}),
            PolicyEvaluation(0.8, 3.0, -2.2, 0.6, {}),
            PolicyEvaluation(0.9, 2.0, -1.1, 0.5, {}),
        ]

        # Low precision → more exploration
        np.random.seed(42)
        selections = [precision_weighted_selection(policies, precision=0.2) for _ in range(300)]

        # Should have more diversity
        assert len(set(selections)) == 3  # All policies selected
        assert 50 < selections.count(0) < 250  # Not too deterministic

    def test_temperature_scaling(self):
        """Temperature should affect selection"""
        policies = [
            PolicyEvaluation(0.5, 5.0, -4.5, 0.9, {}),
            PolicyEvaluation(0.8, 3.0, -2.2, 0.6, {}),
        ]

        # Higher temperature → more uniform
        np.random.seed(42)
        selections_high_temp = [
            precision_weighted_selection(policies, precision=0.5, temperature=2.0)
            for _ in range(100)
        ]

        np.random.seed(42)
        selections_low_temp = [
            precision_weighted_selection(policies, precision=0.5, temperature=0.5)
            for _ in range(100)
        ]

        # Higher temp should have more diversity
        diversity_high = len(set(selections_high_temp))
        diversity_low = len(set(selections_low_temp))

        assert diversity_high >= diversity_low

    def test_empty_policies(self):
        """Should raise ValueError for empty policy list"""
        with pytest.raises(ValueError, match="Cannot select from empty evaluations"):
            precision_weighted_selection([], precision=0.5)


class TestFreeEnergyEdgeCases:
    """Test edge cases"""

    def test_empty_policy(self):
        """Empty policy should have zero G"""
        G = calculate_expected_free_energy([], {}, {"success": 5.0}, None)
        assert G == 0.0

    def test_no_historical_stats(self):
        """Should handle missing historical stats"""
        policy = [MockTool("new_tool")]
        G = calculate_expected_free_energy(policy, {}, {"success": 5.0}, None)

        # Should use neutral priors
        assert -10 < G < 10

    def test_missing_preferences(self):
        """Should handle missing preferences gracefully"""
        policy = [MockTool("tool")]

        # Empty preferences
        G = calculate_expected_free_energy(policy, {}, {}, None)

        # Should still calculate (with zero pragmatic value)
        assert isinstance(G, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
