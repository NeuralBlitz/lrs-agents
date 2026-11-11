"""Behavioral regression tests for the hybrid Expected Free Energy evaluator."""

from types import SimpleNamespace

import pytest

from lrs.core.free_energy import PolicyEvaluation, calculate_expected_free_energy
from lrs.inference.evaluator import HybridGEvaluator, compare_math_vs_llm


@pytest.fixture
def sample_tool():
    return SimpleNamespace(name="fetch", success_rate=0.9)


@pytest.fixture
def proposal(sample_tool):
    return {
        "policy": [sample_tool],
        "llm_success_prob": 0.7,
        "llm_info_gain": 0.4,
        "strategy": "balanced",
    }


@pytest.fixture
def preferences():
    return {"success": 5.0, "error": -3.0, "step_cost": 0.0}


def test_high_precision_uses_mathematical_free_energy(proposal, preferences):
    evaluator = HybridGEvaluator()
    actual = evaluator.evaluate_hybrid(proposal, {}, preferences, precision=1.0)
    expected = calculate_expected_free_energy(
        proposal["policy"], preferences=preferences, epistemic_weight=1.0
    )
    assert actual == pytest.approx(expected)


def test_low_precision_uses_llm_assessment(proposal, preferences):
    evaluator = HybridGEvaluator()
    actual = evaluator.evaluate_hybrid(proposal, {}, preferences, precision=0.0)
    assert actual == pytest.approx(evaluator._calculate_llm_g(proposal, preferences))


def test_intermediate_precision_interpolates_between_both_estimates(
    proposal, preferences
):
    evaluator = HybridGEvaluator()
    math_g = calculate_expected_free_energy(
        proposal["policy"], preferences=preferences, epistemic_weight=1.0
    )
    llm_g = evaluator._calculate_llm_g(proposal, preferences)
    assert evaluator.evaluate_hybrid(proposal, {}, preferences, precision=0.5) == pytest.approx(
        0.5 * math_g + 0.5 * llm_g
    )


def test_custom_interpolation_weight_is_honored(proposal, preferences):
    evaluator = HybridGEvaluator(lambda_fn=lambda _precision: 0.25)
    math_g = calculate_expected_free_energy(
        proposal["policy"], preferences=preferences, epistemic_weight=1.0
    )
    llm_g = evaluator._calculate_llm_g(proposal, preferences)
    assert evaluator.evaluate_hybrid(proposal, {}, preferences, precision=0.3) == pytest.approx(
        0.75 * math_g + 0.25 * llm_g
    )


def test_historical_success_rates_reach_mathematical_evaluator(proposal, preferences):
    evaluator = HybridGEvaluator()
    high = evaluator.evaluate_hybrid(
        proposal,
        {},
        preferences,
        precision=1.0,
        historical_stats={"fetch": {"success_rate": 0.95}},
    )
    low = evaluator.evaluate_hybrid(
        proposal,
        {},
        preferences,
        precision=1.0,
        historical_stats={"fetch": {"success_rate": 0.05}},
    )
    assert high < low


@pytest.mark.parametrize("precision", [-0.01, 1.01, float("nan")])
def test_invalid_precision_is_rejected(proposal, preferences, precision):
    with pytest.raises(ValueError, match="precision"):
        HybridGEvaluator().evaluate_hybrid(proposal, {}, preferences, precision)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_invalid_interpolation_weight_is_rejected(proposal, preferences, weight):
    with pytest.raises(ValueError, match="interpolation"):
        HybridGEvaluator(lambda_fn=lambda _precision: weight).evaluate_hybrid(
            proposal, {}, preferences, precision=0.5
        )


def test_evaluate_all_returns_consistent_policy_components(proposal, preferences):
    evaluator = HybridGEvaluator()
    evaluations = evaluator.evaluate_all([proposal], {}, preferences, precision=0.4)
    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert isinstance(evaluation, PolicyEvaluation)
    assert evaluation.expected_success_prob == pytest.approx(0.7)
    assert evaluation.total_G == pytest.approx(
        evaluator.evaluate_hybrid(proposal, {}, preferences, precision=0.4)
    )
    assert evaluation.components["G_math"] == pytest.approx(
        calculate_expected_free_energy(
            proposal["policy"], preferences=preferences, epistemic_weight=1.0
        )
    )
    assert evaluation.components["G_llm"] == pytest.approx(
        evaluator._calculate_llm_g(proposal, preferences)
    )
    assert evaluation.components["lambda"] == pytest.approx(0.6)
    assert evaluation.components["strategy"] == "balanced"


def test_evaluate_all_empty_proposals(preferences):
    assert HybridGEvaluator().evaluate_all([], {}, preferences, 0.5) == []


def test_evaluate_all_handles_missing_optional_metadata(sample_tool, preferences):
    evaluation = HybridGEvaluator().evaluate_all(
        [{"policy": [sample_tool]}], {}, preferences, 0.5
    )[0]
    assert evaluation.expected_success_prob == 0.5
    assert evaluation.components["strategy"] == "unknown"


def test_comparison_uses_correct_free_energy_call(proposal, preferences):
    comparison = compare_math_vs_llm(proposal, {}, preferences)
    evaluator = HybridGEvaluator()
    assert comparison["G_math"] == pytest.approx(
        calculate_expected_free_energy(
            proposal["policy"], preferences=preferences, epistemic_weight=1.0
        )
    )
    assert comparison["G_llm"] == pytest.approx(
        evaluator._calculate_llm_g(proposal, preferences)
    )
    assert comparison["difference"] == pytest.approx(
        abs(comparison["G_math"] - comparison["G_llm"])
    )


def test_default_llm_assessments_use_neutral_priors(sample_tool, preferences):
    evaluator = HybridGEvaluator()
    expected = 0.5 - (0.5 * preferences["success"] + 0.5 * preferences["error"])
    assert evaluator._calculate_llm_g({"policy": [sample_tool]}, preferences) == pytest.approx(
        expected
    )
