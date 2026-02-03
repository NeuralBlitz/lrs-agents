"""Free energy calculations for Active Inference."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math

from lrs.core.lens import ToolLens  # CRITICAL: Import ToolLens
from lrs.core.registry import ToolRegistry


@dataclass
class PolicyEvaluation:
    """Results of policy evaluation."""

    epistemic_value: float
    pragmatic_value: float
    total_G: float
    expected_success_prob: float
    components: Dict[str, Any]


def calculate_epistemic_value(
    policy: List[ToolLens],
    registry: Optional[ToolRegistry] = None,
    historical_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Calculate epistemic value (information gain) of a policy.

    Higher values indicate more information gain from exploration.

    Args:
        policy: Sequence of tools to execute
        registry: Tool registry with statistics

    Returns:
        Epistemic value (information gain)

    Example:
        >>> epistemic = calculate_epistemic_value([novel_tool])
        >>> # High value for unexplored tools
    """
    if not policy:
        return 0.0

    epistemic = 0.0
    for tool in policy:
        # If historical stats are provided and available, use them
        if historical_stats is not None and hasattr(tool, "name"):
            tool_name = tool.name
            if tool_name in historical_stats:
                # Use known statistics
                stats = historical_stats[tool_name]
                p = stats.get("success_rate", 0.5)
            else:
                # Novel tool - high uncertainty
                p = 0.5  # Maximum uncertainty
        else:
            # No historical stats provided - treat as novel
            p = 0.5  # Maximum uncertainty

        # Clamp to avoid log(0)
        p = max(0.01, min(0.99, p))

        entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
        epistemic += entropy

    return epistemic


def calculate_pragmatic_value(
    policy: List[ToolLens],
    preferences: Dict[str, float],
    registry: Optional[ToolRegistry] = None,
    historical_stats: Optional[Dict[str, Dict[str, float]]] = None,
    discount_factor: float = 0.95,
) -> float:
    """
    Calculate pragmatic value (expected reward) of a policy.

    Higher values indicate higher expected utility.

    Args:
        policy: Sequence of tools to execute
        preferences: Reward/cost for outcomes (success, error, step_cost)
        registry: Tool registry with statistics
        discount: Temporal discount factor (default: 0.95)

    Returns:
        Pragmatic value (expected reward)

    Example:
        >>> pragmatic = calculate_pragmatic_value(
        ...     [reliable_tool],
        ...     preferences={'success': 5.0, 'error': -3.0}
        ... )
        >>> # High value for reliable tools
    """
    if not policy:
        return 0.0

    reward_success = preferences.get("success", 1.0)
    reward_error = preferences.get("error", -1.0)
    step_cost = preferences.get("step_cost", -0.1)

    pragmatic = 0.0
    discount_factor = 1.0

    for tool in policy:
        p_success = tool.success_rate if hasattr(tool, "success_rate") else 0.5

        # Expected reward for this step
        expected_reward = p_success * reward_success + (1 - p_success) * reward_error + step_cost

        pragmatic += discount_factor * expected_reward
        discount_factor *= discount_factor

    return pragmatic


def calculate_expected_free_energy(
    policy: List[ToolLens],
    registry: Optional[ToolRegistry] = None,
    preferences: Optional[Dict[str, float]] = None,
    precision: Optional[float] = 0.5,
    historical_stats: Optional[Dict[str, Dict[str, float]]] = None,
    epistemic_weight: Optional[float] = None,
) -> float:
    """
    Calculate Expected Free Energy G(π) for a policy.

    G(π) = Epistemic Value - Pragmatic Value

    Lower G is better (minimization objective).

    Args:
        policy: Sequence of tools to execute
        registry: Tool registry with statistics
        preferences: Reward structure
        precision: Current precision γ ∈ [0,1]
        epistemic_weight: Override for epistemic term weight

    Returns:
        Expected Free Energy G

    Example:
        >>> G = calculate_expected_free_energy(
        ...     policy=[search_tool, filter_tool],
        ...     preferences={'success': 5.0, 'error': -3.0},
        ...     precision=0.7
        ... )
        >>> # Low G indicates good policy
    """
    if not policy:
        return 0.0

    if preferences is None:
        preferences = {"success": 1.0, "error": -1.0, "step_cost": -0.1}

    # Calculate components
    epistemic = calculate_epistemic_value(policy, registry, historical_stats)
    pragmatic = calculate_pragmatic_value(policy, preferences, registry, historical_stats)

    # Weight epistemic term by uncertainty (1 - precision)
    if epistemic_weight is None and precision is not None:
        epistemic_weight = 1.0 - precision
    elif epistemic_weight is None:
        epistemic_weight = 0.5  # Default

    # G = Epistemic - Pragmatic
    G = epistemic_weight * epistemic - pragmatic

    return G


def evaluate_policy(
    policy: List[ToolLens],
    registry: Optional[ToolRegistry] = None,
    preferences: Optional[Dict[str, float]] = None,
    historical_stats: Optional[Dict[str, Dict[str, float]]] = None,
    precision: Optional[float] = 0.5,
) -> PolicyEvaluation:
    """
    Evaluate a single policy comprehensively.

    Returns:
        PolicyEvaluation with all metrics
    """
    if preferences is None:
        preferences = {"success": 1.0, "error": -1.0, "step_cost": -0.1}

    epistemic = calculate_epistemic_value(policy, registry, historical_stats)
    pragmatic = calculate_pragmatic_value(policy, preferences, registry, historical_stats)
    total_G = calculate_expected_free_energy(
        policy, registry, preferences, precision, historical_stats
    )

    # Calculate expected success probability
    expected_success_prob = 0.5  # Default
    if historical_stats and len(policy) == 1:
        tool_name = policy[0].name if hasattr(policy[0], "name") else str(policy[0])
        if tool_name in historical_stats:
            expected_success_prob = historical_stats[tool_name].get("success_rate", 0.5)

    return PolicyEvaluation(
        epistemic_value=epistemic,
        pragmatic_value=pragmatic,
        total_G=total_G,
        expected_success_prob=expected_success_prob,
        components={"epistemic": epistemic, "pragmatic": pragmatic, "policy_length": len(policy)},
    )


def precision_weighted_selection(
    policy_evaluations: List[PolicyEvaluation],
    precision: float = 0.5,
    temperature: float = 1.0,
    evaluations: Optional[List[PolicyEvaluation]] = None,  # For backward compatibility
) -> int:
    """
    Select policy using precision-weighted softmax.

    P(π) ∝ exp(-β * G(π))

    where β = precision (inverse temperature).

    Args:
        policy_evaluations: Evaluated policies
        precision: Current precision γ ∈ [0,1]
        temperature: Additional temperature parameter

    Returns:
        Index of selected policy

    Example:
        >>> selected_idx = precision_weighted_selection(
        ...     evaluations,
        ...     precision=0.7
        ... )
        >>> best_policy = policies[selected_idx]
    """
    import random

    # Use evaluations parameter if provided (for backward compatibility)
    evaluations = evaluations or policy_evaluations

    if not evaluations:
        raise ValueError("Cannot select from empty evaluations")

    # Extract G values
    G_values = [eval.total_G for eval in evaluations]

    # Softmax with precision as inverse temperature
    beta = precision / temperature
    exp_values = [math.exp(-beta * G) for G in G_values]
    total = sum(exp_values)

    probabilities = [e / total for e in exp_values]

    # Sample from distribution
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probabilities):
        cumsum += p
        if r < cumsum:
            return i

    return len(probabilities) - 1  # Fallback
