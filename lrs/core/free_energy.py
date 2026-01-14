"""
Expected Free Energy calculation for Active Inference.

G = Epistemic Value - Pragmatic Value
  = H[P(o|s)] - E[log P(o|C)]
  = Information Gain - Expected Reward

Lower G is better (more desirable policies have lower expected free energy).
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

from lrs.core.lens import ToolLens


@dataclass
class PolicyEvaluation:
    """
    Result of evaluating a policy's Expected Free Energy.
    
    Attributes:
        epistemic_value: Information gain (uncertainty reduction)
        pragmatic_value: Expected reward
        total_G: Total free energy (epistemic - pragmatic)
        expected_success_prob: Estimated probability of success
        components: Detailed breakdown of G calculation
    """
    epistemic_value: float
    pragmatic_value: float
    total_G: float
    expected_success_prob: float
    components: Dict[str, Any]


def calculate_epistemic_value(
    policy: List[ToolLens],
    state: Dict[str, Any],
    historical_stats: Optional[Dict[str, Dict]] = None
) -> float:
    """
    Calculate epistemic value (information gain) for a policy.
    
    Epistemic value = H[P(o|s)] where H is entropy.
    
    Higher epistemic value = more uncertain about outcomes = more learning potential
    
    Heuristics for estimating entropy:
    1. Novel tools (never used) → high entropy
    2. Tools with high variance in past outcomes → high entropy
    3. Tools with consistent outcomes → low entropy
    
    Args:
        policy: Sequence of tools
        state: Current agent state
        historical_stats: Optional statistics from past executions
    
    Returns:
        Epistemic value (higher = more informative)
    
    Examples:
        >>> policy = [new_tool, established_tool]
        >>> epistemic = calculate_epistemic_value(policy, state)
        >>> print(epistemic)  # High due to new_tool
        0.85
    """
    if not policy:
        return 0.0
    
    total_entropy = 0.0
    
    for tool in policy:
        # Check if we have historical data
        if historical_stats and tool.name in historical_stats:
            stats = historical_stats[tool.name]
            
            # Estimate entropy from success/failure variance
            success_rate = stats.get('success_rate', 0.5)
            
            # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
            if 0 < success_rate < 1:
                p = success_rate
                entropy = -(p * np.log2(p + 1e-10) + (1-p) * np.log2(1-p + 1e-10))
            else:
                entropy = 0.0  # Deterministic
            
            # Add variance in prediction errors (if available)
            error_variance = stats.get('error_variance', 0.0)
            entropy += error_variance
            
            total_entropy += entropy
        else:
            # No historical data → high uncertainty
            total_entropy += 1.0  # Maximum entropy for binary outcome
    
    # Normalize by policy length
    avg_entropy = total_entropy / len(policy)
    
    return avg_entropy


def calculate_pragmatic_value(
    policy: List[ToolLens],
    state: Dict[str, Any],
    preferences: Dict[str, float],
    historical_stats: Optional[Dict[str, Dict]] = None,
    discount_factor: float = 0.95
) -> float:
    """
    Calculate pragmatic value (expected reward) for a policy.
    
    Pragmatic value = E[log P(o|C)] where C is preferences
    
    Higher pragmatic value = more expected reward
    
    Args:
        policy: Sequence of tools
        state: Current agent state
        preferences: Reward weights (e.g., {'success': 5.0, 'error': -3.0})
        historical_stats: Optional statistics from past executions
        discount_factor: Temporal discount for multi-step policies
    
    Returns:
        Pragmatic value (higher = more rewarding)
    
    Examples:
        >>> policy = [reliable_tool]
        >>> pragmatic = calculate_pragmatic_value(
        ...     policy, state, preferences={'success': 5.0}
        ... )
        >>> print(pragmatic)
        4.5
    """
    if not policy:
        return 0.0
    
    total_reward = 0.0
    cumulative_discount = 1.0
    
    for i, tool in enumerate(policy):
        # Estimate success probability
        if historical_stats and tool.name in historical_stats:
            success_prob = historical_stats[tool.name].get('success_rate', 0.5)
        else:
            success_prob = 0.5  # Neutral prior
        
        # Calculate expected reward for this step
        success_reward = preferences.get('success', 0.0)
        error_penalty = preferences.get('error', 0.0)
        step_cost = preferences.get('step_cost', 0.0)
        
        expected_reward = (
            success_prob * success_reward +
            (1 - success_prob) * error_penalty +
            step_cost
        )
        
        # Apply temporal discount
        total_reward += cumulative_discount * expected_reward
        cumulative_discount *= discount_factor
    
    return total_reward


def calculate_expected_free_energy(
    policy: List[ToolLens],
    state: Dict[str, Any],
    preferences: Dict[str, float],
    historical_stats: Optional[Dict[str, Dict]] = None,
    epistemic_weight: float = 1.0
) -> float:
    """
    Calculate Expected Free Energy for a policy.
    
    G = Epistemic Value - Pragmatic Value
    
    Lower G is better:
    - High epistemic value (learning) → Lower G
    - High pragmatic value (reward) → Lower G
    
    Args:
        policy: Sequence of tools to evaluate
        state: Current agent state
        preferences: Reward function
        historical_stats: Optional execution history
        epistemic_weight: Weight for epistemic term (default: 1.0)
    
    Returns:
        G value (lower is better)
    
    Examples:
        >>> policy = [fetch_tool, parse_tool]
        >>> G = calculate_expected_free_energy(
        ...     policy, state, preferences={'success': 5.0, 'error': -2.0}
        ... )
        >>> print(G)
        -2.3
    """
    epistemic = calculate_epistemic_value(policy, state, historical_stats)
    pragmatic = calculate_pragmatic_value(policy, state, preferences, historical_stats)
    
    # G = Epistemic - Pragmatic
    # (but we weight the epistemic term)
    G = epistemic_weight * epistemic - pragmatic
    
    return G


def evaluate_policy(
    policy: List[ToolLens],
    state: Dict[str, Any],
    preferences: Dict[str, float],
    historical_stats: Optional[Dict[str, Dict]] = None
) -> PolicyEvaluation:
    """
    Fully evaluate a policy and return detailed breakdown.
    
    Args:
        policy: Policy to evaluate
        state: Current state
        preferences: Reward function
        historical_stats: Execution history
    
    Returns:
        PolicyEvaluation with detailed components
    
    Examples:
        >>> evaluation = evaluate_policy(policy, state, preferences)
        >>> print(f"G: {evaluation.total_G:.2f}")
        >>> print(f"Success prob: {evaluation.expected_success_prob:.2%}")
    """
    epistemic = calculate_epistemic_value(policy, state, historical_stats)
    pragmatic = calculate_pragmatic_value(policy, state, preferences, historical_stats)
    G = epistemic - pragmatic
    
    # Estimate success probability
    if historical_stats and policy:
        probs = [
            historical_stats.get(tool.name, {}).get('success_rate', 0.5)
            for tool in policy
        ]
        # Joint probability (assuming independence)
        success_prob = np.prod(probs)
    else:
        success_prob = 0.5 ** len(policy) if policy else 0.0
    
    return PolicyEvaluation(
        epistemic_value=epistemic,
        pragmatic_value=pragmatic,
        total_G=G,
        expected_success_prob=success_prob,
        components={
            'epistemic': epistemic,
            'pragmatic': pragmatic,
            'policy_length': len(policy),
            'tool_names': [t.name for t in policy]
        }
    )


def precision_weighted_selection(
    policies: List[PolicyEvaluation],
    precision: float,
    temperature: float = 1.0
) -> int:
    """
    Select policy via precision-weighted softmax over G values.
    
    P(policy) ∝ exp(-γ · G / T)
    
    Where:
    - γ (precision): High → sharp softmax (exploitation)
                     Low → flat softmax (exploration)
    - T (temperature): Scaling factor
    
    Args:
        policies: List of evaluated policies
        precision: Precision value in [0, 1]
        temperature: Temperature scaling (default: 1.0)
    
    Returns:
        Index of selected policy
    
    Examples:
        >>> policies = [
        ...     PolicyEvaluation(0.8, 3.0, -2.2, 0.7, {}),  # Best G
        ...     PolicyEvaluation(0.9, 2.0, -1.1, 0.6, {}),
        ... ]
        >>> 
        >>> # High precision → likely selects policy 0 (best G)
        >>> idx = precision_weighted_selection(policies, precision=0.9)
        >>> 
        >>> # Low precision → more random exploration
        >>> idx = precision_weighted_selection(policies, precision=0.2)
    """
    if not policies:
        return 0
    
    # Extract G values
    G_values = np.array([p.total_G for p in policies])
    
    # Apply precision-weighted softmax
    # High precision → sharp selection (low effective temperature)
    # Low precision → flat selection (high effective temperature)
    effective_temp = temperature / (precision + 1e-10)
    
    # Softmax: exp(-G/T) / sum(exp(-G/T))
    exp_values = np.exp(-G_values / effective_temp)
    probabilities = exp_values / np.sum(exp_values)
    
    # Sample from distribution
    selected_idx = np.random.choice(len(policies), p=probabilities)
    
    return selected_idx
