"""
Expected Free Energy calculation for policy evaluation.

Implements the core active inference objective:
    G(π) = E_Q(s|π)[H[P(o|s)]] - E_Q(s|π)[ln P(o|C)]
         = Epistemic Value - Pragmatic Value

Where:
    - Epistemic Value: Expected information gain (exploration)
    - Pragmatic Value: Expected reward under preferences (exploitation)
"""

from typing import List, Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass

from lrs.core.lens import ToolLens


@dataclass
class PolicyEvaluation:
    """
    Complete evaluation of a policy's Expected Free Energy.
    
    Attributes:
        epistemic_value (float): Information gain H[P(o|s)].
            Higher = more uncertainty reduction.
        pragmatic_value (float): Expected reward E[ln P(o|C)].
            Higher = better goal satisfaction.
        total_G (float): G = Epistemic - Pragmatic.
            Lower G = better policy (Free Energy minimization).
        expected_success_prob (float): P(success|π) from historical stats.
        components (Dict): Breakdown of G calculation for debugging.
    """
    epistemic_value: float
    pragmatic_value: float
    total_G: float
    expected_success_prob: float
    components: Dict[str, float]


def calculate_epistemic_value(
    policy: List[ToolLens],
    state: Optional[Dict] = None
) -> float:
    """
    Calculate epistemic value: expected information gain.
    
    Epistemic value = Σ_t H[P(o_t | s_t)]
    
    For each tool in the policy, calculate the entropy of predicted outcomes.
    High entropy = high uncertainty = high information gain if executed.
    
    Implementation uses historical success rates as proxy for outcome distribution:
        P(success) = 1 - (failures / total_calls)
        P(failure) = failures / total_calls
        H = -Σ p log p
    
    Args:
        policy (List[ToolLens]): Sequence of tools to evaluate.
        state (Dict, optional): Current belief state. Not used in basic version.
    
    Returns:
        float: Total epistemic value. Range: [0, len(policy)].
            0 = deterministic outcomes (no information gain)
            len(policy) = maximum uncertainty (uniform distribution)
    
    Examples:
        >>> tool1 = ToolLens(...)
        >>> tool1.call_count = 10
        >>> tool1.failure_count = 5  # p(success) = 0.5
        >>> calculate_epistemic_value([tool1])
        1.0  # Maximum entropy for binary outcome
        
        >>> tool2 = ToolLens(...)
        >>> tool2.call_count = 10
        >>> tool2.failure_count = 0  # p(success) = 1.0
        >>> calculate_epistemic_value([tool2])
        0.0  # No uncertainty
    """
    total_epistemic = 0.0
    
    for tool in policy:
        if tool.call_count == 0:
            # Never tried = maximum epistemic value
            total_epistemic += 1.0
            continue
        
        # Calculate outcome probabilities
        p_success = 1.0 - (tool.failure_count / tool.call_count)
        p_failure = tool.failure_count / tool.call_count
        
        # Shannon entropy: H = -Σ p log p
        entropy = 0.0
        for p in [p_success, p_failure]:
            if p > 0:  # Avoid log(0)
                entropy -= p * np.log(p)
        
        total_epistemic += entropy
    
    return total_epistemic


def calculate_pragmatic_value(
    policy: List[ToolLens],
    state: Dict,
    preferences: Dict[str, float],
    discount_factor: float = 0.95
) -> float:
    """
    Calculate pragmatic value: expected reward under preferences.
    
    Pragmatic value = Σ_t γ^t E_Q[ln P(o_t | C)]
    
    Where:
        - γ is temporal discount factor
        - C represents goal preferences
        - P(o|C) is likelihood of observation given preferences
    
    Implementation:
        1. Simulate policy execution (forward model)
        2. Calculate reward at each step based on preferences
        3. Apply temporal discounting
    
    Args:
        policy (List[ToolLens]): Sequence of tools to evaluate.
        state (Dict): Current belief state.
        preferences (Dict[str, float]): Goal preferences.
            Keys are state features, values are reward weights.
            Example: {'file_loaded': 2.0, 'error': -5.0}
        discount_factor (float): Temporal discount γ ∈ [0, 1].
            Higher = more weight on immediate rewards.
    
    Returns:
        float: Total pragmatic value. Higher = better goal satisfaction.
    
    Examples:
        >>> preferences = {'data_retrieved': 3.0, 'error': -5.0}
        >>> policy = [fetch_tool, parse_tool]
        >>> calculate_pragmatic_value(policy, state, preferences)
        5.7  # Discounted sum of expected rewards
    """
    total_pragmatic = 0.0
    current_state = state.copy()
    
    for t, tool in enumerate(policy):
        # Simulate tool execution (using historical stats)
        p_success = 1.0 - (tool.failure_count / (tool.call_count + 1))
        
        # Calculate expected reward for this step
        step_reward = 0.0
        
        for feature, weight in preferences.items():
            # Check if tool's output schema includes this feature
            if feature in tool.output_schema.get('required', []):
                # Weight by success probability
                step_reward += weight * p_success
            elif feature == 'error':
                # Penalize expected failures
                step_reward += weight * (1 - p_success)
        
        # Apply temporal discount
        discounted_reward = (discount_factor ** t) * step_reward
        total_pragmatic += discounted_reward
        
        # Simulate state update for next tool
        # (In full implementation, would use tool.set())
        current_state['step'] = t + 1
    
    return total_pragmatic


def calculate_expected_free_energy(
    policy: List[ToolLens],
    state: Dict,
    preferences: Dict[str, float],
    discount_factor: float = 0.95
) -> PolicyEvaluation:
    """
    Complete Expected Free Energy calculation: G(π) = Epistemic - Pragmatic
    
    This is the core objective function for active inference agents.
    Policies are selected to MINIMIZE G, which balances:
        - Minimizing uncertainty (epistemic value)
        - Maximizing reward (pragmatic value)
    
    Args:
        policy (List[ToolLens]): Candidate policy to evaluate.
        state (Dict): Current belief state.
        preferences (Dict[str, float]): Goal preferences.
        discount_factor (float): Temporal discount factor.
    
    Returns:
        PolicyEvaluation: Complete breakdown of G calculation.
    
    Theoretical Background:
        In the Free Energy Principle (Friston, 2010), agents minimize
        Expected Free Energy to balance exploration and exploitation:
        
        - Low precision (γ) → High epistemic weight → Exploration
        - High precision (γ) → High pragmatic weight → Exploitation
        
        This provides a principled alternative to ε-greedy or UCB.
    
    Examples:
        >>> policy = [tool1, tool2]
        >>> state = {'goal': 'extract_data'}
        >>> preferences = {'data_extracted': 5.0, 'error': -3.0}
        >>> 
        >>> eval_result = calculate_expected_free_energy(
        ...     policy, state, preferences
        ... )
        >>> print(eval_result.total_G)
        -2.3  # Negative = good (more pragmatic than epistemic)
        >>> 
        >>> print(eval_result.components)
        {'epistemic': 0.7, 'pragmatic': 3.0, 'G': -2.3}
    """
    # Calculate components
    epistemic = calculate_epistemic_value(policy, state)
    pragmatic = calculate_pragmatic_value(policy, state, preferences, discount_factor)
    
    # G = Epistemic - Pragmatic
    G = epistemic - pragmatic
    
    # Calculate expected success probability
    success_probs = [
        1.0 - (tool.failure_count / (tool.call_count + 1))
        for tool in policy
    ]
    expected_success = np.prod(success_probs) if success_probs else 0.0
    
    return PolicyEvaluation(
        epistemic_value=epistemic,
        pragmatic_value=pragmatic,
        total_G=G,
        expected_success_prob=expected_success,
        components={
            'epistemic': epistemic,
            'pragmatic': pragmatic,
            'G': G,
            'policy_length': len(policy),
            'mean_tool_experience': np.mean([t.call_count for t in policy])
        }
    )


def precision_weighted_selection(
    evaluations: List[PolicyEvaluation],
    precision: float,
    temperature: float = 1.0
) -> int:
    """
    Select policy index via precision-weighted softmax over G values.
    
    Selection probability: P(π_i) ∝ exp(-γ * G_i / T)
    
    Where:
        - γ (precision): Confidence in world model
        - G_i: Expected Free Energy of policy i
        - T: Temperature (exploration parameter)
    
    High precision (γ → 1):
        Softmax becomes sharper → exploit best policy
    
    Low precision (γ → 0):
        Softmax flattens → explore alternatives
    
    Args:
        evaluations (List[PolicyEvaluation]): Evaluated candidate policies.
        precision (float): Current precision value γ ∈ [0, 1].
        temperature (float): Softmax temperature. Higher = more exploration.
    
    Returns:
        int: Index of selected policy.
    
    Raises:
        ValueError: If evaluations is empty.
    
    Examples:
        >>> evals = [
        ...     PolicyEvaluation(epistemic=0.5, pragmatic=2.0, total_G=-1.5, ...),
        ...     PolicyEvaluation(epistemic=0.8, pragmatic=1.0, total_G=-0.2, ...)
        ... ]
        >>> 
        >>> # High precision → exploit (choose lowest G)
        >>> precision_weighted_selection(evals, precision=0.9)
        0  # Policy with G=-1.5
        >>> 
        >>> # Low precision → explore (more random)
        >>> precision_weighted_selection(evals, precision=0.2)
        1  # Might choose suboptimal policy
    """
    if not evaluations:
        raise ValueError("Cannot select from empty evaluations")
    
    # Extract G values
    G_values = np.array([e.total_G for e in evaluations])
    
    # Precision-weighted softmax: exp(-γ * G / T)
    scaled_G = -precision * G_values / temperature
    
    # Numerical stability: subtract max before exp
    scaled_G = scaled_G - np.max(scaled_G)
    exp_vals = np.exp(scaled_G)
    
    # Softmax probabilities
    probs = exp_vals / np.sum(exp_vals)
    
    # Sample from distribution
    selected_idx = np.random.choice(len(evaluations), p=probs)
    
    return selected_idx
