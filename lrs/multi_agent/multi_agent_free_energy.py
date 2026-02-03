"""
Free Energy calculation for multi-agent systems.

Extends single-agent G to include social uncertainty.
"""

from typing import List, Dict, Any
import numpy as np

from lrs.core.free_energy import calculate_epistemic_value, calculate_pragmatic_value
from lrs.core.lens import ToolLens


def calculate_social_free_energy(social_precisions: Dict[str, float], weight: float = 1.0) -> float:
    """
    Calculate Free Energy from social uncertainty.

    G_social = Σ (1 - γ_social[agent_i])

    Low social precision → high social Free Energy → value in communication

    Args:
        social_precisions: Dict mapping agent IDs to social precision values
        weight: Weight for social term

    Returns:
        Social Free Energy value

    Examples:
        >>> social_precs = {
        ...     'agent_b': 0.8,  # High trust
        ...     'agent_c': 0.3   # Low trust → high uncertainty
        ... }
        >>> G_social = calculate_social_free_energy(social_precs)
        >>> print(G_social)
        0.9  # Dominated by uncertainty about agent_c
    """
    if not social_precisions:
        return 0.0

    # Sum of uncertainties
    total_uncertainty = sum(1.0 - precision for precision in social_precisions.values())

    return weight * total_uncertainty


def calculate_total_free_energy(
    policy: List[ToolLens],
    state: Dict[str, Any],
    preferences: Dict[str, float],
    social_precisions: Dict[str, float],
    historical_stats: Dict[str, Dict] = None,
    social_weight: float = 1.0,
) -> float:
    """
    Calculate total Free Energy including social component.

    G_total = G_env + α * G_social

    Where:
    - G_env: Environmental Free Energy (epistemic - pragmatic)
    - G_social: Social uncertainty
    - α: Social weight

    Args:
        policy: Tool sequence
        state: Current state
        preferences: Reward function
        social_precisions: Social precision for each agent
        historical_stats: Execution history
        social_weight: Weight for social term

    Returns:
        Total Free Energy

    Examples:
        >>> # Policy with communication action
        >>> policy = [fetch_tool, communicate_tool]
        >>>
        >>> social_precs = {'agent_b': 0.3}  # Low → communication valuable
        >>>
        >>> G_total = calculate_total_free_energy(
        ...     policy, state, preferences, social_precs
        ... )
        >>>
        >>> # Communication becomes more attractive when social precision is low
    """
    # Environmental Free Energy
    epistemic = calculate_epistemic_value(policy, registry=None)
    pragmatic = calculate_pragmatic_value(policy, preferences, registry=None)
    G_env = epistemic - pragmatic

    # Social Free Energy
    G_social = calculate_social_free_energy(social_precisions, weight=social_weight)

    # Total
    G_total = G_env + G_social

    return G_total


def should_communicate_based_on_G(
    G_communicate: float, G_no_communicate: float, precision: float = 0.5
) -> bool:
    """
    Decide whether to communicate based on Free Energy.

    Communication is chosen when G(communicate) < G(no_communicate)

    Args:
        G_communicate: Free Energy with communication
        G_no_communicate: Free Energy without communication
        precision: Current precision (for stochastic selection)

    Returns:
        True if should communicate

    Examples:
        >>> # High social uncertainty → communication has lower G
        >>> G_comm = -1.5  # Reduces social uncertainty
        >>> G_no_comm = 0.5
        >>>
        >>> should_comm = should_communicate_based_on_G(G_comm, G_no_comm)
        >>> print(should_comm)
        True
    """
    # Deterministic: choose lower G
    if precision > 0.9:
        return G_communicate < G_no_communicate

    # Stochastic: softmax selection
    temp = 1.0 / (precision + 0.1)

    prob_communicate = np.exp(-G_communicate / temp)
    prob_no_communicate = np.exp(-G_no_communicate / temp)

    total = prob_communicate + prob_no_communicate
    prob_communicate /= total

    return np.random.random() < prob_communicate
