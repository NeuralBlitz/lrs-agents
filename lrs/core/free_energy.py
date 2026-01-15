"""Free energy calculations for Active Inference."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from lrs.core.lens import ToolLens  # ADD THIS LINE
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
    policy: List[ToolLens],  # Now ToolLens is defined
    registry: Optional[ToolRegistry] = None
) -> float: Calculate epistemic value (information gain) of a policy.
    
    # ... rest of implementation


import numpy as np
from typing import List, Dict, Optional

# Constants for numerical stability
EPSILON = 1e-12

def calculate_epistemic_value(
    policy: List[ToolLens],
    state: Optional[Dict] = None
) -> float:
    total_epistemic = 0.0
    
    for tool in policy:
        if tool.call_count == 0:
            total_epistemic += 1.0
            continue
        
        # Use Laplace Smoothing (Add-one smoothing) for robust probabilities
        # This prevents p being exactly 0 or 1
        p_success = (tool.call_count - tool.failure_count + 1) / (tool.call_count + 2)
        p_failure = 1.0 - p_success
        
        # Shannon entropy with stability epsilon
        entropy = -(p_success * np.log(p_success + EPSILON) + 
                   p_failure * np.log(p_failure + EPSILON))
        
        total_epistemic += entropy
    
    return total_epistemic

def calculate_pragmatic_value(
    policy: List[ToolLens],
    state: Dict,
    preferences: Dict[str, float],
    discount_factor: float = 0.95
) -> float:
    total_pragmatic = 0.0
    
    for t, tool in enumerate(policy):
        # Laplace-smoothed success probability
        p_success = (tool.call_count - tool.failure_count + 1) / (tool.call_count + 2)
        step_reward = 0.0
        
        # Fix: Better schema checking
        required_features = tool.output_schema.get('required', [])
        
        for feature, weight in preferences.items():
            if feature in required_features:
                step_reward += weight * p_success
            elif feature == 'error':
                step_reward += weight * (1.0 - p_success)
        
        total_pragmatic += (discount_factor ** t) * step_reward
        
    return total_pragmatic

def precision_weighted_selection(
    evaluations: List[PolicyEvaluation],
    precision: float,
    temperature: float = 1.0
) -> int:
    if not evaluations:
        raise ValueError("Cannot select from empty evaluations")
    
    # Extract G values and ensure they are finite
    G_values = np.array([e.total_G for e in evaluations])
    G_values = np.nan_to_num(G_values, nan=100.0, posinf=100.0, neginf=-100.0)
    
    # Scale G by precision and temperature
    # We negate G because we want to MINIMIZE G (min G = max probability)
    logits = - (precision * G_values) / (temperature + EPSILON)
    
    # Numerical stability: subtract max logit
    logits -= np.max(logits)
    exp_vals = np.exp(logits)
    
    probs = exp_vals / np.sum(exp_vals)
    
    # Final guardrail: if probs are NaN (can happen with all -inf logits)
    if np.any(np.isnan(probs)):
        probs = np.ones(len(evaluations)) / len(evaluations)
        
    return np.random.choice(len(evaluations), p=probs)
