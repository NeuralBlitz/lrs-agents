"""Hybrid Expected Free Energy evaluation."""

from dataclasses import dataclass  # Add this
from typing import List, Dict, Any, Optional, Tuple
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from lrs.core.lens import ToolLens
from lrs.core.free_energy import (
    calculate_expected_free_energy,
    PolicyEvaluation
)
from lrs.core.precision import PrecisionParameters

class HybridGEvaluator:
    """
    Evaluate policies using both LLM priors and mathematical statistics.
    
    G_hybrid = (1 - λ) * G_math + λ * G_llm
    
    Where:
    - G_math: Calculated from historical execution statistics
    - G_llm: Derived from LLM's self-assessed success prob and info gain
    - λ: Interpolation factor (adaptive based on precision)
    
    Intuition:
    - Low precision → trust LLM more (world model unreliable, use semantics)
    - High precision → trust math more (world model accurate, use statistics)
    
    Examples:
        >>> evaluator = HybridGEvaluator()
        >>> 
        >>> # LLM proposal with self-assessment
        >>> proposal = {
        ...     'policy': [tool_a, tool_b],
        ...     'llm_success_prob': 0.7,
        ...     'llm_info_gain': 0.4
        ... }
        >>> 
        >>> # Evaluate with hybrid approach
        >>> G = evaluator.evaluate_hybrid(
        ...     proposal, state, preferences, precision=0.5
        ... )
    """
    
    def __init__(
        self,
        lambda_fn: Optional[callable] = None,
        epistemic_weight: float = 1.0
    ):
        """
        Initialize hybrid evaluator.
        
        Args:
            lambda_fn: Function mapping precision → interpolation weight
                      Default: λ = 1 - precision (trust LLM when uncertain)
            epistemic_weight: Weight for epistemic value in G calculation
        """
        self.epistemic_weight = epistemic_weight
        
        # Default lambda function: inverse of precision
        # Low precision → high λ → trust LLM
        # High precision → low λ → trust math
        if lambda_fn is None:
            self.lambda_fn = lambda p: 1.0 - p
        else:
            self.lambda_fn = lambda_fn
    
    def evaluate_hybrid(
        self,
        proposal: Dict[str, Any],
        state: Dict[str, Any],
        preferences: Dict[str, float],
        precision: float,
        historical_stats: Optional[Dict[str, Dict]] = None
    ) -> float:
        """
        Evaluate policy using hybrid approach.
        
        Args:
            proposal: Policy proposal with 'policy', 'llm_success_prob', 'llm_info_gain'
            state: Current agent state
            preferences: Reward function
            precision: Current precision value
            historical_stats: Optional execution history
        
        Returns:
            Hybrid G value
        
        Examples:
            >>> G = evaluator.evaluate_hybrid(proposal, state, preferences, precision=0.3)
            >>> # Low precision → G weighted toward LLM's assessment
        """
        policy = proposal['policy']
        
        # Calculate mathematical G
        G_math = calculate_expected_free_energy(
            policy=policy,
            state=state,
            preferences=preferences,
            historical_stats=historical_stats,
            epistemic_weight=self.epistemic_weight
        )
        
        # Calculate LLM-derived G
        G_llm = self._calculate_llm_g(proposal, preferences)
        
        # Adaptive interpolation
        lambda_weight = self.lambda_fn(precision)
        
        # Hybrid G
        G_hybrid = (1 - lambda_weight) * G_math + lambda_weight * G_llm
        
        return G_hybrid
    
    def _calculate_llm_g(
        self,
        proposal: Dict[str, Any],
        preferences: Dict[str, float]
    ) -> float:
        """
        Calculate G from LLM's self-assessment.
        
        Uses the LLM's estimated success probability and information gain
        to compute an Expected Free Energy value.
        
        Args:
            proposal: Must contain 'llm_success_prob' and 'llm_info_gain'
            preferences: Reward function
        
        Returns:
            G value derived from LLM estimates
        """
        # Extract LLM assessments
        success_prob = proposal.get('llm_success_prob', 0.5)
        info_gain = proposal.get('llm_info_gain', 0.5)
        
        # Epistemic value ≈ info_gain (from LLM)
        epistemic = info_gain * self.epistemic_weight
        
        # Pragmatic value ≈ expected reward (from LLM success prob)
        success_reward = preferences.get('success', 0.0)
        error_penalty = preferences.get('error', 0.0)
        
        pragmatic = success_prob * success_reward + (1 - success_prob) * error_penalty
        
        # G = Epistemic - Pragmatic
        G_llm = epistemic - pragmatic
        
        return G_llm
    
    def evaluate_all(
        self,
        proposals: List[Dict[str, Any]],
        state: Dict[str, Any],
        preferences: Dict[str, float],
        precision: float,
        historical_stats: Optional[Dict[str, Dict]] = None
    ) -> List[PolicyEvaluation]:
        """
        Evaluate multiple proposals.
        
        Args:
            proposals: List of policy proposals
            state: Current state
            preferences: Reward function
            precision: Current precision
            historical_stats: Execution history
        
        Returns:
            List of PolicyEvaluation objects
        """
        evaluations = []
        
        for proposal in proposals:
            policy = proposal['policy']
            
            # Calculate hybrid G
            G_hybrid = self.evaluate_hybrid(
                proposal, state, preferences, precision, historical_stats
            )
            
            # Also calculate pure mathematical G for comparison
            G_math = calculate_expected_free_energy(
                policy, state, preferences, historical_stats
            )
            
            # Estimate success probability
            if 'llm_success_prob' in proposal:
                success_prob = proposal['llm_success_prob']
            else:
                success_prob = 0.5
            
            # Create evaluation
            evaluation = PolicyEvaluation(
                epistemic_value=proposal.get('llm_info_gain', 0.5),
                pragmatic_value=success_prob,
                total_G=G_hybrid,
                expected_success_prob=success_prob,
                components={
                    'G_hybrid': G_hybrid,
                    'G_math': G_math,
                    'G_llm': self._calculate_llm_g(proposal, preferences),
                    'lambda': self.lambda_fn(precision),
                    'strategy': proposal.get('strategy', 'unknown')
                }
            )
            
            evaluations.append(evaluation)
        
        return evaluations


def compare_math_vs_llm(
    proposal: Dict[str, Any],
    state: Dict[str, Any],
    preferences: Dict[str, float],
    historical_stats: Optional[Dict[str, Dict]] = None
) -> Dict[str, float]:
    """
    Compare mathematical vs LLM-based G calculation.
    
    Useful for debugging and understanding how the hybrid evaluator works.
    
    Args:
        proposal: Policy proposal with LLM assessments
        state: Current state
        preferences: Reward function
        historical_stats: Execution history
    
    Returns:
        Dict with 'G_math', 'G_llm', and 'difference'
    
    Examples:
        >>> comparison = compare_math_vs_llm(proposal, state, preferences)
        >>> print(f"Math G: {comparison['G_math']:.2f}")
        >>> print(f"LLM G: {comparison['G_llm']:.2f}")
        >>> print(f"Difference: {comparison['difference']:.2f}")
    """
    evaluator = HybridGEvaluator()
    
    policy = proposal['policy']
    
    # Mathematical G
    G_math = calculate_expected_free_energy(
        policy, state, preferences, historical_stats
    )
    
    # LLM G
    G_llm = evaluator._calculate_llm_g(proposal, preferences)
    
    return {
        'G_math': G_math,
        'G_llm': G_llm,
        'difference': abs(G_math - G_llm)
    }
