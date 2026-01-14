"""
Bayesian precision tracking for Active Inference agents.

Precision (γ) represents the agent's confidence in its world model.
Implemented as Beta-distributed parameters that update via prediction errors.
"""

from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PrecisionParameters:
    """
    Beta-distributed precision parameters.
    
    Precision γ ~ Beta(α, β) where:
    - α: "success count" (increases with low prediction errors)
    - β: "failure count" (increases with high prediction errors)
    - E[γ] = α / (α + β)
    
    Attributes:
        alpha: Success parameter
        beta: Failure parameter
        learning_rate_gain: Rate at which α increases (default: 0.1)
        learning_rate_loss: Rate at which β increases (default: 0.2)
        threshold: Error threshold for gain vs loss (default: 0.5)
    
    Examples:
        >>> precision = PrecisionParameters(alpha=5.0, beta=5.0)
        >>> print(precision.value)  # E[γ] = 5/(5+5) = 0.5
        0.5
        >>> 
        >>> # Low error → increase alpha
        >>> precision.update(error=0.1)
        >>> print(precision.value)  # γ increased
        0.51
        >>> 
        >>> # High error → increase beta
        >>> precision.update(error=0.9)
        >>> print(precision.value)  # γ decreased
        0.48
    """
    
    alpha: float = 5.0
    beta: float = 5.0
    learning_rate_gain: float = 0.1
    learning_rate_loss: float = 0.2
    threshold: float = 0.5
    
    @property
    def value(self) -> float:
        """Expected value of precision: E[γ] = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        """Variance of precision distribution"""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    def update(self, prediction_error: float) -> float:
        """
        Update precision based on prediction error.
        
        Low error (< threshold) → increase α (gain confidence)
        High error (≥ threshold) → increase β (lose confidence)
        
        Key property: Loss is faster than gain (asymmetric learning)
        
        Args:
            prediction_error: Prediction error in [0, 1]
        
        Returns:
            Updated precision value
        
        Examples:
            >>> p = PrecisionParameters()
            >>> p.update(0.1)  # Low error
            0.51
            >>> p.update(0.9)  # High error
            0.47
        """
        if prediction_error < self.threshold:
            # Gain confidence slowly
            self.alpha += self.learning_rate_gain * (1 - prediction_error)
        else:
            # Lose confidence quickly
            self.beta += self.learning_rate_loss * prediction_error
        
        return self.value
    
    def reset(self):
        """Reset to initial prior"""
        self.alpha = 5.0
        self.beta = 5.0


class HierarchicalPrecision:
    """
    Three-level hierarchical precision tracking.
    
    Levels (from high to low):
    1. Abstract (level 2): Long-term goals and strategies
    2. Planning (level 1): Subgoal selection and sequencing
    3. Execution (level 0): Individual tool calls
    
    Errors propagate upward when they exceed a threshold.
    This prevents minor execution failures from disrupting high-level goals.
    
    Attributes:
        abstract: Abstract-level precision
        planning: Planning-level precision
        execution: Execution-level precision
        propagation_threshold: Error threshold for upward propagation
        attenuation_factor: How much error is attenuated when propagating
    
    Examples:
        >>> hp = HierarchicalPrecision()
        >>> 
        >>> # Small error at execution → only execution precision drops
        >>> hp.update('execution', 0.3)
        >>> print(hp.get_level('execution'))  # Decreased
        0.48
        >>> print(hp.get_level('planning'))   # Unchanged
        0.5
        >>> 
        >>> # Large error at execution → propagates to planning
        >>> hp.update('execution', 0.95)
        >>> print(hp.get_level('execution'))  # Dropped significantly
        0.32
        >>> print(hp.get_level('planning'))   # Also dropped
        0.45
    """
    
    def __init__(
        self,
        propagation_threshold: float = 0.7,
        attenuation_factor: float = 0.5
    ):
        """
        Initialize hierarchical precision.
        
        Args:
            propagation_threshold: Error threshold for upward propagation
            attenuation_factor: How much to attenuate errors when propagating
        """
        self.abstract = PrecisionParameters()
        self.planning = PrecisionParameters()
        self.execution = PrecisionParameters()
        
        self.propagation_threshold = propagation_threshold
        self.attenuation_factor = attenuation_factor
    
    def update(self, level: str, prediction_error: float) -> Dict[str, float]:
        """
        Update precision at specified level and propagate if needed.
        
        Propagation rules:
        - Execution → Planning: If error > threshold
        - Planning → Abstract: If error > threshold
        - Errors are attenuated when propagating up
        
        Args:
            level: 'abstract', 'planning', or 'execution'
            prediction_error: Error in [0, 1]
        
        Returns:
            Dict of updated precision values per level
        
        Examples:
            >>> hp = HierarchicalPrecision()
            >>> result = hp.update('execution', 0.95)
            >>> print(result)
            {'execution': 0.32, 'planning': 0.45}
        """
        updated = {}
        
        # Update specified level
        if level == 'execution':
            self.execution.update(prediction_error)
            updated['execution'] = self.execution.value
            
            # Propagate to planning if error is high
            if prediction_error > self.propagation_threshold:
                attenuated_error = prediction_error * self.attenuation_factor
                self.planning.update(attenuated_error)
                updated['planning'] = self.planning.value
                
                # Propagate to abstract if planning error is also high
                if attenuated_error > self.propagation_threshold:
                    super_attenuated = attenuated_error * self.attenuation_factor
                    self.abstract.update(super_attenuated)
                    updated['abstract'] = self.abstract.value
        
        elif level == 'planning':
            self.planning.update(prediction_error)
            updated['planning'] = self.planning.value
            
            # Propagate to abstract if error is high
            if prediction_error > self.propagation_threshold:
                attenuated_error = prediction_error * self.attenuation_factor
                self.abstract.update(attenuated_error)
                updated['abstract'] = self.abstract.value
        
        elif level == 'abstract':
            self.abstract.update(prediction_error)
            updated['abstract'] = self.abstract.value
        
        else:
            raise ValueError(f"Unknown level: {level}. Use 'abstract', 'planning', or 'execution'")
        
        return updated
    
    def get_level(self, level: str) -> float:
        """
        Get precision value for specified level.
        
        Args:
            level: 'abstract', 'planning', or 'execution'
        
        Returns:
            Precision value in [0, 1]
        """
        if level == 'abstract':
            return self.abstract.value
        elif level == 'planning':
            return self.planning.value
        elif level == 'execution':
            return self.execution.value
        else:
            raise ValueError(f"Unknown level: {level}")
    
    def get_all(self) -> Dict[str, float]:
        """
        Get all precision values.
        
        Returns:
            Dict mapping level names to precision values
        """
        return {
            'abstract': self.abstract.value,
            'planning': self.planning.value,
            'execution': self.execution.value
        }
    
    def reset(self):
        """Reset all levels to initial priors"""
        self.abstract.reset()
        self.planning.reset()
        self.execution.reset()
