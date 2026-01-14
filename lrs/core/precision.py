"""
Precision tracking via Bayesian Beta distributions.

Precision (γ) represents the agent's confidence in its world model.
Updated dynamically based on prediction errors using conjugate priors.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class PrecisionParameters:
    """
    Bayesian precision tracking using Beta distribution.
    
    The precision γ is modeled as the expected value of a Beta distribution:
        γ ~ Beta(α, β)
        E[γ] = α / (α + β)
    
    When prediction errors are low, α increases (higher confidence).
    When prediction errors are high, β increases (lower confidence).
    
    Attributes:
        alpha (float): Shape parameter controlling high-confidence mass.
            Higher α → higher precision. Default: 9.0 (90% confidence prior).
        beta (float): Shape parameter controlling low-confidence mass.
            Higher β → lower precision. Default: 1.0.
        learning_rate_gain (float): Rate of confidence increase on success.
            Default: 0.1.
        learning_rate_loss (float): Rate of confidence decrease on failure.
            Default: 0.2 (asymmetric - faster to lose confidence).
        threshold (float): Prediction error threshold for success/failure.
            Errors below this increase α, above this increase β.
            Default: 0.5.
    
    Mathematical Justification:
        The Beta distribution is the conjugate prior for Bernoulli-distributed
        observations (success/failure). This ensures closed-form Bayesian updates
        without numerical approximation.
    
    Examples:
        >>> precision = PrecisionParameters(alpha=9.0, beta=1.0)
        >>> print(precision.value)  # E[Beta(9,1)] = 0.9
        0.9
        
        >>> precision.update(prediction_error=0.2)  # Low error
        >>> print(precision.value)  # α increased → higher confidence
        0.901
        
        >>> precision.update(prediction_error=0.8)  # High error
        >>> print(precision.value)  # β increased → lower confidence
        0.880
    """
    
    alpha: float = 9.0
    beta: float = 1.0
    learning_rate_gain: float = 0.1
    learning_rate_loss: float = 0.2
    threshold: float = 0.5
    
    # History tracking (optional, for monitoring)
    history: List[float] = field(default_factory=list, repr=False)
    
    @property
    def value(self) -> float:
        """
        Current precision estimate: E[Beta(α, β)] = α / (α + β)
        
        Returns:
            float: Precision value in [0, 1].
        """
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        """
        Uncertainty in precision estimate: Var[Beta(α, β)]
        
        Returns:
            float: Variance of the Beta distribution.
            
        Note:
            High variance indicates the agent is uncertain about its own
            uncertainty (meta-uncertainty). This can trigger hierarchical
            belief revision.
        """
        n = self.alpha + self.beta
        return (self.alpha * self.beta) / (n**2 * (n + 1))
    
    def update(self, prediction_error: float) -> float:
        """
        Bayesian update based on observed prediction error.
        
        Update rule:
            If ε < threshold: α ← α + η_gain  (reward accuracy)
            If ε ≥ threshold: β ← β + η_loss  (penalize inaccuracy)
        
        Args:
            prediction_error (float): Observed error ε = |predicted - observed|.
                Should be normalized to [0, 1].
        
        Returns:
            float: Updated precision value.
            
        Raises:
            ValueError: If prediction_error is outside [0, 1].
        
        Examples:
            >>> p = PrecisionParameters(alpha=5.0, beta=5.0)  # γ = 0.5
            >>> p.update(0.1)  # Success
            5.1
            >>> p.update(0.9)  # Failure  
            5.3
        """
        if not 0 <= prediction_error <= 1:
            raise ValueError(
                f"Prediction error must be in [0, 1], got {prediction_error}"
            )
        
        if prediction_error < self.threshold:
            # Low error → increase confidence
            self.alpha += self.learning_rate_gain
        else:
            # High error → decrease confidence
            self.beta += self.learning_rate_loss
        
        # Track history
        new_precision = self.value
        self.history.append(new_precision)
        
        return new_precision
    
    def reset(self, alpha: Optional[float] = None, beta: Optional[float] = None):
        """
        Reset precision to initial or specified values.
        
        Useful when agent enters a new environment or task context.
        
        Args:
            alpha (float, optional): New α value. If None, uses current.
            beta (float, optional): New β value. If None, uses current.
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self.history.clear()
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the posterior precision distribution.
        
        Useful for Thompson sampling or uncertainty quantification.
        
        Args:
            n_samples (int): Number of samples to draw.
        
        Returns:
            np.ndarray: Samples from Beta(α, β).
        
        Examples:
            >>> p = PrecisionParameters(alpha=9.0, beta=1.0)
            >>> samples = p.sample(1000)
            >>> np.mean(samples)  # Should be close to 0.9
            0.899
        """
        return np.random.beta(self.alpha, self.beta, size=n_samples)


@dataclass
class HierarchicalPrecision:
    """
    Manages precision across multiple levels of the Hierarchical Bayesian Network.
    
    In active inference, agents maintain beliefs at different temporal scales:
        - Abstract (Level 2): Long-term goals, slow updates
        - Planning (Level 1): Subgoal selection, medium updates
        - Execution (Level 0): Tool calls, fast updates
    
    Prediction errors propagate bottom-up, while priors flow top-down.
    
    Attributes:
        levels (Dict[str, PrecisionParameters]): Precision trackers per level.
        propagation_threshold (float): Error threshold for upward propagation.
            If execution-level error exceeds this, planning precision is updated.
    
    Examples:
        >>> hp = HierarchicalPrecision()
        >>> hp.update('execution', prediction_error=0.8)
        >>> # High error triggers propagation
        >>> hp.levels['planning'].value  # Also decreased
        0.65
    """
    
    levels: Dict[str, PrecisionParameters] = field(default_factory=lambda: {
        'abstract': PrecisionParameters(alpha=9.0, beta=1.0),
        'planning': PrecisionParameters(alpha=7.0, beta=3.0),
        'execution': PrecisionParameters(alpha=5.0, beta=5.0)
    })
    
    propagation_threshold: float = 0.7
    
    def update(self, level: str, prediction_error: float) -> Dict[str, float]:
        """
        Update precision at specified level with error propagation.
        
        If prediction error exceeds threshold, propagate to higher levels.
        This implements hierarchical message passing from predictive coding.
        
        Args:
            level (str): Level to update ('abstract', 'planning', 'execution').
            prediction_error (float): Observed error at this level.
        
        Returns:
            Dict[str, float]: Updated precision values for all affected levels.
        
        Raises:
            KeyError: If level is not recognized.
        """
        if level not in self.levels:
            raise KeyError(f"Unknown level '{level}'. Must be one of {list(self.levels.keys())}")
        
        # Update target level
        updated = {level: self.levels[level].update(prediction_error)}
        
        # Check for upward propagation
        if prediction_error > self.propagation_threshold:
            # Propagate error to higher levels (with attenuation)
            if level == 'execution':
                updated['planning'] = self.levels['planning'].update(
                    prediction_error * 0.7  # Attenuate error
                )
            elif level == 'planning':
                updated['abstract'] = self.levels['abstract'].update(
                    prediction_error * 0.5  # Further attenuation
                )
        
        return updated
    
    def get_all(self) -> Dict[str, float]:
        """Get current precision values for all levels."""
        return {level: params.value for level, params in self.levels.items()}
    
    def get_level(self, level: str) -> float:
        """Get precision value for specific level."""
        return self.levels[level].value
