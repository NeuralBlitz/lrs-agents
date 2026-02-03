"""Precision tracking for Active Inference agents."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PrecisionParameters:
    """
    Precision parameters using Beta distribution.

    Precision γ = α/(α+β) represents confidence in predictions.

    Args:
        alpha: Success parameter (default: 1.0)
        beta: Failure parameter (default: 1.0)
        gain_learning_rate: Learning rate for successes (default: 0.1)
        loss_learning_rate: Learning rate for failures (default: 0.2)
        adaptation_threshold: Threshold below which adaptation triggers (default: 0.4)

    Example:
        >>> precision = PrecisionParameters()
        >>> precision.value  # 0.5 (maximum uncertainty)
        >>> precision.update(prediction_error=0.1)  # Success
        >>> precision.value  # ~0.52 (slight increase)
    """

    alpha: float = 1.0
    beta: float = 1.0
    gain_learning_rate: float = 0.1
    loss_learning_rate: float = 0.2
    adaptation_threshold: float = 0.4

    @property
    def value(self) -> float:
        """
        Get current precision value γ = α/(α+β).

        Returns:
            Precision in [0,1]
        """
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """
        Get variance of Beta distribution.

        Returns:
            Variance of precision estimate
        """
        a = self.alpha
        b = self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update(self, prediction_error: float) -> None:
        """
        Update precision based on prediction error.

        Uses asymmetric learning rates:
        - Low error (success) → small increase in α
        - High error (failure) → larger increase in β

        Args:
            prediction_error: Prediction error δ ∈ [0,1]

        Example:
            >>> precision = PrecisionParameters()
            >>> precision.update(0.1)  # Success
            >>> precision.update(0.9)  # Failure
        """
        # Inverse error = success signal
        inverse_error = 1.0 - prediction_error

        # Asymmetric updates
        self.alpha += self.gain_learning_rate * inverse_error
        self.beta += self.loss_learning_rate * prediction_error

    def should_adapt(self) -> bool:
        """
        Check if precision is below adaptation threshold.

        Returns:
            True if adaptation should be triggered
        """
        return self.value < self.adaptation_threshold

    def reset(self) -> None:
        """Reset to initial uniform prior."""
        self.alpha = 1.0
        self.beta = 1.0

    def get_all(self) -> Dict[str, float]:
        """
        Get all precision statistics.

        Returns:
            Dictionary with value, alpha, beta, variance
        """
        return {
            "value": self.value,
            "alpha": self.alpha,
            "beta": self.beta,
            "variance": self.variance,
        }

    # Backward compatibility properties
    @property
    def learning_rate_gain(self) -> float:
        """Alias for gain_learning_rate."""
        return self.gain_learning_rate

    @property
    def learning_rate_loss(self) -> float:
        """Alias for loss_learning_rate."""
        return self.loss_learning_rate

    @property
    def threshold(self) -> float:
        """Alias for adaptation_threshold."""
        return self.adaptation_threshold

    def get_all_values(self) -> Dict[str, float]:
        """Alias for get_all()."""
        return self.get_all()


@dataclass
class HierarchicalPrecision:
    """
    Hierarchical precision tracking across abstraction levels.

    Precision is maintained at three levels:
    - Abstract: Long-term goals and strategies
    - Planning: Policy sequences
    - Execution: Individual tool calls

    Errors propagate upward with attenuation.

    Example:
        >>> hp = HierarchicalPrecision()
        >>> hp.update('execution', 0.9)  # High error
        >>> hp.execution  # Decreased
        >>> hp.planning   # Also decreased (propagation)
    """

    _abstract: PrecisionParameters = field(default_factory=PrecisionParameters)
    _planning: PrecisionParameters = field(default_factory=PrecisionParameters)
    _execution: PrecisionParameters = field(default_factory=PrecisionParameters)

    propagation_threshold: float = 0.7
    attenuation_factor: float = 0.5

    # Properties that return float values (for convenience)
    @property
    def abstract(self) -> float:
        """Get abstract level precision value."""
        return self._abstract.value

    @property
    def planning(self) -> float:
        """Get planning level precision value."""
        return self._planning.value

    @property
    def execution(self) -> float:
        """Get execution level precision value."""
        return self._execution.value

    def get_level(self, level: str) -> PrecisionParameters:
        """
        Get PrecisionParameters object for a specific level.

        Args:
            level: One of 'abstract', 'planning', or 'execution'

        Returns:
            PrecisionParameters object for that level

        Example:
            >>> hp = HierarchicalPrecision()
            >>> exec_params = hp.get_level('execution')
            >>> exec_params.value  # 0.5
        """
        if level == "abstract":
            return self._abstract
        elif level == "planning":
            return self._planning
        elif level == "execution":
            return self._execution
        else:
            raise ValueError(
                f"Invalid level: {level}. Must be 'abstract', 'planning', or 'execution'"
            )

    def update(self, level: str, prediction_error: float) -> None:
        """
        Update precision at a specific level with upward propagation.

        High prediction errors (>0.7) propagate upward with attenuation.

        Args:
            level: Level to update ('abstract', 'planning', 'execution')
            prediction_error: Prediction error δ ∈ [0,1]

        Example:
            >>> hp = HierarchicalPrecision()
            >>> hp.update('execution', 0.95)  # High error
            >>> # Execution precision drops AND planning is affected
        """
        # Update the specified level
        params = self.get_level(level)
        params.update(prediction_error)

        # Propagate upward if error is high
        if prediction_error > self.propagation_threshold:
            attenuated_error = prediction_error * self.attenuation_factor

            if level == "execution":
                self._planning.update(attenuated_error)
            elif level == "planning":
                self._abstract.update(attenuated_error)

    def get_all_values(self) -> Dict[str, float]:
        """
        Get all precision values as a dictionary.

        Returns:
            Dictionary with abstract, planning, execution values
        """
        return {
            "abstract": self._abstract.value,
            "planning": self._planning.value,
            "execution": self._execution.value,
        }

    def get_all(self) -> Dict[str, float]:
        """Alias for get_all_values()."""
        return self.get_all_values()

    def reset(self) -> None:
        """Reset all levels to initial values."""
        self._abstract.reset()
        self._planning.reset()
        self._execution.reset()

    def should_adapt(self, level: str = "execution") -> bool:
        """
        Check if adaptation is needed at specified level.

        Args:
            level: Level to check (default: 'execution')

        Returns:
            True if adaptation should be triggered
        """
        return self.get_level(level).should_adapt()


def beta_mean(alpha: float, beta: float) -> float:
    """
    Calculate mean of Beta distribution.

    Args:
        alpha: Alpha parameter
        beta: Beta parameter

    Returns:
        Mean = α/(α+β)
    """
    return alpha / (alpha + beta)


def beta_variance(alpha: float, beta: float) -> float:
    """
    Calculate variance of Beta distribution.

    Args:
        alpha: Alpha parameter
        beta: Beta parameter

    Returns:
        Variance
    """
    a_plus_b = alpha + beta
    return (alpha * beta) / (a_plus_b**2 * (a_plus_b + 1))
