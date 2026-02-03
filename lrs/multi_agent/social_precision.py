"""
Social precision tracking for multi-agent systems.

Tracks confidence in other agents' models via prediction errors on their actions.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

from lrs.core.precision import PrecisionParameters


@dataclass
class SocialPrecisionParameters(PrecisionParameters):
    """
    Precision parameters for social beliefs.

    Extends environmental precision with social-specific defaults.
    Social precision tends to have:
    - Slower gain (agents are more complex than tools)
    - Faster loss (agents can change behavior unpredictably)
    """

    def __init__(
        self,
        alpha: float = 5.0,
        beta: float = 5.0,
        gain_learning_rate: float = 0.05,  # Slower than environmental
        loss_learning_rate: float = 0.25,  # Faster than environmental
        adaptation_threshold: float = 0.5,
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            gain_learning_rate=gain_learning_rate,
            loss_learning_rate=loss_learning_rate,
            adaptation_threshold=adaptation_threshold,
        )


class SocialPrecisionTracker:
    """
    Track precision (confidence/trust) in other agents.

    Each agent maintains separate precision values for every other agent,
    representing how well they can predict that agent's behavior.

    High social precision = "I understand what this agent will do"
    Low social precision = "This agent is unpredictable to me"

    Examples:
        >>> tracker = SocialPrecisionTracker(agent_id="agent_a")
        >>>
        >>> # Agent A observes Agent B
        >>> tracker.register_agent("agent_b")
        >>>
        >>> # Agent B acts as predicted
        >>> tracker.update_social_precision(
        ...     other_agent_id="agent_b",
        ...     predicted_action="fetch_data",
        ...     observed_action="fetch_data"
        ... )
        >>> print(tracker.get_social_precision("agent_b"))  # Increased
        0.52
        >>>
        >>> # Agent B acts unexpectedly
        >>> tracker.update_social_precision(
        ...     other_agent_id="agent_b",
        ...     predicted_action="fetch_data",
        ...     observed_action="use_cache"  # Surprise!
        ... )
        >>> print(tracker.get_social_precision("agent_b"))  # Decreased
        0.44
    """

    def __init__(self, agent_id: str):
        """
        Initialize social precision tracker.

        Args:
            agent_id: ID of the agent doing the tracking
        """
        self.agent_id = agent_id
        self.social_precision: Dict[str, SocialPrecisionParameters] = {}
        self.action_history: Dict[str, List[Dict[str, Any]]] = {}

    def register_agent(self, other_agent_id: str):
        """
        Register another agent for tracking.

        Args:
            other_agent_id: ID of agent to track
        """
        if other_agent_id not in self.social_precision:
            self.social_precision[other_agent_id] = SocialPrecisionParameters()
            self.action_history[other_agent_id] = []

    def update_social_precision(
        self, other_agent_id: str, predicted_action: Any, observed_action: Any
    ) -> float:
        """
        Update social precision based on action prediction.

        Args:
            other_agent_id: ID of observed agent
            predicted_action: What we predicted they would do
            observed_action: What they actually did

        Returns:
            Updated social precision value

        Examples:
            >>> # Correct prediction
            >>> new_prec = tracker.update_social_precision(
            ...     "agent_b", "fetch", "fetch"
            ... )
            >>> # new_prec increased
            >>>
            >>> # Incorrect prediction
            >>> new_prec = tracker.update_social_precision(
            ...     "agent_b", "fetch", "cache"
            ... )
            >>> # new_prec decreased
        """
        if other_agent_id not in self.social_precision:
            self.register_agent(other_agent_id)

        # Calculate social prediction error
        error = self._calculate_social_prediction_error(predicted_action, observed_action)

        # Update precision
        precision_params = self.social_precision[other_agent_id]
        new_precision = precision_params.update(error)

        # Record in history
        self.action_history[other_agent_id].append(
            {
                "predicted": predicted_action,
                "observed": observed_action,
                "error": error,
                "precision": new_precision,
            }
        )

        return new_precision

    def _calculate_social_prediction_error(self, predicted: Any, observed: Any) -> float:
        """
        Calculate prediction error for social action.

        Simple version: exact match = 0.0, mismatch = 1.0
        Could be extended to consider action similarity.

        Args:
            predicted: Predicted action
            observed: Observed action

        Returns:
            Social prediction error in [0, 1]
        """
        # Exact match
        if predicted == observed:
            return 0.0

        # Could add fuzzy matching for similar actions
        # For now, any mismatch is full surprise
        return 1.0

    def get_social_precision(self, other_agent_id: str) -> float:
        """
        Get current social precision for an agent.

        Args:
            other_agent_id: Agent to query

        Returns:
            Social precision value [0, 1]
        """
        if other_agent_id not in self.social_precision:
            return 0.5  # Neutral prior

        return self.social_precision[other_agent_id].value

    def get_all_social_precisions(self) -> Dict[str, float]:
        """
        Get social precision for all tracked agents.

        Returns:
            Dict mapping agent IDs to precision values
        """
        return {agent_id: params.value for agent_id, params in self.social_precision.items()}

    def should_communicate(
        self, other_agent_id: str, threshold: float = 0.5, env_precision: float = 0.5
    ) -> bool:
        """
        Decide whether to communicate with another agent.

        Communication is valuable when:
        1. Social precision is low (uncertain about other agent)
        2. Environmental precision is high (so problem is social, not environmental)

        Args:
            other_agent_id: Target agent
            threshold: Social precision threshold for communication
            env_precision: Current environmental precision

        Returns:
            True if should communicate

        Examples:
            >>> # Low social precision, high env precision → communicate
            >>> should_comm = tracker.should_communicate(
            ...     "agent_b", threshold=0.5, env_precision=0.8
            ... )
            >>> print(should_comm)
            True
            >>>
            >>> # High social precision → no need to communicate
            >>> tracker.social_precision["agent_b"].alpha = 10.0
            >>> should_comm = tracker.should_communicate(
            ...     "agent_b", threshold=0.5, env_precision=0.8
            ... )
            >>> print(should_comm)
            False
        """
        social_prec = self.get_social_precision(other_agent_id)

        # Communicate when social precision is low AND env precision is high
        # (If env precision is also low, problem might not be social)
        return social_prec < threshold and env_precision > 0.6

    def get_action_history(self, other_agent_id: str) -> List[Dict[str, Any]]:
        """
        Get prediction history for an agent.

        Args:
            other_agent_id: Agent to query

        Returns:
            List of prediction records
        """
        return self.action_history.get(other_agent_id, [])

    def predict_action(self, other_agent_id: str, context: Dict[str, Any]) -> Any:
        """
        Predict what another agent will do (simple version).

        Uses recent action patterns to predict.

        Args:
            other_agent_id: Agent to predict
            context: Current context

        Returns:
            Predicted action
        """
        history = self.get_action_history(other_agent_id)

        if not history:
            return None  # No data to predict

        # Simple: return most recent action
        # Could be extended with pattern matching
        return history[-1]["observed"]


class RecursiveBeliefState:
    """
    Recursive theory-of-mind: model other agents' beliefs about you.

    Tracks:
    - My precision
    - My belief about Agent B's precision
    - My belief about Agent B's belief about my precision

    This enables sophisticated coordination where agents reason about
    each other's models.

    Examples:
        >>> beliefs = RecursiveBeliefState(agent_id="agent_a")
        >>>
        >>> # I think Agent B's precision is 0.7
        >>> beliefs.set_belief_about_other("agent_b", 0.7)
        >>>
        >>> # I think Agent B thinks my precision is 0.8
        >>> beliefs.set_belief_about_other_belief("agent_b", 0.8)
        >>>
        >>> # Should I tell Agent B I'm uncertain?
        >>> should_share = beliefs.should_share_uncertainty("agent_b")
    """

    def __init__(self, agent_id: str):
        """Initialize recursive belief state"""
        self.agent_id = agent_id
        self.my_precision: float = 0.5

        # What I think about other agents
        self.belief_about_other: Dict[str, float] = {}

        # What I think other agents think about me
        self.belief_about_other_belief: Dict[str, float] = {}

    def set_my_precision(self, precision: float):
        """Set my actual precision"""
        self.my_precision = precision

    def set_belief_about_other(self, other_agent_id: str, precision: float):
        """Set belief about another agent's precision"""
        self.belief_about_other[other_agent_id] = precision

    def set_belief_about_other_belief(self, other_agent_id: str, precision: float):
        """Set belief about what another agent thinks my precision is"""
        self.belief_about_other_belief[other_agent_id] = precision

    def should_share_uncertainty(self, other_agent_id: str, threshold: float = 0.3) -> bool:
        """
        Decide if should communicate uncertainty to another agent.

        Share when: I'm uncertain, but other agent thinks I'm confident

        Args:
            other_agent_id: Target agent
            threshold: Precision threshold for "uncertain"

        Returns:
            True if should share uncertainty

        Examples:
            >>> beliefs = RecursiveBeliefState("agent_a")
            >>> beliefs.set_my_precision(0.3)  # I'm uncertain
            >>> beliefs.set_belief_about_other_belief("agent_b", 0.8)  # B thinks I'm confident
            >>>
            >>> should_share = beliefs.should_share_uncertainty("agent_b")
            >>> print(should_share)
            True
        """
        my_actual = self.my_precision
        other_thinks = self.belief_about_other_belief.get(other_agent_id, 0.5)

        # Share if: I'm uncertain but other thinks I'm confident
        return my_actual < threshold and other_thinks > 0.7

    def should_seek_help(
        self, other_agent_id: str, my_threshold: float = 0.4, other_threshold: float = 0.6
    ) -> bool:
        """
        Decide if should ask another agent for help.

        Seek help when: I'm uncertain, and other agent is confident

        Args:
            other_agent_id: Target agent
            my_threshold: My precision threshold
            other_threshold: Required precision of helper

        Returns:
            True if should seek help
        """
        my_actual = self.my_precision
        other_precision = self.belief_about_other.get(other_agent_id, 0.5)

        return my_actual < my_threshold and other_precision > other_threshold
