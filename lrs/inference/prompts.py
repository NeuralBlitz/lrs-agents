"""
Meta-cognitive prompting for LRS-Agents.

Generates precision-adaptive prompts that guide LLMs to produce
diverse policy proposals appropriate to the agent's epistemic state.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class StrategyMode(Enum):
    """Strategic mode based on precision level"""
    EXPLOITATION = "exploit"  # High precision
    EXPLORATION = "explore"   # Low precision
    BALANCED = "balanced"     # Medium precision


@dataclass
class PromptContext:
    """
    Context for generating meta-cognitive prompts.
    
    Attributes:
        precision: Current precision value [0, 1]
        recent_errors: List of recent prediction errors
        available_tools: List of tool names
        goal: Current goal description
        state: Current agent state
        tool_history: Recent tool executions
    """
    precision: float
    recent_errors: List[float]
    available_tools: List[str]
    goal: str
    state: Dict[str, Any]
    tool_history: List[Dict[str, Any]]


class MetaCognitivePrompter:
    """
    Generates precision-adaptive prompts for LLM policy generation.
    
    The prompts adapt based on:
    1. Precision level (confidence in world model)
    2. Recent prediction errors (surprise events)
    3. Available tools
    4. Current goal
    
    Examples:
        >>> prompter = MetaCognitivePrompter()
        >>> 
        >>> context = PromptContext(
        ...     precision=0.3,  # Low precision
        ...     recent_errors=[0.9, 0.85, 0.7],
        ...     available_tools=["api_fetch", "cache_fetch"],
        ...     goal="Fetch user data",
        ...     state={},
        ...     tool_history=[]
        ... )
        >>> 
        >>> prompt = prompter.generate_prompt(context)
        >>> print("EXPLORATION MODE" in prompt)
        True
    """
    
    def __init__(
        self,
        high_precision_threshold: float = 0.7,
        low_precision_threshold: float = 0.4,
        high_error_threshold: float = 0.7
    ):
        """
        Initialize prompter.
        
        Args:
            high_precision_threshold: Threshold for exploitation mode
            low_precision_threshold: Threshold for exploration mode
            high_error_threshold: Threshold for "high surprise"
        """
        self.high_precision_threshold = high_precision_threshold
        self.low_precision_threshold = low_precision_threshold
        self.high_error_threshold = high_error_threshold
    
    def generate_prompt(self, context: PromptContext) -> str:
        """
        Generate precision-adaptive prompt.
        
        Args:
            context: Prompt context with precision, errors, tools, etc.
        
        Returns:
            Complete prompt string for LLM
        
        Examples:
            >>> prompt = prompter.generate_prompt(context)
            >>> # Prompt includes precision value, strategy guidance, tool list
        """
        # Determine strategy mode
        mode = self._determine_mode(context.precision)
        
        # Build prompt sections
        header = self._build_header()
        precision_info = self._build_precision_info(context.precision, mode)
        strategy_guidance = self._build_strategy_guidance(mode, context)
        error_analysis = self._build_error_analysis(context.recent_errors)
        tool_context = self._build_tool_context(context.available_tools)
        goal_description = self._build_goal_description(context.goal)
        output_format = self._build_output_format()
        diversity_requirements = self._build_diversity_requirements()
        calibration_instructions = self._build_calibration_instructions()
        
        # Combine all sections
        prompt = "\n\n".join([
            header,
            precision_info,
            strategy_guidance,
            error_analysis,
            tool_context,
            goal_description,
            output_format,
            diversity_requirements,
            calibration_instructions
        ])
        
        return prompt
    
    def _determine_mode(self, precision: float) -> StrategyMode:
        """Determine strategic mode from precision value"""
        if precision >= self.high_precision_threshold:
            return StrategyMode.EXPLOITATION
        elif precision <= self.low_precision_threshold:
            return StrategyMode.EXPLORATION
        else:
            return StrategyMode.BALANCED
    
    def _build_header(self) -> str:
        """Build prompt header"""
        return """You are a Bayesian policy generator for an Active Inference agent.

Your role is to PROPOSE diverse policy candidates, not to DECIDE which is best.
The agent will evaluate your proposals using Expected Free Energy (G).

Your proposals should span the exploration-exploitation spectrum based on
the agent's current precision (confidence in its world model)."""
    
    def _build_precision_info(self, precision: float, mode: StrategyMode) -> str:
        """Build precision information section"""
        confidence_level = "HIGH" if precision > 0.7 else "LOW" if precision < 0.4 else "MEDIUM"
        
        return f"""CURRENT PRECISION (γ): {precision:.3f} ({confidence_level})

This represents the agent's confidence that its world model is correct.
- High precision (>0.7): Agent is confident → Focus on exploitation
- Low precision (<0.4): Agent is uncertain → Focus on exploration
- Medium precision: Balance both strategies

CURRENT MODE: {mode.value.upper()}"""
    
    def _build_strategy_guidance(
        self,
        mode: StrategyMode,
        context: PromptContext
    ) -> str:
        """Build strategy-specific guidance"""
        if mode == StrategyMode.EXPLOITATION:
            return """STRATEGIC GUIDANCE: EXPLOITATION MODE

Your proposal strategy:
1. Prioritize reward - Focus on proven, high-success approaches
2. Leverage patterns - Use tools that have worked reliably before
3. Minimize risk - Avoid experimental or untested combinations
4. Optimize efficiency - Prefer shorter, well-understood policies

Generate proposals with:
- 70% exploitation (high success probability, low information gain)
- 30% exploration (maintain some diversity)"""
        
        elif mode == StrategyMode.EXPLORATION:
            return """STRATEGIC GUIDANCE: EXPLORATION MODE

The agent's world model is unreliable. Prioritize learning over reward.

Your proposal strategy:
1. Prioritize information - Focus on reducing uncertainty
2. Test assumptions - Include diagnostic actions that reveal environment state
3. Accept risk - Exploratory policies may have lower immediate success
4. Question patterns - Previous successful strategies may be outdated

Generate proposals with:
- 70% exploration (high information gain, lower certainty)
- 30% exploitation (maintain some reliable options)"""
        
        else:  # BALANCED
            return """STRATEGIC GUIDANCE: BALANCED MODE

The agent has moderate confidence. Balance exploration and exploitation.

Your proposal strategy:
1. Mix approaches - Combine proven tools with experimental ones
2. Hedge uncertainty - Include both safe and informative actions
3. Gradual adaptation - Test small variations on known patterns
4. Maintain optionality - Keep fallback plans available

Generate proposals with:
- 50% exploitation (reliable approaches)
- 50% exploration (learning opportunities)"""
    
    def _build_error_analysis(self, recent_errors: List[float]) -> str:
        """Build error analysis section"""
        if not recent_errors:
            return "RECENT ERRORS: None (no execution history yet)"
        
        avg_error = sum(recent_errors) / len(recent_errors)
        high_errors = [e for e in recent_errors if e > self.high_error_threshold]
        
        analysis = f"""RECENT PREDICTION ERRORS: {len(recent_errors)} recent executions
Average error: {avg_error:.3f}
High-surprise events: {len(high_errors)}"""
        
        if high_errors:
            analysis += f"""

⚠️  RECENT SURPRISES DETECTED
The agent has experienced {len(high_errors)} unexpected outcomes.
This suggests the environment may have changed or tools are behaving differently.

Consider:
- Alternative approaches to recent failures
- Diagnostic actions to understand what changed
- Conservative strategies that fail gracefully"""
        
        return analysis
    
    def _build_tool_context(self, available_tools: List[str]) -> str:
        """Build available tools section"""
        tools_str = "\n".join(f"  - {tool}" for tool in available_tools)
        
        return f"""AVAILABLE TOOLS ({len(available_tools)} tools):
{tools_str}

You must only propose policies using these exact tool names.
Policies can use the same tool multiple times if needed."""
    
    def _build_goal_description(self, goal: str) -> str:
        """Build goal description section"""
        return f"""GOAL: {goal}

Your proposals should work toward this goal while respecting the
current precision level and strategic mode."""
    
    def _build_output_format(self) -> str:
        """Build output format specification"""
        return """OUTPUT FORMAT

Generate 3-7 policy proposals in JSON format:

{
  "proposals": [
    {
      "policy_id": 1,
      "tools": ["tool_name_1", "tool_name_2"],
      "estimated_success_prob": 0.8,
      "expected_information_gain": 0.3,
      "strategy": "exploit|explore|balanced",
      "rationale": "Brief explanation of why this policy makes sense",
      "failure_modes": ["Potential failure scenario 1", "Scenario 2"]
    },
    {
      "policy_id": 2,
      ...
    }
  ],
  "current_uncertainty": 0.6,
  "known_unknowns": ["What we know we don't know"]
}

FIELD DESCRIPTIONS:
- policy_id: Unique integer ID (1, 2, 3, ...)
- tools: List of tool names in execution order
- estimated_success_prob: Your estimate of P(success) in [0, 1]
- expected_information_gain: How much we'd learn in [0, 1]
- strategy: "exploit", "explore", or "balanced"
- rationale: 1-2 sentence explanation
- failure_modes: List of ways this could fail"""
    
    def _build_diversity_requirements(self) -> str:
        """Build diversity requirements"""
        return """DIVERSITY REQUIREMENTS (CRITICAL)

Your proposal set MUST include:
1. At least 1 exploitative policy (estimated_success_prob > 0.7, low info_gain)
2. At least 1 exploratory policy (high info_gain, lower success_prob)
3. At least 1 balanced policy

Do NOT generate 5 nearly-identical proposals. The agent needs genuine alternatives
spanning different risk-reward tradeoffs.

VARIETY CHECKLIST:
☐ Different tool combinations
☐ Different policy lengths (1-5 tools)
☐ Different risk levels
☐ Different information-gathering strategies"""
    
    def _build_calibration_instructions(self) -> str:
        """Build calibration instructions"""
        return """CALIBRATION INSTRUCTIONS

⚠️  Avoid overconfidence: If you're uncertain, reflect that in lower success probabilities.

✓ Be honest: The agent's mathematical evaluation will assess your proposals objectively.
  Don't inflate success probabilities to make proposals look better.

CALIBRATION TEST:
If ALL your proposals have estimated_success_prob > 0.8, you're likely overconfident.
Include riskier, more exploratory options with honest uncertainty estimates.

The agent will COMBINE your generative creativity with rigorous mathematical evaluation.
Your job is diverse proposal generation, not final decision-making."""


def build_simple_prompt(
    goal: str,
    tools: List[str],
    precision: float,
    num_proposals: int = 5
) -> str:
    """
    Build a simple prompt without full context.
    
    Convenience function for quick prompting.
    
    Args:
        goal: Task goal
        tools: Available tool names
        precision: Current precision value
        num_proposals: Number of proposals to generate
    
    Returns:
        Prompt string
    
    Examples:
        >>> prompt = build_simple_prompt(
        ...     goal="Fetch data",
        ...     tools=["api", "cache"],
        ...     precision=0.5
        ... )
    """
    context = PromptContext(
        precision=precision,
        recent_errors=[],
        available_tools=tools,
        goal=goal,
        state={},
        tool_history=[]
    )
    
    prompter = MetaCognitivePrompter()
    return prompter.generate_prompt(context)
