"""LLM-based policy generation for Active Inference."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator  # Add field_validator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from lrs.core.lens import ToolLens
from lrs.core.registry import ToolRegistry
from lrs.core.precision import PrecisionParameters
from lrs.inference.prompts import MetaCognitivePrompter, StrategyMode


class PolicyProposal(BaseModel):
    """Single policy proposal from LLM."""
    
    tool_sequence: List[str] = Field(
        ...,
        description="Sequence of tool names to execute"
    )
    reasoning: str = Field(
        ...,
        description="Why this policy is good given current precision"
    )
    estimated_success_prob: float = Field(
        ...,
        description="Estimated P(success)",
        ge=0.0,
        le=1.0
    )
    estimated_info_gain: float = Field(
        ...,
        description="Expected information gain",
        ge=0.0
    )
    strategy: str = Field(
        ...,
        description="exploitation, exploration, or balanced"
    )
    failure_modes: List[str] = Field(
        default_factory=list,
        description="Known potential failure modes"
    )
    
    @field_validator('strategy')  # Changed from @validator
    @classmethod
    def validate_strategy(cls, v):
        """Validate strategy is one of the allowed modes."""
        valid = ['exploitation', 'exploration', 'balanced']
        if v.lower() not in valid:
            raise ValueError(f"Strategy must be one of {valid}")
        return v.lower()


class PolicyProposalSet(BaseModel):
    """Set of policy proposals with metadata."""
    
    proposals: List[PolicyProposal] = Field(
        ...,
        description="3-7 diverse policy proposals",
        min_length=3,  # Pydantic V2
        max_length=7   # Pydantic V2
    )
    current_uncertainty: float = Field(
        ..., 
        description="Current epistemic uncertainty (1 - precision)",
        ge=0.0,
        le=1.0
    )
    known_unknowns: List[str] = Field(
        default_factory=list,
        description="Known gaps in knowledge"
    )


# ... rest of file unchanged

    )



class LLMPolicyGenerator:
    """
    Generate policy proposals using an LLM.
    
    The LLM acts as a variational proposal mechanism:
    1. Receives precision-adaptive prompt
    2. Generates 3-7 diverse policy proposals
    3. Each proposal includes self-assessment of success prob and info gain
    
    The mathematical components (G calculation, precision-weighted selection)
    then evaluate and select from these proposals.
    
    Examples:
        >>> from langchain_anthropic import ChatAnthropic
        >>> 
        >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        >>> registry = ToolRegistry()
        >>> # ... register tools ...
        >>> 
        >>> generator = LLMPolicyGenerator(llm, registry)
        >>> 
        >>> proposals = generator.generate_proposals(
        ...     state={'goal': 'Fetch data'},
        ...     precision=0.5
        ... )
        >>> 
        >>> for p in proposals:
        ...     print(f"Policy {p['policy_id']}: {p['strategy']}")
    """
    
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        prompter: Optional[MetaCognitivePrompter] = None,
        temperature_fn: Optional[Callable[[float], float]] = None,
        base_temperature: float = 0.7
    ):
        """
        Initialize LLM policy generator.
        
        Args:
            llm: Language model (LangChain-compatible)
            registry: Tool registry
            prompter: Optional custom prompter (default: MetaCognitivePrompter)
            temperature_fn: Optional function mapping precision → temperature
            base_temperature: Base temperature value
        """
        self.llm = llm
        self.registry = registry
        self.prompter = prompter or MetaCognitivePrompter()
        self.base_temperature = base_temperature
        
        # Default temperature function: inverse relationship with precision
        if temperature_fn is None:
            self.temperature_fn = lambda p: base_temperature * (1.0 / (p + 0.1))
        else:
            self.temperature_fn = temperature_fn
    
    def generate_proposals(
        self,
        state: Dict[str, Any],
        precision: float,
        num_proposals: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate policy proposals.
        
        Args:
            state: Current agent state
            precision: Current precision value
            num_proposals: Target number of proposals (3-7)
        
        Returns:
            List of validated proposals with tool sequences
        
        Examples:
            >>> proposals = generator.generate_proposals(
            ...     state={'goal': 'Fetch user data'},
            ...     precision=0.3
            ... )
            >>> 
            >>> # Low precision → exploratory proposals
            >>> print([p['strategy'] for p in proposals])
            ['explore', 'explore', 'balanced', 'explore', 'exploit']
        """
        # Build prompt context
        context = self._build_context(state, precision)
        
        # Generate prompt
        prompt = self.prompter.generate_prompt(context)
        
        # Adapt temperature based on precision
        temperature = self._adapt_temperature(precision)
        
        # Call LLM with structured output
        try:
            response = self._call_llm(prompt, temperature)
            
            # Parse and validate
            proposal_set = self._parse_response(response)
            
            # Convert to executable policies
            validated = self._validate_and_convert(proposal_set.proposals)
            
            return validated
        
        except Exception as e:
            # Fallback to empty proposals
            print(f"Warning: LLM proposal generation failed: {e}")
            return []
    
    def _build_context(
        self,
        state: Dict[str, Any],
        precision: float
    ) -> PromptContext:
        """Build prompt context from state"""
        # Extract recent errors from tool history
        tool_history = state.get('tool_history', [])
        recent_errors = [
            entry.get('prediction_error', 0.5)
            for entry in tool_history[-5:]  # Last 5 executions
        ]
        
        # Get available tools
        available_tools = self.registry.list_tools()
        
        # Extract goal
        goal = state.get('belief_state', {}).get('goal', 'Unknown goal')
        if not isinstance(goal, str):
            goal = str(goal)
        
        return PromptContext(
            precision=precision,
            recent_errors=recent_errors,
            available_tools=available_tools,
            goal=goal,
            state=state,
            tool_history=tool_history
        )
    
    def _adapt_temperature(self, precision: float) -> float:
        """
        Adapt LLM temperature based on precision.
        
        Low precision → high temperature → diverse exploration
        High precision → low temperature → focused exploitation
        
        Args:
            precision: Precision value in [0, 1]
        
        Returns:
            Temperature value (typically in [0, 2])
        """
        temp = self.temperature_fn(precision)
        
        # Clamp to reasonable range
        return max(0.1, min(2.0, temp))
    
    def _call_llm(self, prompt: str, temperature: float) -> str:
        """
        Call LLM with prompt.
        
        Handles different LLM interfaces (LangChain, OpenAI, etc.)
        
        Args:
            prompt: Prompt string
            temperature: Temperature value
        
        Returns:
            LLM response text
        """
        # Try LangChain interface first
        if hasattr(self.llm, 'invoke'):
            from langchain_core.messages import HumanMessage
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages, temperature=temperature)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        
        # Try OpenAI interface
        elif hasattr(self.llm, 'chat') and hasattr(self.llm.chat, 'completions'):
            response = self.llm.chat.completions.create(
                model=getattr(self.llm, 'model', 'gpt-4'),
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        
        # Fallback: assume callable
        else:
            return self.llm(prompt, temperature=temperature)
    
    def _parse_response(self, response: str) -> PolicyProposalSet:
        """
        Parse LLM response into structured proposals.
        
        Args:
            response: JSON string from LLM
        
        Returns:
            Validated PolicyProposalSet
        
        Raises:
            ValueError: If response is invalid
        """
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}")
        
        # Validate with Pydantic
        try:
            proposal_set = PolicyProposalSet(**data)
            return proposal_set
        except Exception as e:
            raise ValueError(f"Invalid proposal schema: {e}")
    
    def _validate_and_convert(
        self,
        proposals: List[PolicyProposal]
    ) -> List[Dict[str, Any]]:
        """
        Validate tool names and convert to executable format.
        
        Filters out proposals with invalid tool names.
        
        Args:
            proposals: List of policy proposals
        
        Returns:
            List of validated proposals with ToolLens objects
        """
        validated = []
        
        for proposal in proposals:
            try:
                # Convert tool names to ToolLens objects
                tool_sequence = []
                for tool_name in proposal.tools:
                    tool = self.registry.get_tool(tool_name)
                    if tool is None:
                        # Invalid tool name - skip this proposal
                        raise ValueError(f"Unknown tool: {tool_name}")
                    tool_sequence.append(tool)
                
                # Create validated proposal
                validated.append({
                    'policy_id': proposal.policy_id,
                    'policy': tool_sequence,  # List of ToolLens objects
                    'llm_success_prob': proposal.estimated_success_prob,
                    'llm_info_gain': proposal.expected_information_gain,
                    'strategy': proposal.strategy,
                    'rationale': proposal.rationale,
                    'failure_modes': proposal.failure_modes,
                    'tool_names': proposal.tools  # Keep names for debugging
                })
            
            except ValueError as e:
                # Skip invalid proposals
                print(f"Skipping invalid proposal {proposal.policy_id}: {e}")
                continue
        
        return validated


def create_mock_generator(registry: ToolRegistry) -> LLMPolicyGenerator:
    """
    Create a mock generator for testing (no real LLM needed).
    
    Args:
        registry: Tool registry
    
    Returns:
        LLMPolicyGenerator with mock LLM
    
    Examples:
        >>> from unittest.mock import Mock
        >>> registry = ToolRegistry()
        >>> generator = create_mock_generator(registry)
    """
    from unittest.mock import Mock
    
    # Create mock LLM that returns valid JSON
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value=Mock(content="""
    {
      "proposals": [
        {
          "policy_id": 1,
          "tools": ["tool_a"],
          "estimated_success_prob": 0.8,
          "expected_information_gain": 0.3,
          "strategy": "exploit",
          "rationale": "Test policy",
          "failure_modes": []
        }
      ]
    }
    """))
    
    return LLMPolicyGenerator(mock_llm, registry)
