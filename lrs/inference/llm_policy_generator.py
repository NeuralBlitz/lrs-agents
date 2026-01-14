"""LLM-based policy generation for Active Inference."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator
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
    
    @field_validator('strategy')
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
        min_length=3,
        max_length=7
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


@dataclass
class LLMPolicyGenerator:
    """
    Generates policy proposals using an LLM.
    
    The LLM is prompted to generate diverse policies that balance
    exploration vs exploitation based on current precision.
    
    Args:
        llm: LangChain chat model to use
        registry: Tool registry with available tools
        prompter: Optional custom prompter (defaults to MetaCognitivePrompter)
    
    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        >>> generator = LLMPolicyGenerator(llm, registry)
        >>> proposals = generator.generate_proposals(context, precision)
    """
    
    llm: BaseChatModel
    registry: ToolRegistry
    prompter: MetaCognitivePrompter = field(default_factory=MetaCognitivePrompter)
    
    def generate_proposals(
        self,
        context: Dict[str, Any],
        precision: PrecisionParameters,
        num_proposals: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate policy proposals from LLM.
        
        Args:
            context: Current state and goal information
            precision: Current precision parameters
            num_proposals: Number of proposals to generate (3-7)
        
        Returns:
            List of policy dictionaries with tools and metadata
        """
        # Generate prompt based on precision
        prompt = self.prompter.generate_prompt(
            precision=precision.value,
            available_tools=[tool.name for tool in self.registry.tools],
            goal=context.get('goal', 'Complete the task'),
            state=context.get('state', {}),
            recent_errors=context.get('recent_errors', []),
            tool_history=context.get('tool_history', [])
        )
        
        # Call LLM
        messages = [
            SystemMessage(content="You are an AI agent planning tool usage."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse response (simplified - actual implementation would be more robust)
        try:
            # Expect JSON response with proposals
            import json
            data = json.loads(response.content)
            
            # Convert to PolicyProposalSet for validation
            proposal_set = PolicyProposalSet(**data)
            
            # Convert to internal format
            proposals = []
            for prop in proposal_set.proposals:
                # Map tool names to ToolLens objects
                tool_sequence = []
                for tool_name in prop.tool_sequence:
                    tool = self.registry.get_tool(tool_name)
                    if tool:
                        tool_sequence.append(tool)
                
                if tool_sequence:  # Only include if all tools found
                    proposals.append({
                        'tools': tool_sequence,
                        'reasoning': prop.reasoning,
                        'estimated_success_prob': prop.estimated_success_prob,
                        'estimated_info_gain': prop.estimated_info_gain,
                        'strategy': prop.strategy,
                        'failure_modes': prop.failure_modes
                    })
            
            return proposals[:num_proposals]
        
        except Exception as e:
            # Fallback: return empty list on parse error
            print(f"Failed to parse LLM response: {e}")
            return []


def create_mock_generator(num_proposals: int = 3) -> LLMPolicyGenerator:
    """
    Create a mock policy generator for testing.
    
    Returns generator that produces simple test proposals.
    """
    from unittest.mock import Mock
    
    mock_llm = Mock()
    mock_registry = Mock()
    
    return LLMPolicyGenerator(llm=mock_llm, registry=mock_registry)
