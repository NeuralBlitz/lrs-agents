"""LLM-based policy generation for Active Inference."""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from unittest.mock import Mock, MagicMock

from pydantic import BaseModel, Field, field_validator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from lrs.core.lens import ToolLens
from lrs.core.registry import ToolRegistry
from lrs.core.precision import PrecisionParameters
from lrs.inference.prompts import MetaCognitivePrompter, StrategyMode, PromptContext


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
        prompt_context = PromptContext(
            precision=precision.value,
            available_tools=[tool.name for tool in self.registry.tools],
            goal=context.get('goal', 'Complete the task'),
            state=context.get('state', {}),
            recent_errors=context.get('recent_errors', []),
            tool_history=context.get('tool_history', [])
        )
        prompt = self.prompter.generate_prompt(prompt_context)
        
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
    # 1. Create a valid JSON response that the mock LLM will return.
    # This response must conform to the PolicyProposalSet schema.
    proposals_data = []
    tool_names = []
    for i in range(num_proposals):
        tool_name = f"mock_tool_{i}"
        tool_names.append(tool_name)
        proposals_data.append({
            "tool_sequence": [tool_name],
            "reasoning": f"Reasoning for using {tool_name}",
            "estimated_success_prob": 0.85,
            "estimated_info_gain": 0.6,
            "strategy": "balanced",
            "failure_modes": ["It might fail if the input is wrong."]
        })

    response_data = {
        "proposals": proposals_data,
        "current_uncertainty": 0.3,
        "known_unknowns": ["The exact format of the API response."]
    }

    # The response content must be a JSON string
    json_response = json.dumps(response_data)

    # 2. Configure the mock LLM to return the JSON response.
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json_response
    mock_llm.invoke.return_value = mock_response
    
    # 3. Configure the mock ToolRegistry.
    mock_registry = MagicMock()

    # The generate_proposals method needs `registry.get_tool` to be callable
    # and to return a tool object for the names in our mock response.
    # It also needs `registry.tools` to be iterable for prompt generation.

    # Create mock tools. Using MagicMock is fine for this purpose.
    mock_tools = {}
    for name in tool_names:
        tool = MagicMock()
        tool.name = name
        mock_tools[name] = tool

    mock_registry.get_tool.side_effect = lambda name: mock_tools.get(name)
    mock_registry.tools = list(mock_tools.values())
    
    return LLMPolicyGenerator(llm=mock_llm, registry=mock_registry)
