"""LLM-based policy generation for Active Inference."""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from unittest.mock import Mock, MagicMock

from pydantic import BaseModel, Field, field_validator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from lrs.core.lens import ToolLens
from lrs.core.registry import ToolRegistry
from lrs.core.precision import PrecisionParameters
from lrs.inference.prompts import MetaCognitivePrompter, StrategyMode, PromptContext


class PolicyProposal(BaseModel):
    """A single policy proposal with metadata."""

    tool_sequence: List[str] = Field(description="Ordered list of tool names to execute")
    reasoning: str = Field(description="Explanation of why this policy might work")
    estimated_success_prob: float = Field(
        ge=0.0, le=1.0, description="Estimated probability of success"
    )
    estimated_info_gain: float = Field(ge=0.0, le=1.0, description="Expected information gain")
    strategy: str = Field(description="Strategy type: exploitation, exploration, or balanced")
    failure_modes: List[str] = Field(
        default_factory=list, description="Potential failure scenarios"
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = ["exploitation", "exploration", "balanced"]
        if v not in valid:
            raise ValueError(f"Strategy must be one of {valid}")
        return v


class PolicyProposalSet(BaseModel):
    """Complete set of policy proposals with metadata."""

    proposals: List[PolicyProposal]
    current_uncertainty: float = Field(ge=0.0, le=1.0)
    known_unknowns: List[str] = Field(default_factory=list)


class LLMPolicyGenerator:
    """
    Generates policy proposals using an LLM with Active Inference principles.

    The generator uses meta-cognitive prompting to produce diverse policies
    that balance exploration and exploitation based on precision parameters.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        registry: ToolRegistry,
        prompter: Optional[MetaCognitivePrompter] = None,
    ):
        """
        Initialize the policy generator.

        Args:
            llm: Language model for generating proposals
            registry: Tool registry for available actions
            prompter: Optional custom prompter (creates default if None)
        """
        self.llm = llm
        self.registry = registry
        self.prompter = prompter or MetaCognitivePrompter()

    def generate_proposals(
        self,
        state: Dict[str, Any] = None,
        precision: PrecisionParameters = None,
        num_proposals: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate policy proposals based on current context and precision.

        Args:
            state: Current state, goal, and history (deprecated, use context instead)
            context: Current state, goal, and history
            precision: Precision parameters guiding exploration/exploitation
            num_proposals: Number of proposals to generate

        Returns:
            List of policy dictionaries with tools and metadata
        """
        # Support both 'state' and 'context' for backward compatibility
        if state is None:
            context = {}
        elif "state" in state or "goal" in state:
            # Already in context format
            context = state
        else:
            # Legacy format - wrap in context
            context = {"state": state}

        if precision is None:
            precision = PrecisionParameters()

        # Generate prompt based on precision
        prompt_context = PromptContext(
            precision=precision.value,
            available_tools=[tool.name for tool in self.registry.tools],
            goal=context.get("goal", "Complete task"),
            state=context.get("state", {}),
            recent_errors=context.get("recent_errors", []),
            tool_history=context.get("tool_history", []),
        )
        prompt = self.prompter.generate_prompt(prompt_context)

        # Call LLM
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Generate {num_proposals} policy proposals."),
        ]

        response = self.llm.invoke(messages)

        # Parse and validate response
        try:
            # Extract JSON from response
            content = response.content
            if isinstance(content, str):
                # Handle markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

            proposal_set = PolicyProposalSet.model_validate_json(content)
        except Exception as e:
            # Fallback to simple proposals if parsing fails
            print(f"Warning: Failed to parse LLM response: {e}")
            return self._create_fallback_proposals(num_proposals)

        # Convert to policy dictionaries
        policies = []
        for proposal in proposal_set.proposals:
            # Get actual tool objects
            tools = []
            for tool_name in proposal.tool_sequence:
                tool = self.registry.get_tool(tool_name)
                if tool:
                    tools.append(tool)

            if tools:  # Only include if we found valid tools
                policies.append(
                    {
                        "tools": tools,
                        "reasoning": proposal.reasoning,
                        "estimated_success": proposal.estimated_success_prob,
                        "estimated_info_gain": proposal.estimated_info_gain,
                        "strategy": proposal.strategy,
                        "failure_modes": proposal.failure_modes,
                    }
                )

        return policies[:num_proposals]

    def _create_fallback_proposals(self, num_proposals: int) -> List[Dict[str, Any]]:
        """Create simple fallback proposals when LLM parsing fails."""
        proposals = []
        tools = list(self.registry.tools)[:num_proposals]

        for i, tool in enumerate(tools):
            proposals.append(
                {
                    "tools": [tool],
                    "reasoning": f"Fallback proposal using {tool.name}",
                    "estimated_success": 0.5,
                    "estimated_info_gain": 0.5,
                    "strategy": "balanced",
                    "failure_modes": ["Unknown - fallback proposal"],
                }
            )

        return proposals


def create_mock_generator(num_proposals: int = 3) -> LLMPolicyGenerator:
    """
    Create a mock policy generator for testing.

    Args:
        num_proposals: Number of proposals the mock should generate

    Returns:
        Generator that produces simple test proposals.
    """
    # 1. Create a valid JSON response that the mock LLM will return.
    # This response must conform to the PolicyProposalSet schema.
    proposals_data = []
    tool_names = []
    for i in range(num_proposals):
        tool_name = f"mock_tool_{i}"
        tool_names.append(tool_name)
        proposals_data.append(
            {
                "tool_sequence": [tool_name],
                "reasoning": f"Reasoning for using {tool_name}",
                "estimated_success_prob": 0.85,
                "estimated_info_gain": 0.6,
                "strategy": "balanced",
                "failure_modes": ["It might fail if the input is wrong."],
            }
        )

    response_data = {
        "proposals": proposals_data,
        "current_uncertainty": 0.3,
        "known_unknowns": ["The exact format of the API response."],
    }

    # The response content must be a JSON string
    json_response = json.dumps(response_data)

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
