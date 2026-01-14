"""
Tests for LLM policy generator.
"""

import pytest
from unittest.mock import Mock, MagicMock
import json

from lrs.inference.llm_policy_generator import (
    LLMPolicyGenerator,
    PolicyProposal,
    PolicyProposalSet,
    create_mock_generator
)
from lrs.core.registry import ToolRegistry
from lrs.core.lens import ToolLens, ExecutionResult


class DummyTool(ToolLens):
    """Dummy tool for testing"""
    def __init__(self, name):
        super().__init__(name, {}, {})
    
    def get(self, state):
        return ExecutionResult(True, "result", None, 0.1)
    
    def set(self, state, obs):
        return state


class TestPolicyProposal:
    """Test PolicyProposal Pydantic model"""
    
    def test_valid_proposal(self):
        """Test creating valid proposal"""
        proposal = PolicyProposal(
            policy_id=1,
            tools=["tool_a", "tool_b"],
            estimated_success_prob=0.8,
            expected_information_gain=0.3,
            strategy="exploit",
            rationale="Test policy",
            failure_modes=["timeout"]
        )
        
        assert proposal.policy_id == 1
        assert len(proposal.tools) == 2
        assert proposal.strategy == "exploit"
    
    def test_invalid_success_prob(self):
        """Test that success prob must be in [0, 1]"""
        with pytest.raises(ValueError):
            PolicyProposal(
                policy_id=1,
                tools=["tool"],
                estimated_success_prob=1.5,  # Invalid
                expected_information_gain=0.5,
                strategy="exploit",
                rationale="Test"
            )
    
    def test_invalid_strategy(self):
        """Test that strategy must be valid"""
        with pytest.raises(ValueError):
            PolicyProposal(
                policy_id=1,
                tools=["tool"],
                estimated_success_prob=0.8,
                expected_information_gain=0.5,
                strategy="invalid_strategy",  # Invalid
                rationale="Test"
            )
    
    def test_optional_failure_modes(self):
        """Test that failure_modes is optional"""
        proposal = PolicyProposal(
            policy_id=1,
            tools=["tool"],
            estimated_success_prob=0.8,
            expected_information_gain=0.5,
            strategy="exploit",
            rationale="Test"
        )
        
        assert proposal.failure_modes == []


class TestPolicyProposalSet:
    """Test PolicyProposalSet Pydantic model"""
    
    def test_valid_proposal_set(self):
        """Test creating valid proposal set"""
        proposals = [
            PolicyProposal(
                policy_id=i,
                tools=[f"tool_{i}"],
                estimated_success_prob=0.8,
                expected_information_gain=0.3,
                strategy="exploit",
                rationale=f"Policy {i}"
            )
            for i in range(1, 4)
        ]
        
        proposal_set = PolicyProposalSet(proposals=proposals)
        
        assert len(proposal_set.proposals) == 3
    
    def test_minimum_proposals(self):
        """Test that minimum 3 proposals required"""
        with pytest.raises(ValueError):
            PolicyProposalSet(proposals=[
                PolicyProposal(
                    policy_id=1,
                    tools=["tool"],
                    estimated_success_prob=0.8,
                    expected_information_gain=0.3,
                    strategy="exploit",
                    rationale="Only one"
                )
            ])
    
    def test_maximum_proposals(self):
        """Test that maximum 7 proposals allowed"""
        proposals = [
            PolicyProposal(
                policy_id=i,
                tools=[f"tool_{i}"],
                estimated_success_prob=0.8,
                expected_information_gain=0.3,
                strategy="exploit",
                rationale=f"Policy {i}"
            )
            for i in range(1, 9)  # 8 proposals
        ]
        
        with pytest.raises(ValueError):
            PolicyProposalSet(proposals=proposals)
    
    def test_optional_metadata(self):
        """Test optional metadata fields"""
        proposals = [
            PolicyProposal(
                policy_id=i,
                tools=["tool"],
                estimated_success_prob=0.8,
                expected_information_gain=0.3,
                strategy="exploit",
                rationale="Test"
            )
            for i in range(3)
        ]
        
        proposal_set = PolicyProposalSet(
            proposals=proposals,
            current_uncertainty=0.6,
            known_unknowns=["What we don't know"]
        )
        
        assert proposal_set.current_uncertainty == 0.6
        assert len(proposal_set.known_unknowns) == 1


class TestLLMPolicyGenerator:
    """Test LLMPolicyGenerator class"""
    
    def test_initialization(self):
        """Test generator initialization"""
        mock_llm = Mock()
        registry = ToolRegistry()
        
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        assert generator.llm == mock_llm
        assert generator.registry == registry
    
    def test_temperature_adaptation(self):
        """Test temperature adaptation based on precision"""
        mock_llm = Mock()
        registry = ToolRegistry()
        generator = LLMPolicyGenerator(mock_llm, registry, base_temperature=0.7)
        
        # Low precision → high temperature
        temp_low = generator._adapt_temperature(0.2)
        
        # High precision → low temperature
        temp_high = generator._adapt_temperature(0.9)
        
        assert temp_low > temp_high
    
    def test_temperature_clamping(self):
        """Test that temperature is clamped to reasonable range"""
        mock_llm = Mock()
        registry = ToolRegistry()
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        # Very low precision
        temp = generator._adapt_temperature(0.01)
        
        # Should be clamped
        assert 0.1 <= temp <= 2.0
    
    def test_parse_valid_response(self):
        """Test parsing valid LLM response"""
        mock_llm = Mock()
        registry = ToolRegistry()
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        response = json.dumps({
            "proposals": [
                {
                    "policy_id": 1,
                    "tools": ["tool_a"],
                    "estimated_success_prob": 0.8,
                    "expected_information_gain": 0.3,
                    "strategy": "exploit",
                    "rationale": "Test",
                    "failure_modes": []
                },
                {
                    "policy_id": 2,
                    "tools": ["tool_b"],
                    "estimated_success_prob": 0.6,
                    "expected_information_gain": 0.7,
                    "strategy": "explore",
                    "rationale": "Test",
                    "failure_modes": []
                },
                {
                    "policy_id": 3,
                    "tools": ["tool_c"],
                    "estimated_success_prob": 0.7,
                    "expected_information_gain": 0.5,
                    "strategy": "balanced",
                    "rationale": "Test",
                    "failure_modes": []
                }
            ]
        })
        
        proposal_set = generator._parse_response(response)
        
        assert len(proposal_set.proposals) == 3
    
    def test_parse_response_with_markdown(self):
        """Test parsing response with markdown code blocks"""
        mock_llm = Mock()
        registry = ToolRegistry()
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        response = """```json
        {
            "proposals": [
                {
                    "policy_id": 1,
                    "tools": ["tool"],
                    "estimated_success_prob": 0.8,
                    "expected_information_gain": 0.3,
                    "strategy": "exploit",
                    "rationale": "Test",
                    "failure_modes": []
                },
                {
                    "policy_id": 2,
                    "tools": ["tool"],
                    "estimated_success_prob": 0.6,
                    "expected_information_gain": 0.7,
                    "strategy": "explore",
                    "rationale": "Test",
                    "failure_modes": []
                },
                {
                    "policy_id": 3,
                    "tools": ["tool"],
                    "estimated_success_prob": 0.7,
                    "expected_information_gain": 0.5,
                    "strategy": "balanced",
                    "rationale": "Test",
                    "failure_modes": []
                }
            ]
        }
        ```"""
        
        proposal_set = generator._parse_response(response)
        
        assert len(proposal_set.proposals) == 3
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON raises error"""
        mock_llm = Mock()
        registry = ToolRegistry()
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        with pytest.raises(ValueError):
            generator._parse_response("not valid json")
    
    def test_validate_and_convert_valid_tools(self):
        """Test validating proposals with valid tools"""
        mock_llm = Mock()
        registry = ToolRegistry()
        
        tool_a = DummyTool("tool_a")
        tool_b = DummyTool("tool_b")
        registry.register(tool_a)
        registry.register(tool_b)
        
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        proposals = [
            PolicyProposal(
                policy_id=1,
                tools=["tool_a", "tool_b"],
                estimated_success_prob=0.8,
                expected_information_gain=0.3,
                strategy="exploit",
                rationale="Test"
            )
        ]
        
        validated = generator._validate_and_convert(proposals)
        
        assert len(validated) == 1
        assert len(validated[0]['policy']) == 2
        assert validated[0]['policy'][0] == tool_a
        assert validated[0]['policy'][1] == tool_b
    
    def test_validate_and_convert_invalid_tool(self):
        """Test that invalid tool names are filtered out"""
        mock_llm = Mock()
        registry = ToolRegistry()
        registry.register(DummyTool("valid_tool"))
        
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        proposals = [
            PolicyProposal(
                policy_id=1,
                tools=["invalid_tool"],  # Not in registry
                estimated_success_prob=0.8,
                expected_information_gain=0.3,
                strategy="exploit",
                rationale="Test"
            )
        ]
        
        validated = generator._validate_and_convert(proposals)
        
        # Should be filtered out
        assert len(validated) == 0
    
    def test_generate_proposals_success(self):
        """Test full proposal generation"""
        mock_llm = Mock()
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "proposals": [
                {
                    "policy_id": i,
                    "tools": ["test_tool"],
                    "estimated_success_prob": 0.8,
                    "expected_information_gain": 0.3,
                    "strategy": "exploit",
                    "rationale": f"Policy {i}",
                    "failure_modes": []
                }
                for i in range(1, 6)
            ]
        })
        
        mock_llm.invoke = Mock(return_value=mock_response)
        
        registry = ToolRegistry()
        registry.register(DummyTool("test_tool"))
        
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        proposals = generator.generate_proposals(
            state={'goal': 'test'},
            precision=0.5
        )
        
        assert len(proposals) == 5
        assert mock_llm.invoke.called
    
    def test_generate_proposals_handles_llm_failure(self):
        """Test that LLM failures are handled gracefully"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=Exception("LLM failed"))
        
        registry = ToolRegistry()
        generator = LLMPolicyGenerator(mock_llm, registry)
        
        proposals = generator.generate_proposals(
            state={'goal': 'test'},
            precision=0.5
        )
        
        # Should return empty list on failure
        assert proposals == []


class TestCreateMockGenerator:
    """Test mock generator creation"""
    
    def test_creates_mock_generator(self):
        """Test that mock generator is created"""
        registry = ToolRegistry()
        
        generator = create_mock_generator(registry)
        
        assert isinstance(generator, LLMPolicyGenerator)
    
    def test_mock_generator_returns_proposals(self):
        """Test that mock generator returns proposals"""
        registry = ToolRegistry()
        registry.register(DummyTool("tool_a"))
        
        generator = create_mock_generator(registry)
        
        proposals = generator.generate_proposals(
            state={'goal': 'test'},
            precision=0.5
        )
        
        # Mock should return at least one proposal
        assert len(proposals) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
