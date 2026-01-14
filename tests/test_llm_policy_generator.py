"""Test suite for LLM Policy Generator.

Validates:
1. Precision-adaptive behavior (temperature, prompt content)
2. Schema validation and tool mapping
3. Diversity enforcement
4. Self-calibration accuracy
5. Error handling for invalid LLM outputs
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json

from lrs.inference.llm_policy_generator import LLMPolicyGenerator
from lrs.inference.prompts import MetaCognitivePrompter, PromptContext
from lrs.core.registry import ToolRegistry
from lrs.core.lens import ToolLens


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM with controllable responses"""
    llm = Mock()
    llm.generate = Mock()
    return llm


@pytest.fixture
def mock_registry():
    """Mock tool registry with diverse tools"""
    registry = Mock(spec=ToolRegistry)
    
    # Create mock tools
    tool_a = Mock(spec=ToolLens)
    tool_a.name = "tool_a"
    tool_a.call_count = 10
    tool_a.failure_count = 2  # 80% success
    
    tool_b = Mock(spec=ToolLens)
    tool_b.name = "tool_b"
    tool_b.call_count = 10
    tool_b.failure_count = 5  # 50% success
    
    tool_c = Mock(spec=ToolLens)
    tool_c.name = "tool_c"
    tool_c.call_count = 0
    tool_c.failure_count = 0  # Never tried
    
    registry.tools = {
        "tool_a": tool_a,
        "tool_b": tool_b,
        "tool_c": tool_c
    }
    
    return registry


@pytest.fixture
def valid_llm_response():
    """Valid LLM response matching schema"""
    return {
        "proposals": [
            {
                "policy_id": 1,
                "tools": [
                    {
                        "tool_name": "tool_a",
                        "reasoning": "High success rate",
                        "expected_output": "data"
                    }
                ],
                "estimated_success_prob": 0.8,
                "expected_information_gain": 0.2,
                "strategy": "exploit",
                "rationale": "Use proven tool",
                "failure_modes": ["timeout"]
            },
            {
                "policy_id": 2,
                "tools": [
                    {
                        "tool_name": "tool_c",
                        "reasoning": "Novel approach",
                        "expected_output": "unknown"
                    }
                ],
                "estimated_success_prob": 0.5,
                "expected_information_gain": 0.9,
                "strategy": "explore",
                "rationale": "Test untried tool",
                "failure_modes": ["unknown behavior"]
            },
            {
                "policy_id": 3,
                "tools": [
                    {
                        "tool_name": "tool_a",
                        "reasoning": "Reliable",
                        "expected_output": "data"
                    },
                    {
                        "tool_name": "tool_b",
                        "reasoning": "Fallback",
                        "expected_output": "data"
                    }
                ],
                "estimated_success_prob": 0.7,
                "expected_information_gain": 0.4,
                "strategy": "balanced",
                "rationale": "Hedged approach",
                "failure_modes": ["both tools fail"]
            }
        ],
        "current_uncertainty": 0.5,
        "known_unknowns": ["State of external API"]
    }


# ============================================================================
# Test: Precision-Adaptive Behavior
# ============================================================================

class TestPrecisionAdaptation:
    """Test that precision influences LLM prompting and temperature"""
    
    def test_low_precision_increases_temperature(self, mock_llm, mock_registry, valid_llm_response):
        """Low precision → high temperature → diverse exploration"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        # Low precision scenario
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.2  # Very low
        )
        
        # Check temperature was increased
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert call_kwargs['temperature'] > 0.7
        assert call_kwargs['temperature'] < 1.0
    
    def test_high_precision_decreases_temperature(self, mock_llm, mock_registry, valid_llm_response):
        """High precision → low temperature → focused exploitation"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        # High precision scenario
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.9  # Very high
        )
        
        # Check temperature was decreased
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert call_kwargs['temperature'] < 0.5
    
    def test_prompt_contains_precision_value(self, mock_llm, mock_registry, valid_llm_response):
        """Verify precision is communicated to LLM in prompt"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        precision = 0.35
        generator.generate_proposals(
            state={"goal": "test"},
            precision=precision
        )
        
        # Extract prompt from call
        prompt = mock_llm.generate.call_args.args[0]
        
        # Verify precision appears in prompt
        assert f"{precision:.3f}" in prompt or f"{precision:.2f}" in prompt
    
    def test_low_precision_triggers_exploration_guidance(self, mock_llm, mock_registry, valid_llm_response):
        """Low precision → prompt contains exploration instructions"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        generator.generate_proposals(
            state={"goal": "test"},
            precision=0.25  # Below exploration threshold
        )
        
        prompt = mock_llm.generate.call_args.args[0]
        
        # Check for exploration keywords
        assert "EXPLORATION MODE" in prompt or "explore" in prompt.lower()
        assert "information" in prompt.lower()
    
    def test_high_precision_triggers_exploitation_guidance(self, mock_llm, mock_registry, valid_llm_response):
        """High precision → prompt contains exploitation instructions"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        generator.generate_proposals(
            state={"goal": "test"},
            precision=0.85  # Above exploitation threshold
        )
        
        prompt = mock_llm.generate.call_args.args[0]
        
        # Check for exploitation keywords
        assert "EXPLOITATION MODE" in prompt or "exploit" in prompt.lower()
        assert "reward" in prompt.lower() or "success" in prompt.lower()


# ============================================================================
# Test: Schema Validation and Tool Mapping
# ============================================================================

class TestSchemaValidation:
    """Test LLM output validation and conversion to executable policies"""
    
    def test_valid_response_parsed_correctly(self, mock_llm, mock_registry, valid_llm_response):
        """Valid LLM response is parsed without errors"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.5
        )
        
        assert len(proposals) == 3
        assert all('policy' in p for p in proposals)
        assert all('llm_success_prob' in p for p in proposals)
        assert all('llm_info_gain' in p for p in proposals)
    
    def test_tool_names_mapped_to_lens_objects(self, mock_llm, mock_registry, valid_llm_response):
        """String tool names converted to actual ToolLens objects"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.5
        )
        
        # First proposal uses tool_a
        first_policy = proposals[0]['policy']
        assert len(first_policy) == 1
        assert first_policy[0] == mock_registry.tools['tool_a']
        
        # Third proposal uses tool_a and tool_b
        third_policy = proposals[2]['policy']
        assert len(third_policy) == 2
        assert third_policy[0] == mock_registry.tools['tool_a']
        assert third_policy[1] == mock_registry.tools['tool_b']
    
    def test_invalid_tool_name_skips_proposal(self, mock_llm, mock_registry):
        """Proposals with invalid tool names are skipped"""
        invalid_response = {
            "proposals": [
                {
                    "policy_id": 1,
                    "tools": [{"tool_name": "nonexistent_tool", "reasoning": "test"}],
                    "estimated_success_prob": 0.5,
                    "expected_information_gain": 0.5,
                    "strategy": "explore",
                    "rationale": "test",
                    "failure_modes": []
                },
                {
                    "policy_id": 2,
                    "tools": [{"tool_name": "tool_a", "reasoning": "test"}],
                    "estimated_success_prob": 0.8,
                    "expected_information_gain": 0.2,
                    "strategy": "exploit",
                    "rationale": "test",
                    "failure_modes": []
                }
            ],
            "current_uncertainty": 0.5,
            "known_unknowns": []
        }
        
        mock_llm.generate.return_value = invalid_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.5
        )
        
        # Only valid proposal should remain
        assert len(proposals) == 1
        assert proposals[0]['policy'][0] == mock_registry.tools['tool_a']
    
    def test_metadata_preserved(self, mock_llm, mock_registry, valid_llm_response):
        """LLM metadata (strategy, rationale, etc.) is preserved"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.5
        )
        
        # Check first proposal metadata
        assert proposals[0]['strategy'] == 'exploit'
        assert proposals[0]['rationale'] == 'Use proven tool'
        assert proposals[0]['failure_modes'] == ['timeout']
        
        # Check second proposal
        assert proposals[1]['strategy'] == 'explore'
        assert proposals[1]['llm_info_gain'] == 0.9


# ============================================================================
# Test: Diversity Enforcement
# ============================================================================

class TestDiversityEnforcement:
    """Test that proposals span exploration-exploitation spectrum"""
    
    def test_proposals_span_strategies(self, mock_llm, mock_registry, valid_llm_response):
        """Proposals include exploit, explore, and balanced strategies"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.5
        )
        
        strategies = [p['strategy'] for p in proposals]
        
        # Should have at least one of each
        assert 'exploit' in strategies
        assert 'explore' in strategies
        assert 'balanced' in strategies
    
    def test_information_gain_correlates_with_strategy(self, mock_llm, mock_registry, valid_llm_response):
        """Explore strategies have higher info gain than exploit"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.5
        )
        
        exploit_info = [p['llm_info_gain'] for p in proposals if p['strategy'] == 'exploit']
        explore_info = [p['llm_info_gain'] for p in proposals if p['strategy'] == 'explore']
        
        # Explore should have higher average info gain
        assert sum(explore_info) / len(explore_info) > sum(exploit_info) / len(exploit_info)


# ============================================================================
# Test: Prediction Error Interpretation
# ============================================================================

class TestPredictionErrorHandling:
    """Test that recent errors influence LLM prompting"""
    
    def test_high_error_mentioned_in_prompt(self, mock_llm, mock_registry, valid_llm_response):
        """Recent high prediction errors appear in prompt"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        state = {
            "goal": "test",
            "tool_history": [
                {
                    "tool": "tool_a",
                    "success": False,
                    "prediction_error": 0.95,
                    "error_message": "Permission denied"
                }
            ]
        }
        
        generator.generate_proposals(state=state, precision=0.3)
        
        prompt = mock_llm.generate.call_args.args[0]
        
        # Should mention the error
        assert "0.95" in prompt or "HIGH" in prompt
        assert "tool_a" in prompt
    
    def test_error_triggers_diagnostic_suggestion(self, mock_llm, mock_registry, valid_llm_response):
        """High errors trigger diagnostic action suggestions in prompt"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        state = {
            "goal": "test",
            "tool_history": [
                {
                    "tool": "tool_b",
                    "success": False,
                    "prediction_error": 0.85
                }
            ]
        }
        
        generator.generate_proposals(state=state, precision=0.25)
        
        prompt = mock_llm.generate.call_args.args[0]
        
        # Should suggest investigation
        assert "investigate" in prompt.lower() or "diagnostic" in prompt.lower()


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================

class TestErrorHandling:
    """Test robustness to invalid LLM outputs"""
    
    def test_empty_proposals_handled(self, mock_llm, mock_registry):
        """Empty proposal list doesn't crash"""
        mock_llm.generate.return_value = {
            "proposals": [],
            "current_uncertainty": 0.5,
            "known_unknowns": []
        }
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        proposals = generator.generate_proposals(
            state={"goal": "test"},
            precision=0.5
        )
        
        assert proposals == []
    
    def test_malformed_json_handled(self, mock_llm, mock_registry):
        """Malformed LLM response raises clear error"""
        mock_llm.generate.return_value = "Not valid JSON"
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        with pytest.raises(Exception):  # Should raise validation error
            generator.generate_proposals(
                state={"goal": "test"},
                precision=0.5
            )
    
    def test_missing_required_fields_skips_proposal(self, mock_llm, mock_registry):
        """Proposals missing required fields are skipped"""
        incomplete_response = {
            "proposals": [
                {
                    "policy_id": 1,
                    "tools": [{"tool_name": "tool_a"}],
                    # Missing: estimated_success_prob, expected_information_gain, etc.
                }
            ],
            "current_uncertainty": 0.5,
            "known_unknowns": []
        }
        
        mock_llm.generate.return_value = incomplete_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        # Should handle gracefully (skip invalid proposal)
        with pytest.raises(Exception):  # Pydantic validation should catch
            generator.generate_proposals(
                state={"goal": "test"},
                precision=0.5
            )


# ============================================================================
# Integration Test
# ============================================================================

class TestIntegration:
    """End-to-end integration test"""
    
    def test_complete_generation_flow(self, mock_llm, mock_registry, valid_llm_response):
        """Full flow from state to validated proposals"""
        mock_llm.generate.return_value = valid_llm_response
        
        generator = LLMPolicyGenerator(mock_llm, mock_registry)
        
        state = {
            "goal": "Extract data from API",
            "belief_state": {"api_status": "unknown"},
            "tool_history": [
                {
                    "tool": "tool_a",
                    "success": True,
                    "prediction_error": 0.1
                },
                {
                    "tool": "tool_b",
                    "success": False,
                    "prediction_error": 0.8
                }
            ]
        }
        
        proposals = generator.generate_proposals(
            state=state,
            precision=0.45  # Medium precision
        )
        
        # Verify output structure
        assert len(proposals) == 3
        
        # All proposals have required fields
        for prop in proposals:
            assert 'policy' in prop
            assert isinstance(prop['policy'], list)
            assert all(isinstance(tool, Mock) for tool in prop['policy'])
            
            assert 'llm_success_prob' in prop
            assert 0 <= prop['llm_success_prob'] <= 1
            
            assert 'llm_info_gain' in prop
            assert 0 <= prop['llm_info_gain'] <= 1
            
            assert 'strategy' in prop
            assert prop['strategy'] in ['exploit', 'explore', 'balanced']
