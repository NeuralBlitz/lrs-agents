"""
Inference components for LRS-Agents.

This module provides:
- Meta-cognitive prompting (precision-adaptive)
- LLM policy generation (variational proposals)
- Hybrid G evaluation (LLM + mathematical)
"""

from lrs.inference.prompts import MetaCognitivePrompter, PromptContext
from lrs.inference.llm_policy_generator import LLMPolicyGenerator
from lrs.inference.evaluator import HybridGEvaluator

__all__ = [
    "MetaCognitivePrompter",
    "PromptContext",
    "LLMPolicyGenerator",
    "HybridGEvaluator",
]
