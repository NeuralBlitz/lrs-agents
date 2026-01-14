"""
Integration components for LRS-Agents.

This module provides integrations with:
- LangGraph (native graph-based execution)
- LangChain (tool adapters)
- OpenAI Assistants API
- AutoGPT
"""

from lrs.integration.langgraph import (
    LRSGraphBuilder,
    create_lrs_agent,
    LRSState
)
from lrs.integration.langchain_adapter import (
    LangChainToolLens,
    wrap_langchain_tool
)
from lrs.integration.openai_assistants import (
    OpenAIAssistantLens,
    OpenAIAssistantPolicyGenerator,
    create_openai_lrs_agent
)
from lrs.integration.autogpt_adapter import (
    LRSAutoGPTAgent,
    AutoGPTCommand,
    convert_autogpt_to_lrs
)

__all__ = [
    # LangGraph
    "LRSGraphBuilder",
    "create_lrs_agent",
    "LRSState",
    # LangChain
    "LangChainToolLens",
    "wrap_langchain_tool",
    # OpenAI
    "OpenAIAssistantLens",
    "OpenAIAssistantPolicyGenerator",
    "create_openai_lrs_agent",
    # AutoGPT
    "LRSAutoGPTAgent",
    "AutoGPTCommand",
    "convert_autogpt_to_lrs",
]
