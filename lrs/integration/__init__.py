"""Integration modules for LRS-Agents."""

# Don't import here - causes circular imports
# Users should import directly from submodules:
#   from lrs.integration.langgraph import create_lrs_agent
#   from lrs.integration.langchain_adapter import wrap_langchain_tool

__all__ = [
    "langgraph",
    "langchain_adapter", 
    "openai_assistants",
    "autogpt_adapter",
]
