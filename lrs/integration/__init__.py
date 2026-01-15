"""Integration modules for LRS-Agents."""

from lrs.integration.langgraph import create_lrs_agent
from lrs.integration.langchain_adapter import wrap_langchain_tool

__all__ = [
    "langgraph",
    "langchain_adapter", 
    "openai_assistants",
    "autogpt_adapter",
]
