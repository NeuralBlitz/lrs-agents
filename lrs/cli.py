"""CLI utilities and banner for LRS-Agents."""

import sys
from typing import Optional


BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██╗     ██████╗ ███████╗      █████╗  ██████╗ ███████╗███╗   ██╗████████╗║
║   ██║     ██╔══██╗██╔════╝     ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝║
║   ██║     ██████╔╝███████╗     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ║
║   ██║     ██╔══██╗╚════██║     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ║
║   ███████╗██║  ██║███████║     ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ║
║   ╚══════╝╚═╝  ╚═╝╚══════╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ║
║                                                                              ║
║                  🧠 Resilient AI Agents via Active Inference                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Version: {version}
  
  ✨ Features:
     • Automatic adaptation when tools fail
     • Precision tracking via Beta distributions  
     • Active Inference & Free Energy minimization
     • LangChain, OpenAI, AutoGPT integrations
  
  📚 Quick Start:
     from lrs.integration.langgraph import create_lrs_agent
     from langchain_anthropic import ChatAnthropic
     
     llm = ChatAnthropic(model="claude-sonnet-4-20250514")
     agent = create_lrs_agent(llm, tools=[...])
  
  📖 Documentation: https://lrs-agents.readthedocs.io
  🐛 Issues:        https://github.com/NeuralBlitz/lrs-agents/issues
  ⭐ Star us:       https://github.com/NeuralBlitz/lrs-agents

"""


COMPACT_BANNER = """
┌─────────────────────────────────────────────────────────────┐
│  🧠 LRS-Agents v{version}                                   │
│  Resilient AI Agents via Active Inference                   │
│  📚 https://lrs-agents.readthedocs.io                       │
└─────────────────────────────────────────────────────────────┘
"""


def show_banner(compact: bool = False, version: Optional[str] = None) -> None:
    """
    Display the LRS-Agents banner.
    
    Args:
        compact: Show compact version
        version: Version string (auto-detected if None)
    """
    if version is None:
        try:
            from lrs import __version__
            version = __version__
        except ImportError:
            version = "unknown"
    
    banner = COMPACT_BANNER if compact else BANNER
    print(banner.format(version=version))


def welcome() -> None:
    """Display welcome message on first import."""
    show_banner(compact=False)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LRS-Agents: Resilient AI Agents via Active Inference"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=get_version())
    )
    parser.add_argument(
        "--banner",
        action="store_true",
        help="Show banner"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Show compact banner"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show package information"
    )
    
    args = parser.parse_args()
    
    if args.banner or args.compact:
        show_banner(compact=args.compact)
    elif args.info:
        show_info()
    else:
        parser.print_help()


def get_version() -> str:
    """Get package version."""
    try:
        from lrs import __version__
        return __version__
    except ImportError:
        return "unknown"


def show_info() -> None:
    """Show package information."""
    try:
        from lrs import __version__
    except ImportError:
        __version__ = "unknown"
    
    info = f"""
LRS-Agents Package Information
{'=' * 50}

Version:        {__version__}
Python:         {sys.version.split()[0]}
Platform:       {sys.platform}

Installation:
  pip install lrs-agents

Optional Dependencies:
  pip install lrs-agents[langchain]  # LangChain integration
  pip install lrs-agents[openai]     # OpenAI Assistants
  pip install lrs-agents[monitoring] # Dashboard & logging
  pip install lrs-agents[all]        # Everything

Documentation:  https://lrs-agents.readthedocs.io
Repository:     https://github.com/NeuralBlitz/lrs-agents
Issues:         https://github.com/NeuralBlitz/lrs-agents/issues

{'=' * 50}
"""
    print(info)


if __name__ == "__main__":
    main()
