Integration API
===============

The integration module provides adapters for popular AI frameworks.

LangGraph
---------

.. automodule:: lrs.integration.langgraph
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.integration.langgraph.LRSGraphBuilder
   :members:
   :special-members: __init__
   :show-inheritance:

   Builder for LangGraph-based LRS agents.

   **Methods:**

   .. automethod:: build

   **Example:**

   .. code-block:: python

      from lrs.integration.langgraph import LRSGraphBuilder
      from langchain_anthropic import ChatAnthropic

      llm = ChatAnthropic(model="claude-sonnet-4-20250514")
      
      builder = LRSGraphBuilder(
          llm=llm,
          registry=registry,
          preferences={'success': 5.0, 'error': -3.0}
      )
      
      graph = builder.build()
      
      # Execute
      result = graph.invoke({
          'messages': [{'role': 'user', 'content': 'Task'}],
          'max_iterations': 10
      })

TypedDicts
^^^^^^^^^^

.. autoclass:: lrs.integration.langgraph.LRSState
   :members:
   :show-inheritance:

   State schema for LangGraph agents.

Functions
^^^^^^^^^

.. autofunction:: lrs.integration.langgraph.create_lrs_agent

   Convenience function to create a complete LRS agent.

   :param llm: Language model for policy generation
   :param tools: List of ToolLens objects
   :param preferences: Reward structure (optional)
   :param tracker: LRSStateTracker for monitoring (optional)
   :param use_llm_proposals: Use LLM for proposals vs exhaustive search
   :return: Compiled LangGraph agent

LangChain Adapter
-----------------

.. automodule:: lrs.integration.langchain_adapter
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.integration.langchain_adapter.LangChainToolLens
   :members:
   :special-members: __init__
   :show-inheritance:

   Wraps LangChain BaseTool as ToolLens.

   **Methods:**

   .. automethod:: get
   .. automethod:: set

   **Example:**

   .. code-block:: python

      from langchain.tools import Tool
      from lrs.integration.langchain_adapter import wrap_langchain_tool

      lc_tool = Tool(
          name="search",
          func=lambda q: f"Results for {q}",
          description="Search tool"
      )
      
      lrs_tool = wrap_langchain_tool(lc_tool, timeout=10.0)
      
      result = lrs_tool.get({'query': 'test'})

Functions
^^^^^^^^^

.. autofunction:: lrs.integration.langchain_adapter.wrap_langchain_tool

   Convenience wrapper for LangChain tools.

OpenAI Assistants
-----------------

.. automodule:: lrs.integration.openai_assistants
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.integration.openai_assistants.OpenAIAssistantLens
   :members:
   :special-members: __init__
   :show-inheritance:

   Wraps OpenAI Assistant as ToolLens for policy generation.

.. autoclass:: lrs.integration.openai_assistants.OpenAIAssistantPolicyGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

   High-level interface for OpenAI Assistants-based policy generation.

   **Example:**

   .. code-block:: python

      from openai import OpenAI
      from lrs.integration.openai_assistants import OpenAIAssistantPolicyGenerator

      client = OpenAI(api_key="sk-...")
      
      generator = OpenAIAssistantPolicyGenerator(
          client=client,
          model="gpt-4-turbo-preview"
      )
      
      proposals = generator.generate_proposals(
          state={'goal': 'Research topic'},
          precision=0.5
      )

Functions
^^^^^^^^^

.. autofunction:: lrs.integration.openai_assistants.create_openai_lrs_agent

   Create complete LRS agent powered by OpenAI Assistants.

AutoGPT Adapter
---------------

.. automodule:: lrs.integration.autogpt_adapter
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.integration.autogpt_adapter.AutoGPTCommand
   :members:
   :special-members: __init__
   :show-inheritance:

   Wraps AutoGPT command function as ToolLens.

.. autoclass:: lrs.integration.autogpt_adapter.LRSAutoGPTAgent
   :members:
   :special-members: __init__
   :show-inheritance:

   LRS-powered AutoGPT agent with automatic adaptation.

   **Methods:**

   .. automethod:: run

   **Example:**

   .. code-block:: python

      from lrs.integration.autogpt_adapter import LRSAutoGPTAgent

      def search_web(query: str) -> dict:
          # Implementation
          return {'results': [...]}

      agent = LRSAutoGPTAgent(
          name="ResearchAgent",
          role="Research assistant",
          commands={'search_web': search_web},
          llm=llm,
          goals=["Research topic", "Write report"]
      )
      
      result = agent.run(task="Research AI trends")

Functions
^^^^^^^^^

.. autofunction:: lrs.integration.autogpt_adapter.convert_autogpt_to_lrs

   Convert AutoGPT configuration to LRS-compatible format.

