LangChain Integration
=====================

LRS-Agents integrates seamlessly with LangChain, allowing you to use any LangChain tool with automatic adaptation.

Overview
--------

This guide covers:

* Converting LangChain tools to LRS tools
* Using LangChain agents with LRS
* Combining LangChain chains with precision tracking
* Best practices for integration

Quick Start
-----------

Convert a LangChain Tool
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from langchain.tools import Tool
   from lrs.integration.langchain_adapter import wrap_langchain_tool

   # Create LangChain tool
   search_tool = Tool(
       name="search",
       func=lambda q: f"Results for {q}",
       description="Search the web"
   )

   # Convert to LRS tool
   lrs_search = wrap_langchain_tool(search_tool, timeout=10.0)

   # Use in LRS agent
   from lrs import create_lrs_agent
   from langchain_anthropic import ChatAnthropic

   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
   agent = create_lrs_agent(llm, [lrs_search])

Using LangChain Tools
---------------------

Basic Wrapper
^^^^^^^^^^^^^

The simplest way to use LangChain tools:

.. code-block:: python

   from langchain_community.tools import DuckDuckGoSearchRun
   from lrs.integration.langchain_adapter import wrap_langchain_tool

   # Create LangChain tool
   duckduckgo = DuckDuckGoSearchRun()

   # Wrap for LRS
   lrs_search = wrap_langchain_tool(duckduckgo)

   # Tool now has:
   # - Automatic timeout handling
   # - Prediction error calculation
   # - Call/failure statistics tracking

Advanced Wrapper
^^^^^^^^^^^^^^^^

For more control over prediction errors:

.. code-block:: python

   from lrs.integration.langchain_adapter import LangChainToolLens

   def custom_error_fn(result, output_schema):
       """Custom prediction error calculation"""
       if result is None:
           return 0.9  # High surprise for null
       elif len(result) == 0:
           return 0.7  # Medium surprise for empty
       else:
           return 0.1  # Low surprise for success
   
   lrs_tool = LangChainToolLens(
       tool=langchain_tool,
       timeout=15.0,
       error_fn=custom_error_fn
   )

Common LangChain Tools
----------------------

Web Search
^^^^^^^^^^

.. code-block:: python

   from langchain_community.tools import DuckDuckGoSearchRun
   from lrs.integration.langchain_adapter import wrap_langchain_tool

   search = wrap_langchain_tool(DuckDuckGoSearchRun())

Wikipedia
^^^^^^^^^

.. code-block:: python

   from langchain_community.tools import WikipediaQueryRun
   from langchain_community.utilities import WikipediaAPIWrapper

   wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
   lrs_wiki = wrap_langchain_tool(wikipedia)

Python REPL
^^^^^^^^^^^

.. code-block:: python

   from langchain_experimental.tools import PythonREPLTool

   python_repl = wrap_langchain_tool(PythonREPLTool())

File Operations
^^^^^^^^^^^^^^^

.. code-block:: python

   from langchain_community.tools import ReadFileTool, WriteFileTool

   read_file = wrap_langchain_tool(ReadFileTool())
   write_file = wrap_langchain_tool(WriteFileTool())

Using LangChain Agents with LRS
--------------------------------

You can use LangChain's agent executors with LRS precision tracking:

.. code-block:: python

   from langchain.agents import create_react_agent, AgentExecutor
   from langchain_anthropic import ChatAnthropic
   from langchain.prompts import PromptTemplate
   from lrs.core.precision import HierarchicalPrecision
   from lrs.monitoring.tracker import LRSStateTracker

   # Create LangChain agent
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
   tools = [search, wikipedia, calculator]
   
   agent = create_react_agent(llm, tools, prompt)
   executor = AgentExecutor(agent=agent, tools=tools)

   # Add LRS tracking
   hp = HierarchicalPrecision()
   tracker = LRSStateTracker()

   # Execute with tracking
   result = executor.invoke({"input": "Research Active Inference"})
   
   # Track precision based on result
   if result.get('output'):
       hp.update('execution', prediction_error=0.1)
   else:
       hp.update('execution', prediction_error=0.9)

LangChain + LRS Hybrid
-----------------------

Combine LangChain's ecosystem with LRS's adaptation:

.. code-block:: python

   from langchain_anthropic import ChatAnthropic
   from langchain_community.tools import DuckDuckGoSearchRun
   from lrs import create_lrs_agent
   from lrs.integration.langchain_adapter import wrap_langchain_tool
   from lrs.core.lens import ToolLens, ExecutionResult

   # Use LangChain tools
   search = wrap_langchain_tool(DuckDuckGoSearchRun())

   # Mix with custom LRS tools
   class CacheTool(ToolLens):
       def __init__(self):
           super().__init__("cache", {}, {})
           self.cache = {}
       
       def get(self, state):
           query = state.get('query')
           if query in self.cache:
               return ExecutionResult(True, self.cache[query], None, 0.0)
           return ExecutionResult(False, None, "Not in cache", 0.5)
       
       def set(self, state, obs):
           return {**state, 'result': obs}

   # Create hybrid agent
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
   agent = create_lrs_agent(
       llm=llm,
       tools=[search, CacheTool()],  # Mix LangChain + custom
       preferences={'success': 5.0, 'error': -3.0}
   )

   # Agent automatically:
   # - Tries search first (high reward if successful)
   # - Falls back to cache if search fails
   # - Adapts based on precision

Using LangChain Chains
----------------------

You can wrap entire LangChain chains as LRS tools:

.. code-block:: python

   from langchain.chains import LLMChain
   from langchain.prompts import PromptTemplate
   from lrs.core.lens import ToolLens, ExecutionResult

   class LangChainChainTool(ToolLens):
       """Wrap a LangChain chain as a tool"""
       
       def __init__(self, chain, name="chain"):
           super().__init__(name, {}, {})
           self.chain = chain
       
       def get(self, state):
           self.call_count += 1
           try:
               result = self.chain.run(**state)
               return ExecutionResult(True, result, None, 0.2)
           except Exception as e:
               self.failure_count += 1
               return ExecutionResult(False, None, str(e), 0.8)
       
       def set(self, state, obs):
           return {**state, f'{self.name}_output': obs}

   # Use it
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
   prompt = PromptTemplate.from_template("Summarize: {text}")
   chain = LLMChain(llm=llm, prompt=prompt)

   summarize_tool = LangChainChainTool(chain, name="summarize")

LangGraph Integration
---------------------

LRS provides native LangGraph support:

.. code-block:: python

   from lrs.integration.langgraph import create_lrs_agent
   from langchain_anthropic import ChatAnthropic
   from lrs.integration.langchain_adapter import wrap_langchain_tool

   # Create tools
   tools = [
       wrap_langchain_tool(DuckDuckGoSearchRun()),
       wrap_langchain_tool(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))
   ]

   # Create LangGraph-based LRS agent
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
   agent = create_lrs_agent(
       llm=llm,
       tools=tools,
       use_llm_proposals=True
   )

   # Execute
   result = agent.invoke({
       'messages': [{'role': 'user', 'content': 'Research quantum computing'}],
       'max_iterations': 20
   })

See the full :doc:`../api/integration` for more details.

Error Handling
--------------

LangChain tools can fail in various ways. LRS handles them automatically:

Timeouts
^^^^^^^^

.. code-block:: python

   # Set timeout when wrapping
   tool = wrap_langchain_tool(slow_tool, timeout=30.0)

   # Tool will return ExecutionResult with:
   # - success=False
   # - error="Timeout after 30.0s"
   # - prediction_error=0.7

Rate Limits
^^^^^^^^^^^

.. code-block:: python

   from langchain_community.tools import DuckDuckGoSearchRun
   from lrs.integration.langchain_adapter import wrap_langchain_tool

   search = wrap_langchain_tool(DuckDuckGoSearchRun())

   # When rate limited:
   # - LRS detects high prediction error
   # - Precision drops
   # - Agent explores alternatives (cache, different API, etc.)

Network Errors
^^^^^^^^^^^^^^

.. code-block:: python

   # Network failures have high prediction error
   # Agent automatically:
   # 1. Detects surprise (error = 0.9)
   # 2. Precision drops
   # 3. Explores alternatives
   # No manual retry logic needed!

Best Practices
--------------

1. Set Appropriate Timeouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Fast tools
   fast_tool = wrap_langchain_tool(cache_tool, timeout=1.0)

   # Slow tools (API calls, web scraping)
   slow_tool = wrap_langchain_tool(web_scraper, timeout=30.0)

   # Very slow tools (database queries, large files)
   very_slow = wrap_langchain_tool(db_tool, timeout=120.0)

2. Provide Custom Error Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For domain-specific prediction errors:

.. code-block:: python

   def api_error_fn(result, schema):
       if result is None:
           return 0.95  # Total failure
       elif result.get('status') == 'rate_limited':
           return 0.7   # Expected but bad
       elif result.get('status') == 'success':
           return 0.1   # Expected and good
       else:
           return 0.5   # Uncertain

   tool = LangChainToolLens(api_tool, error_fn=api_error_fn)

3. Register Alternatives
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lrs.core.registry import ToolRegistry

   registry = ToolRegistry()
   
   # Primary tool with alternatives
   registry.register(
       wrap_langchain_tool(DuckDuckGoSearchRun()),
       alternatives=["wikipedia", "cache"]
   )
   registry.register(wrap_langchain_tool(WikipediaQueryRun(...)))
   registry.register(CacheTool())

4. Monitor Performance
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lrs.monitoring.tracker import LRSStateTracker

   tracker = LRSStateTracker()
   agent = create_lrs_agent(llm, tools, tracker=tracker)

   # After execution
   stats = tracker.get_tool_usage_stats()
   
   for tool_name, tool_stats in stats.items():
       print(f"{tool_name}:")
       print(f"  Success rate: {tool_stats['success_rate']:.1%}")
       print(f"  Avg error: {tool_stats['avg_error']:.2f}")

Complete Example
----------------

Here's a complete example combining everything:

.. code-block:: python

   from langchain_anthropic import ChatAnthropic
   from langchain_community.tools import (
       DuckDuckGoSearchRun,
       WikipediaQueryRun,
       WikipediaAPIWrapper
   )
   from lrs import create_lrs_agent
   from lrs.integration.langchain_adapter import wrap_langchain_tool
   from lrs.core.registry import ToolRegistry
   from lrs.monitoring.tracker import LRSStateTracker

   # Initialize
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
   tracker = LRSStateTracker()

   # Create LangChain tools
   search = wrap_langchain_tool(DuckDuckGoSearchRun(), timeout=10.0)
   wikipedia = wrap_langchain_tool(
       WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
       timeout=15.0
   )

   # Create registry with alternatives
   registry = ToolRegistry()
   registry.register(search, alternatives=["wikipedia"])
   registry.register(wikipedia)

   # Create agent
   agent = create_lrs_agent(
       llm=llm,
       tools=[search, wikipedia],
       tracker=tracker,
       preferences={
           'success': 5.0,
           'error': -3.0,
           'step_cost': -0.1
       }
   )

   # Execute task
   result = agent.invoke({
       'messages': [{
           'role': 'user',
           'content': 'Research the latest developments in quantum computing'
       }],
       'max_iterations': 20
   })

   # Analyze results
   print(f"Steps: {len(result['tool_history'])}")
   print(f"Adaptations: {result.get('adaptation_count', 0)}")
   print(f"\nTool usage:")
   
   for tool_name, stats in tracker.get_tool_usage_stats().items():
       print(f"  {tool_name}: {stats['calls']} calls, "
             f"{stats['success_rate']:.1%} success rate")

Troubleshooting
---------------

Tool Not Working
^^^^^^^^^^^^^^^^

Check that:

.. code-block:: python

   # Tool is properly wrapped
   lrs_tool = wrap_langchain_tool(langchain_tool)
   
   # Tool is registered (if using registry)
   registry.register(lrs_tool)
   
   # Tool has correct input/output schemas
   print(lrs_tool.input_schema)
   print(lrs_tool.output_schema)

High Prediction Errors
^^^^^^^^^^^^^^^^^^^^^^

If all tools have high prediction errors:

.. code-block:: python

   # Provide custom error function
   def better_error_fn(result, schema):
       # Your domain-specific logic
       return calculated_error

   tool = LangChainToolLens(lc_tool, error_fn=better_error_fn)

Agent Not Adapting
^^^^^^^^^^^^^^^^^^

Ensure prediction errors are in the right range:

.. code-block:: python

   # Test tool error calculation
   result = tool.get(test_state)
   print(f"Prediction error: {result.prediction_error}")
   
   # Should be:
   # - 0.0-0.2 for expected successes
   # - 0.6-0.9 for failures

Next Steps
----------

* Try the :doc:`../tutorials/05_llm_integration` tutorial
* Read about :doc:`openai_assistants` integration
* Explore the :doc:`../api/integration` API reference
* See :doc:`production_deployment` for scaling

