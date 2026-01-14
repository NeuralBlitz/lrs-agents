OpenAI Assistants Integration
==============================

Use OpenAI Assistants API with LRS-Agents for policy generation and tool execution.

Overview
--------

OpenAI Assistants provide:

* Built-in tools (Code Interpreter, File Search, Function Calling)
* Persistent threads and message history
* Automatic retry logic

LRS-Agents adds:

* Precision-adaptive behavior
* Automatic adaptation when tools fail
* Hierarchical belief tracking

Quick Start
-----------

Basic Setup
^^^^^^^^^^^

.. code-block:: python

   from openai import OpenAI
   from lrs.integration.openai_assistants import (
       OpenAIAssistantPolicyGenerator,
       create_openai_lrs_agent
   )

   # Initialize OpenAI client
   client = OpenAI(api_key="sk-...")

   # Create LRS agent with OpenAI Assistants
   agent = create_openai_lrs_agent(
       client=client,
       model="gpt-4-turbo-preview",
       tools=[...],  # Your LRS tools
       preferences={'success': 5.0, 'error': -3.0}
   )

   # Run task
   result = agent.run(
       task="Analyze the uploaded dataset",
       max_iterations=20
   )

Using Assistants for Policy Generation
---------------------------------------

Basic Policy Generator
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from openai import OpenAI
   from lrs.integration.openai_assistants import OpenAIAssistantPolicyGenerator
   from lrs.core.registry import ToolRegistry

   client = OpenAI(api_key="sk-...")
   registry = ToolRegistry()
   
   # Register your tools
   registry.register(fetch_tool)
   registry.register(process_tool)
   registry.register(save_tool)

   # Create generator
   generator = OpenAIAssistantPolicyGenerator(
       client=client,
       model="gpt-4-turbo-preview",
       registry=registry
   )

   # Generate proposals
   proposals = generator.generate_proposals(
       state={'goal': 'Fetch and process data'},
       precision=0.5
   )

   # Proposals are automatically converted to LRS policies
   for proposal in proposals:
       print(f"Strategy: {proposal['strategy']}")
       print(f"Tools: {proposal['tool_names']}")
       print(f"Rationale: {proposal['rationale']}")

Custom Assistant
^^^^^^^^^^^^^^^^

Create a custom assistant with specific instructions:

.. code-block:: python

   from openai import OpenAI
   from lrs.integration.openai_assistants import OpenAIAssistantLens

   client = OpenAI(api_key="sk-...")

   # Create assistant
   assistant = client.beta.assistants.create(
       name="Data Analyst",
       instructions="""You are a data analysis assistant.
       When given a task, propose 3-5 different strategies.
       Consider both quick approaches and thorough analyses.""",
       model="gpt-4-turbo-preview",
       tools=[{"type": "code_interpreter"}]
   )

   # Wrap as LRS tool
   assistant_lens = OpenAIAssistantLens(
       client=client,
       assistant_id=assistant.id,
       temperature=0.7
   )

   # Use in policy generation
   result = assistant_lens.get({
       'query': 'Generate proposals for data analysis',
       'precision': 0.5,
       'available_tools': ['fetch', 'analyze', 'visualize']
   })

Precision-Adaptive Temperature
-------------------------------

Temperature automatically adjusts based on precision:

.. code-block:: python

   generator = OpenAIAssistantPolicyGenerator(client, model="gpt-4-turbo-preview")

   # Low precision → High temperature (explore)
   proposals_explore = generator.generate_proposals(
       state={'goal': 'Task'},
       precision=0.3  # Temperature ≈ 1.2
   )

   # High precision → Low temperature (exploit)
   proposals_exploit = generator.generate_proposals(
       state={'goal': 'Task'},
       precision=0.8  # Temperature ≈ 0.5
   )

The formula:

.. math::

   T = T_{base} \times \frac{1}{\gamma + 0.1}

where γ is precision and T_base is the base temperature (default 0.7).

Using Built-in Assistant Tools
-------------------------------

Code Interpreter
^^^^^^^^^^^^^^^^

Enable code execution in proposals:

.. code-block:: python

   assistant = client.beta.assistants.create(
       name="Code Analysis Agent",
       model="gpt-4-turbo-preview",
       tools=[{"type": "code_interpreter"}]
   )

   generator = OpenAIAssistantPolicyGenerator(
       client=client,
       assistant_id=assistant.id
   )

   # Assistant can now propose policies that use code execution
   proposals = generator.generate_proposals(
       state={'goal': 'Analyze CSV file', 'file_path': 'data.csv'},
       precision=0.5
   )

File Search
^^^^^^^^^^^

Enable file search capabilities:

.. code-block:: python

   # Upload files
   file = client.files.create(
       file=open("knowledge_base.pdf", "rb"),
       purpose="assistants"
   )

   # Create assistant with file search
   assistant = client.beta.assistants.create(
       name="Research Assistant",
       model="gpt-4-turbo-preview",
       tools=[{"type": "file_search"}],
       file_ids=[file.id]
   )

   # Assistant can now search uploaded files in proposals

Function Calling
^^^^^^^^^^^^^^^^

Define functions for the assistant to use:

.. code-block:: python

   functions = [
       {
           "type": "function",
           "function": {
               "name": "search_database",
               "description": "Search the customer database",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "query": {"type": "string"},
                       "limit": {"type": "integer"}
                   },
                   "required": ["query"]
               }
           }
       }
   ]

   assistant = client.beta.assistants.create(
       name="Database Agent",
       model="gpt-4-turbo-preview",
       tools=functions
   )

   # Implement function
   def search_database(query, limit=10):
       # Your implementation
       return results

   # Use with LRS
   generator = OpenAIAssistantPolicyGenerator(
       client=client,
       assistant_id=assistant.id
   )

Complete Agent Example
----------------------

Here's a complete example with file analysis:

.. code-block:: python

   from openai import OpenAI
   from lrs.integration.openai_assistants import create_openai_lrs_agent
   from lrs.core.lens import ToolLens, ExecutionResult
   from lrs.monitoring.tracker import LRSStateTracker

   # Initialize
   client = OpenAI(api_key="sk-...")
   tracker = LRSStateTracker()

   # Custom tool for saving results
   class SaveResultsTool(ToolLens):
       def __init__(self):
           super().__init__("save_results", {}, {})
       
       def get(self, state):
           self.call_count += 1
           results = state.get('analysis_results')
           if not results:
               self.failure_count += 1
               return ExecutionResult(False, None, "No results", 0.9)
           
           # Save to file
           with open('results.json', 'w') as f:
               json.dump(results, f)
           
           return ExecutionResult(True, "Saved", None, 0.05)
       
       def set(self, state, obs):
           return {**state, 'saved': True}

   # Create assistant
   assistant = client.beta.assistants.create(
       name="Data Analyst",
       instructions="""Analyze data files and provide insights.
       When proposing strategies, consider:
       1. Quick exploratory analysis
       2. Thorough statistical analysis
       3. Visualization-focused analysis""",
       model="gpt-4-turbo-preview",
       tools=[{"type": "code_interpreter"}]
   )

   # Create LRS agent
   agent = create_openai_lrs_agent(
       client=client,
       assistant_id=assistant.id,
       tools=[SaveResultsTool()],
       tracker=tracker,
       preferences={
           'success': 5.0,
           'error': -3.0,
           'step_cost': -0.2
       }
   )

   # Upload file
   file = client.files.create(
       file=open("sales_data.csv", "rb"),
       purpose="assistants"
   )

   # Run analysis
   result = agent.run(
       task=f"Analyze the sales data in file {file.id}",
       max_iterations=15
   )

   # Review results
   print(f"Steps: {len(result['tool_history'])}")
   print(f"Adaptations: {result.get('adaptation_count', 0)}")
   print(f"\nFinal precision: {result['precision']['execution']:.2f}")

Handling Long-Running Operations
---------------------------------

Assistants can take time to respond. Handle this gracefully:

.. code-block:: python

   from lrs.integration.openai_assistants import OpenAIAssistantLens

   assistant_lens = OpenAIAssistantLens(
       client=client,
       assistant_id=assistant.id,
       max_wait=300,  # Wait up to 5 minutes
       poll_interval=2.0  # Check every 2 seconds
   )

   # LRS will:
   # 1. Submit query to assistant
   # 2. Poll for completion
   # 3. Return high prediction error if timeout
   # 4. Trigger adaptation if precision drops

Error Handling
--------------

Rate Limits
^^^^^^^^^^^

OpenAI has rate limits. LRS handles these automatically:

.. code-block:: python

   # When rate limited:
   # - Assistant returns error
   # - LRS records high prediction error (0.9)
   # - Precision drops
   # - Agent explores alternatives
   # - Might wait and retry, or use different tools

.. code-block:: python

   from lrs.core.lens import ToolLens, ExecutionResult

   class RateLimitAwareTool(ToolLens):
       def get(self, state):
           try:
               result = openai_call(state)
               return ExecutionResult(True, result, None, 0.1)
           except openai.RateLimitError:
               # Expected but bad
               return ExecutionResult(False, None, "Rate limited", 0.7)
           except Exception as e:
               # Unexpected
               return ExecutionResult(False, None, str(e), 0.95)

Timeouts
^^^^^^^^

Handle assistant timeouts:

.. code-block:: python

   assistant_lens = OpenAIAssistantLens(
       client=client,
       assistant_id=assistant.id,
       max_wait=120  # Timeout after 2 minutes
   )

   # If assistant doesn't respond:
   # - Returns ExecutionResult with success=False
   # - High prediction error (0.9)
   # - Agent adapts and tries alternatives

Best Practices
--------------

1. Provide Clear Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   assistant = client.beta.assistants.create(
       name="Research Assistant",
       instructions="""You are a research assistant using Active Inference.
       
       When proposing policies:
       - Generate 3-5 diverse proposals
       - Include exploration strategies (novel tools)
       - Include exploitation strategies (reliable tools)
       - Balance information gain vs reward
       
       For each proposal, specify:
       - Tool sequence
       - Estimated success probability
       - Expected information gain
       - Potential failure modes""",
       model="gpt-4-turbo-preview"
   )

2. Monitor Performance
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lrs.monitoring.tracker import LRSStateTracker

   tracker = LRSStateTracker()
   agent = create_openai_lrs_agent(client, assistant_id, tracker=tracker)

   # After execution
   stats = tracker.get_tool_usage_stats()
   
   # Check if assistant is generating good proposals
   print(f"Assistant success rate: {stats['assistant']['success_rate']:.1%}")

3. Combine with Custom Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use assistant for policy generation
   # Use custom tools for execution
   
   agent = create_openai_lrs_agent(
       client=client,
       assistant_id=assistant.id,
       tools=[
           CustomCacheTool(),
           CustomDatabaseTool(),
           CustomAPITool()
       ]
   )

   # Assistant proposes, custom tools execute
   # Best of both worlds!

4. Set Appropriate Timeouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Quick tasks
   quick_assistant = OpenAIAssistantLens(
       client, assistant_id, max_wait=30
   )

   # Complex analyses
   analysis_assistant = OpenAIAssistantLens(
       client, assistant_id, max_wait=300
   )

Cost Optimization
-----------------

Minimize Costs
^^^^^^^^^^^^^^

.. code-block:: python

   # Use GPT-3.5 for simple tasks
   cheap_generator = OpenAIAssistantPolicyGenerator(
       client=client,
       model="gpt-3.5-turbo",  # Cheaper
       registry=registry
   )

   # Use GPT-4 for complex tasks
   smart_generator = OpenAIAssistantPolicyGenerator(
       client=client,
       model="gpt-4-turbo-preview",  # More expensive but better
       registry=registry
   )

   # Switch based on task complexity
   if task_complexity < 0.5:
       proposals = cheap_generator.generate_proposals(state, precision)
   else:
       proposals = smart_generator.generate_proposals(state, precision)

Cache Responses
^^^^^^^^^^^^^^^

.. code-block:: python

   from functools import lru_cache

   @lru_cache(maxsize=100)
   def cached_generate_proposals(state_hash, precision):
       return generator.generate_proposals(state, precision)

   # Use hash of state for caching
   import hashlib
   state_hash = hashlib.md5(str(state).encode()).hexdigest()
   proposals = cached_generate_proposals(state_hash, precision)

Troubleshooting
---------------

Assistant Not Responding
^^^^^^^^^^^^^^^^^^^^^^^^

Check:

.. code-block:: python

   # Verify assistant exists
   assistant = client.beta.assistants.retrieve(assistant_id)
   print(f"Assistant: {assistant.name}")

   # Check status
   run = client.beta.threads.runs.retrieve(thread_id, run_id)
   print(f"Status: {run.status}")

Poor Proposals
^^^^^^^^^^^^^^

Improve instructions:

.. code-block:: python

   # Update assistant instructions
   client.beta.assistants.update(
       assistant_id,
       instructions="""More detailed instructions here..."""
   )

High Costs
^^^^^^^^^^

Monitor usage:

.. code-block:: python

   # Track token usage
   from lrs.monitoring.structured_logging import create_logger_for_agent

   logger = create_logger_for_agent("agent_1")
   
   # Log costs
   logger.log_custom({
       'event': 'assistant_call',
       'tokens': run.usage.total_tokens,
       'cost_estimate': run.usage.total_tokens * 0.00001
   })

Next Steps
----------

* Try the :doc:`../tutorials/05_llm_integration` tutorial
* Read about :doc:`langchain_integration`
* See :doc:`autogpt_integration` for research agents
* Explore :doc:`production_deployment` for scaling

