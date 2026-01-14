Quickstart
==========

This guide will get you up and running with LRS-Agents in 5 minutes.

Overview
--------

We'll build a simple agent that:

1. Tries to fetch data from an API
2. Automatically falls back to cache when the API fails
3. Adapts its strategy based on precision

Step 1: Install LRS-Agents
---------------------------

.. code-block:: bash

   pip install lrs-agents langchain-anthropic

Step 2: Set API Key
-------------------

.. code-block:: bash

   export ANTHROPIC_API_KEY="sk-ant-api03-..."

Step 3: Create Your First Tool
-------------------------------

.. code-block:: python

   from lrs.core.lens import ToolLens, ExecutionResult
   import random

   class APITool(ToolLens):
       """Fetch from API (sometimes fails)"""
       
       def __init__(self):
           super().__init__(name="fetch_api", input_schema={}, output_schema={})
       
       def get(self, state):
           self.call_count += 1
           
           # Simulate 30% failure rate
           if random.random() < 0.3:
               self.failure_count += 1
               return ExecutionResult(
                   success=False,
                   value=None,
                   error="API timeout",
                   prediction_error=0.9  # High surprise
               )
           
           return ExecutionResult(
               success=True,
               value={"data": "from_api"},
               error=None,
               prediction_error=0.1  # Low surprise
           )
       
       def set(self, state, observation):
           return {**state, 'data': observation}

   class CacheTool(ToolLens):
       """Fetch from cache (always works)"""
       
       def __init__(self):
           super().__init__(name="fetch_cache", input_schema={}, output_schema={})
       
       def get(self, state):
           self.call_count += 1
           return ExecutionResult(
               success=True,
               value={"data": "from_cache"},
               error=None,
               prediction_error=0.05  # Very predictable
           )
       
       def set(self, state, observation):
           return {**state, 'data': observation}

Step 4: Create an LRS Agent
----------------------------

.. code-block:: python

   from langchain_anthropic import ChatAnthropic
   from lrs import create_lrs_agent

   # Initialize LLM
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")

   # Create tools
   tools = [APITool(), CacheTool()]

   # Create agent
   agent = create_lrs_agent(
       llm=llm,
       tools=tools,
       preferences={
           'success': 5.0,    # Reward for success
           'error': -3.0,     # Penalty for errors
           'step_cost': -0.1  # Small cost per step
       }
   )

Step 5: Run a Task
------------------

.. code-block:: python

   result = agent.invoke({
       'messages': [{
           'role': 'user',
           'content': 'Fetch the data'
       }],
       'belief_state': {'goal': 'fetch_data'},
       'max_iterations': 10
   })

   print(f"Success: {result['belief_state'].get('data') is not None}")
   print(f"Steps taken: {len(result['tool_history'])}")
   print(f"Adaptations: {result.get('adaptation_count', 0)}")

Understanding the Output
------------------------

The agent will:

1. **Try the API first** (high reward if successful)
2. **Detect failure** (prediction error = 0.9)
3. **Precision drops** (0.5 → 0.4)
4. **Adaptation triggered** (precision < 0.4)
5. **Explore alternatives** (tries cache)
6. **Cache succeeds** (prediction error = 0.05)
7. **Precision recovers** (0.4 → 0.5)

Example Output
^^^^^^^^^^^^^^

.. code-block:: text

   Execution trace:
   1. ✗ fetch_api (error: 0.90)  [precision: 0.50 → 0.40]
   2. ✓ fetch_cache (error: 0.05) [precision: 0.40 → 0.50]
   
   Success: True
   Steps taken: 2
   Adaptations: 1

What Just Happened?
-------------------

The agent automatically:

* **Tracked precision**: Confidence in its world model
* **Detected surprise**: High prediction error from API failure
* **Adapted strategy**: Switched to cache when precision dropped
* **No manual error handling**: The framework handled everything

Key Concepts
------------

Precision (γ)
^^^^^^^^^^^^^

Precision represents the agent's confidence. It:

* Starts at 0.5 (neutral)
* Increases with successful predictions
* Decreases with surprises
* Triggers adaptation when low

Prediction Error
^^^^^^^^^^^^^^^^

Each tool execution returns a prediction error:

* **0.0-0.2**: Expected (deterministic operations)
* **0.3-0.5**: Medium surprise
* **0.6-0.8**: High surprise
* **0.9-1.0**: Very unexpected

Expected Free Energy (G)
^^^^^^^^^^^^^^^^^^^^^^^^

The agent selects policies by minimizing G:

.. math::

   G = \text{Epistemic Value} - \text{Pragmatic Value}

* **Epistemic**: Information gain (exploration)
* **Pragmatic**: Expected reward (exploitation)

Adaptation
^^^^^^^^^^

When precision drops below threshold (default 0.4):

1. Agent explores alternative tools
2. Temperature increases (more random selection)
3. Favors high epistemic value (information gain)

Next Steps
----------

Now that you have a basic agent working:

1. **Understand the theory**: Read :doc:`core_concepts`
2. **Build complex agents**: See :doc:`../tutorials/03_tool_composition`
3. **Add monitoring**: Learn about :doc:`../tutorials/06_monitoring_dashboard`
4. **Deploy to production**: Follow :doc:`../guides/production_deployment`

Common Patterns
---------------

Multiple Alternatives
^^^^^^^^^^^^^^^^^^^^^

Register alternatives in the registry:

.. code-block:: python

   from lrs.core.registry import ToolRegistry

   registry = ToolRegistry()
   registry.register(APITool(), alternatives=["fetch_cache", "fetch_db"])
   registry.register(CacheTool())
   registry.register(DBTool())

   # Agent automatically tries alternatives when primary tool fails

Tool Composition
^^^^^^^^^^^^^^^^

Compose tools into pipelines:

.. code-block:: python

   pipeline = FetchTool() >> ParseTool() >> ExtractTool()
   
   # Executes sequentially, short-circuits on failure

Monitoring
^^^^^^^^^^

Add state tracking:

.. code-block:: python

   from lrs.monitoring.tracker import LRSStateTracker

   tracker = LRSStateTracker()
   
   agent = create_lrs_agent(llm, tools, tracker=tracker)
   
   # After execution
   print(f"Tool stats: {tracker.get_tool_usage_stats()}")
   print(f"Precision trajectory: {tracker.get_precision_trajectory('execution')}")

Troubleshooting
---------------

Agent Always Fails
^^^^^^^^^^^^^^^^^^

Check that:

* Tools are registered correctly
* Prediction errors are in [0, 1]
* Preferences are reasonable (positive for success)

No Adaptation
^^^^^^^^^^^^^

Ensure:

* Prediction errors are high enough (>0.7 for failures)
* Adaptation threshold is appropriate (default 0.4)
* Precision is being updated

High Step Count
^^^^^^^^^^^^^^^

If agent takes too many steps:

* Increase step cost penalty
* Add more informative tools
* Ensure tools return accurate prediction errors

Further Reading
---------------

* :doc:`core_concepts` - Deep dive into Active Inference
* :doc:`../tutorials/02_understanding_precision` - Precision mechanics
* :doc:`../tutorials/04_chaos_scriptorium` - Resilience testing
* :doc:`../api/core` - Complete API reference

