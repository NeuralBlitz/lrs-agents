LRS-Agents Documentation
========================

**LRS-Agents** is a Python framework for building resilient AI agents using Active Inference and Predictive Processing. Agents automatically adapt when tools fail, tracking precision (confidence) across hierarchical levels and exploring alternatives when surprises occur.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/core_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/langchain_integration
   guides/openai_assistants
   guides/autogpt_integration
   guides/production_deployment

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/inference
   api/integration
   api/monitoring

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/active_inference
   theory/free_energy
   theory/precision_dynamics

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_quickstart
   tutorials/02_understanding_precision
   tutorials/03_tool_composition
   tutorials/04_chaos_scriptorium

Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
--------

LRS-Agents provides:

* **Automatic Adaptation**: Agents adapt when tools fail—no manual error handling needed
* **Precision Tracking**: Hierarchical confidence tracking (abstract, planning, execution)
* **Expected Free Energy**: Mathematical framework for policy selection
* **LLM Integration**: Use LLMs as variational proposal mechanisms
* **Multi-Agent Support**: Social precision tracking and coordination

Key Features
------------

🎯 **Automatic Adaptation**
   Agents detect surprises and adapt strategies automatically

🧠 **Hierarchical Precision**
   Track confidence at multiple levels (abstract, planning, execution)

⚡ **Expected Free Energy**
   Balance exploration (information gain) vs exploitation (reward)

🤖 **LLM Integration**
   Use Claude, GPT-4, or any LLM for policy generation

🔧 **Framework Agnostic**
   Works with LangChain, OpenAI Assistants, AutoGPT

📊 **Built-in Monitoring**
   Streamlit dashboard, structured logging, state tracking

Installation
------------

.. code-block:: bash

   pip install lrs-agents

Quick Example
-------------

.. code-block:: python

   from langchain_anthropic import ChatAnthropic
   from lrs import create_lrs_agent
   from lrs.core.lens import ToolLens, ExecutionResult

   # Define a tool
   class WeatherTool(ToolLens):
       def get(self, state):
           # Tool implementation
           return ExecutionResult(True, "sunny", None, 0.1)
       
       def set(self, state, obs):
           return {**state, 'weather': obs}

   # Create agent
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
   agent = create_lrs_agent(llm, [WeatherTool()])

   # Run task
   result = agent.invoke({
       'messages': [{'role': 'user', 'content': 'Get weather'}],
       'max_iterations': 10
   })

Why LRS-Agents?
---------------

Traditional agents fail when tools break. They need manual error handling, retry logic, and fallback strategies. LRS-Agents solve this through **Active Inference**:

1. **Track Precision**: Agents maintain confidence in their world model
2. **Detect Surprises**: High prediction errors signal problems
3. **Adapt Automatically**: Low precision triggers exploration of alternatives
4. **No Manual Handling**: The framework handles failures automatically

Learn More
----------

* Read the :doc:`getting_started/quickstart` guide
* Understand :doc:`getting_started/core_concepts`
* Explore :doc:`tutorials/01_quickstart`
* Check out the `GitHub repository <https://github.com/YourOrg/lrs-agents>`_

Citation
--------

If you use LRS-Agents in your research, please cite:

.. code-block:: bibtex

   @software{lrs_agents,
     title = {LRS-Agents: Resilient AI Agents via Active Inference},
     author = {LRS-Agents Contributors},
     year = {2024},
     url = {https://github.com/YourOrg/lrs-agents}
   }

License
-------

LRS-Agents is released under the MIT License. See the LICENSE file for details.

