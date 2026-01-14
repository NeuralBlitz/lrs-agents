Installation
============

LRS-Agents requires Python 3.9 or later.

Quick Install
-------------

Install from PyPI:

.. code-block:: bash

   pip install lrs-agents

This installs the core package with minimal dependencies.

Install with Extras
-------------------

LRS-Agents provides several optional dependency groups:

All Features
^^^^^^^^^^^^

Install everything:

.. code-block:: bash

   pip install lrs-agents[all]

Specific Features
^^^^^^^^^^^^^^^^^

Install only what you need:

.. code-block:: bash

   # LangChain integration
   pip install lrs-agents[langchain]

   # OpenAI Assistants
   pip install lrs-agents[openai]

   # Monitoring dashboard
   pip install lrs-agents[dashboard]

   # Development tools
   pip install lrs-agents[dev]

   # Documentation building
   pip install lrs-agents[docs]

   # Combine multiple extras
   pip install lrs-agents[langchain,openai,dashboard]

Development Installation
------------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/YourOrg/lrs-agents.git
   cd lrs-agents
   pip install -e ".[dev,test]"

This installs the package in development mode with all development an​​​​​​​​​​​​​​​​

