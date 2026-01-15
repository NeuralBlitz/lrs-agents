Installation
============

LRS-Agents can be installed via pip or from source.

Requirements
------------

* Python 3.9+
* pip or conda

Basic Installation
------------------

Install the latest stable version from PyPI:

.. code-block:: bash

   pip install lrs-agents

This installs the core package with minimal dependencies.

Installation with Optional Dependencies
----------------------------------------

LRS-Agents has several optional dependency groups for different use cases:

All Features
^^^^^^^^^^^^

Install everything:

.. code-block:: bash

   pip install lrs-agents[all]

LangChain Integration
^^^^^^^^^^^^^^^^^^^^^

For LangChain tools and agents:

.. code-block:: bash

   pip install lrs-agents[langchain]

This includes:

* ``langchain``
* ``langchain-anthropic``
* ``langchain-openai``
* ``langchain-community``

OpenAI Integration
^^^^^^^^^^^^^^^^^^

For OpenAI Assistants:

.. code-block:: bash

   pip install lrs-agents[openai]

Monitoring & Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For dashboards and visualizations:

.. code-block:: bash

   pip install lrs-agents[monitoring]

This includes:

* ``streamlit``
* ``plotly``
* ``matplotlib``

Development
^^^^^^^^^^^

For contributing to LRS-Agents:

.. code-block:: bash

   pip install lrs-agents[dev]

This includes linting, formatting, and testing tools.

Installation from Source
-------------------------

For the latest development version:

.. code-block:: bash

   git clone https://github.com/NeuralBlitz/lrs-agents.git
   cd lrs-agents
   pip install -e .

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

To contribute to LRS-Agents:

.. code-block:: bash

   git clone https://github.com/NeuralBlitz/lrs-agents.git
   cd lrs-agents
   pip install -e ".[dev,test]"

Verify Installation
-------------------

Check that LRS-Agents is installed correctly:

.. code-block:: python

   import lrs
   print(lrs.__version__)
   # Output: 0.2.0

Quick Test
^^^^^^^^^^

Run a simple test:

.. code-block:: python

   from lrs.core.precision import PrecisionParameters

   precision = PrecisionParameters()
   print(f"Initial precision: {precision.value}")
   
   precision.update(0.1)  # Low error
   print(f"After success: {precision.value}")

Configuration
-------------

API Keys
^^^^^^^^

LRS-Agents requires API keys for LLM providers. Set them as environment variables:

.. code-block:: bash

   # Anthropic (Claude)
   export ANTHROPIC_API_KEY="sk-ant-api03-..."

   # OpenAI (GPT-4)
   export OPENAI_API_KEY="sk-..."

Or in Python:

.. code-block:: python

   import os
   os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-..."

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Optional configuration:

.. code-block:: bash

   # Logging
   export LRS_LOG_LEVEL="INFO"
   export LRS_LOG_DIR="./logs"

   # Database (optional)
   export DATABASE_URL="postgresql://user:pass@localhost/lrs"

   # Performance
   export LRS_MAX_WORKERS="4"
   export LRS_CACHE_ENABLED="true"

Docker Installation
-------------------

Run LRS-Agents in Docker:

.. code-block:: bash

   docker pull lrsagents/lrs-agents:latest
   docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY lrsagents/lrs-agents

Or with Docker Compose:

.. code-block:: bash

   cd docker
   docker-compose up -d

This starts:

* LRS-Agents API server (port 8000)
* Streamlit dashboard (port 8501)
* PostgreSQL database (port 5432)

Kubernetes Deployment
---------------------

Deploy to Kubernetes:

.. code-block:: bash

   kubectl create namespace lrs-agents
   kubectl apply -f k8s/

See the :doc:`../guides/production_deployment` guide for details.

Troubleshooting
---------------

Import Errors
^^^^^^^^^^^^^

If you get import errors:

.. code-block:: bash

   # Ensure package is installed
   pip list | grep lrs-agents

   # Reinstall if needed
   pip install --force-reinstall lrs-agents

Missing Dependencies
^^^^^^^^^^^^^^^^^^^^

If specific features don't work:

.. code-block:: bash

   # Install all optional dependencies
   pip install lrs-agents[all]

Version Conflicts
^^^^^^^^^^^^^^^^^

If you have dependency conflicts:

.. code-block:: bash

   # Create fresh virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install lrs-agents[all]

GPU Support
^^^^^^^^^^^

For GPU-accelerated inference (optional):

.. code-block:: bash

   # Install PyTorch with CUDA
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # Then install LRS-Agents
   pip install lrs-agents

Getting Help
------------

If you encounter issues:

* Check the `GitHub Issues <https://github.com/NeuralBlitz/lrs-agents/issues>`_
* Join our `Hugging Face community <https://huggingface.co/NuralNexus>`_
* Join our `Discord community <https://huggingface.co/NuralNexus>`_
* Email nuralnexus@icloud.com

Next Steps
----------

* Read the :doc:`quickstart` guide
* Understand :doc:`core_concepts`
* Explore :doc:`../tutorials/01_quickstart`
