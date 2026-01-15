Monitoring API
==============

The monitoring module provides state tracking, visualization, and logging for LRS agents.

State Tracking
--------------

.. automodule:: lrs.monitoring.tracker
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.monitoring.tracker.StateSnapshot
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   Snapshot of agent state at a point in time.

   **Attributes:**

   * **timestamp** (datetime): When snapshot was taken
   * **precision** (Dict[str, float]): Precision values by level
   * **prediction_errors** (List[float]): Recent prediction errors
   * **tool_history** (List[Dict]): Tool execution history
   * **adaptation_count** (int): Number of adaptations so far
   * **belief_state** (Dict): Current belief state

.. autoclass:: lrs.monitoring.tracker.LRSStateTracker
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:
   :no-index:

   Tracks agent state over time for analysis and visualization.

   **Methods:**

   .. automethod:: track_state
      
      Track current agent state.

   .. automethod:: get_precision_trajectory
      
      Get precision history for a specific level.

   .. automethod:: get_all_precision_trajectories
      
      Get precision histories for all levels.

   .. automethod:: get_prediction_errors
      
      Get recent prediction errors.

   .. automethod:: get_adaptation_events
      
      Get timeline of adaptation events.

   .. automethod:: get_tool_usage_stats
      
      Get tool usage statistics and reliability.

   .. automethod:: get_summary
      
      Get summary statistics.

   .. automethod:: export_history
      
      Export history to JSON file.

   .. automethod:: clear
      
      Clear all tracked history.

   **Example:**

   .. code-block:: python

      from lrs.monitoring.tracker import LRSStateTracker

      tracker = LRSStateTracker(max_history=1000)
      
      # Track state after each step
      tracker.track_state({
          'precision': {'execution': 0.7, 'planning': 0.6},
          'tool_history': [...],
          'adaptation_count': 2,
          'belief_state': {}
      })
      
      # Analyze precision trajectory
      exec_prec = tracker.get_precision_trajectory('execution')
      # Returns: [0.5, 0.6, 0.45, 0.55, 0.7, ...]
      
      # Get tool statistics
      stats = tracker.get_tool_usage_stats()
      # Returns: {
      #   'fetch_api': {
      #     'calls': 10,
      #     'successes': 7,
      #     'failures': 3,
      #     'success_rate': 0.7,
      #     'avg_error': 0.35
      #   }
      # }
      
      # Export for analysis
      tracker.export_history('execution_history.json')

Structured Logging
------------------

.. automodule:: lrs.monitoring.structured_logging
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Classes
^^^^^^^

.. autoclass:: lrs.monitoring.structured_logging.LRSLogger
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:
   :no-index:

   Structured JSON logger for production monitoring.

   Logs all agent events as structured JSON for integration with
   monitoring systems like ELK, Datadog, CloudWatch, or Grafana.

   **Methods:**

   .. automethod:: log_precision_update
      
      Log precision parameter update.

   .. automethod:: log_policy_selection
      
      Log policy selection and G values.

   .. automethod:: log_tool_execution
      
      Log tool execution result.

   .. automethod:: log_adaptation_event
      
      Log adaptation trigger and response.

   .. automethod:: log_performance_metrics
      
      Log aggregate performance metrics.

   .. automethod:: log_error
      
      Log error or exception.

   **Example:**

   .. code-block:: python

      from lrs.monitoring.structured_logging import create_logger_for_agent

      logger = create_logger_for_agent(
          agent_id="production_agent_1",
          log_file="logs/agent.jsonl",
          console=True
      )
      
      # Log tool execution
      logger.log_tool_execution(
          tool_name="fetch_api",
          success=False,
          execution_time=150.5,
          prediction_error=0.9,
          error_message="Timeout after 30s"
      )
      
      # Log adaptation
      logger.log_adaptation_event(
          trigger="precision_threshold",
          old_precision=0.6,
          new_precision=0.4,
          action="explore_alternatives"
      )
      
      # Log performance metrics
      logger.log_performance_metrics(
          total_steps=50,
          success_rate=0.8,
          avg_precision=0.65,
          adaptation_count=3,
          total_time=125.3
      )

Functions
^^^^^^^^^

.. autofunction:: lrs.monitoring.structured_logging.create_logger_for_agent
   :no-index:

   Create configured logger for an agent.

   :param agent_id: Unique agent identifier
   :param log_file: Path to log file (JSONL format)
   :param console: Also log to console (default: False)
   :param level: Logging level (default: INFO)
   :return: Configured LRSLogger instance

   **Example:**

   .. code-block:: python

      logger = create_logger_for_agent(
          agent_id="my_agent",
          log_file="/var/log/lrs/agent.jsonl",
          console=True,
          level="DEBUG"
      )

Dashboard (Optional)
--------------------

.. note::
   The dashboard module requires ``streamlit`` and ``matplotlib`` which are optional dependencies.
   Install with: ``pip install lrs-agents[monitoring]``

.. automodule:: lrs.monitoring.dashboard
   :members:
   :undoc-members:
   :show-inheritance:
   :ignore-module-all:

The dashboard provides real-time visualization of agent execution including:

* **Precision trajectories** - Execution, planning, and abstract levels over time
* **Prediction error stream** - Real-time error monitoring
* **Tool usage statistics** - Success rates and reliability by tool
* **Adaptation timeline** - When and why adaptations occurred
* **Execution history** - Complete step-by-step trace

**Example Usage:**

.. code-block:: python

   from lrs.monitoring.dashboard import run_dashboard
   from lrs.monitoring.tracker import LRSStateTracker

   tracker = LRSStateTracker()
   
   # Run dashboard (blocks)
   run_dashboard(tracker, port=8501)
   # Dashboard available at http://localhost:8501

**Command Line:**

.. code-block:: bash

   # Start dashboard
   python -m lrs.monitoring.dashboard --port 8501
   
   # Or with Streamlit directly
   streamlit run lrs/monitoring/dashboard.py
