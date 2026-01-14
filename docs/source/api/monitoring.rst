Monitoring API
==============

The monitoring module provides state tracking, visualization, and logging.

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
   :show-inheritance:

   Snapshot of agent state at a point in time.

   **Attributes:**

   * **timestamp** (datetime): When snapshot was taken
   * **precision** (dict): Precision values by level
   * **prediction_errors** (List[float]): Recent prediction errors
   * **tool_history** (List[dict]): Tool execution history
   * **adaptation_count** (int): Number of adaptations so far
   * **belief_state** (dict): Current belief state

.. autoclass:: lrs.monitoring.tracker.LRSStateTracker
   :members:
   :special-members: __init__
   :show-inheritance:

   Tracks agent state over time for analysis.

   **Methods:**

   .. automethod:: track_state
   .. automethod:: get_precision_trajectory
   .. automethod:: get_all_precision_trajectories
   .. automethod:: get_prediction_errors
   .. automethod:: get_adaptation_events
   .. automethod:: get_tool_usage_stats
   .. automethod:: get_summary
   .. automethod:: export_history
   .. automethod:: clear

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
      #   },
      #   ...
      # }
      
      # Export for analysis
      tracker.export_history('execution_history.json')

Dashboard
---------

.. automodule:: lrs.monitoring.dashboard
   :members:
   :undoc-members:
   :show-inheritance:

Functions
^^^^^^^^^

.. autofunction:: lrs.monitoring.dashboard.create_dashboard

   Create Streamlit dashboard for real-time monitoring.

   **Features:**

   * Precision trajectories (execution, planning, abstract)
   * Prediction error stream
   * Tool usage statistics and reliability
   * Adaptation timeline
   * Execution history

   **Example:**

   .. code-block:: python

      from lrs.monitoring.dashboard import create_dashboard
      from lrs.monitoring.tracker import LRSStateTracker

      tracker = LRSStateTracker()
      
      # In separate thread/process
      create_dashboard(tracker)
      
      # Dashboard runs at http://localhost:8501

.. autofunction:: lrs.monitoring.dashboard.run_dashboard

   Run dashboard as standalone application.

   .. code-block:: bash

      # Command line
      python -m lrs.monitoring.dashboard

Structured Logging
------------------

.. automodule:: lrs.monitoring.structured_logging
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.monitoring.structured_logging.LRSLogger
   :members:
   :special-members: __init__
   :show-inheritance:

   Structured JSON logger for production.

   **Methods:**

   .. automethod:: log_precision_update
   .. automethod:: log_policy_selection
   .. automethod:: log_tool_execution
   .. automethod:: log_adaptation_event
   .. automethod:: log_performance_metrics
   .. automethod:: log_error

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
          error_message="Timeout"
      )
      
      # Log adaptation
      logger.log_adaptation_event(
          trigger="High prediction error",
          old_precision=0.6,
          new_precision=0.4,
          action="Switch to cache_fetch"
      )
      
      # Log metrics
      logger.log_performance_metrics(
          total_steps=50,
          success_rate=0.8,
          avg_precision=0.65,
          adaptation_count=3,
          execution_time=125.3
      )

Functions
^^^^^^^^^

.. autofunction:: lrs.monitoring.structured_logging.create_logger_for_agent

   Create configured logger for an agent.

   :param agent_id: Unique agent identifier
   :param log_file: Path to log file (JSONL format)
   :param console: Also log to console
   :param level: Logging level (INFO, DEBUG, etc.)
   :return: Configured LRSLogger

