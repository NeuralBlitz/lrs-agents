Core API
========

The core module provides the fundamental building blocks for LRS agents: precision tracking, free energy calculation, tool lenses, and the tool registry.

Precision Tracking
------------------

.. automodule:: lrs.core.precision
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.core.precision.PrecisionParameters
   :members:
   :special-members: __init__
   :show-inheritance:

   Bayesian precision tracking using Beta distribution.

   The precision parameter γ (gamma) represents the agent's confidence in its world model.
   It is tracked using a Beta distribution with parameters α (alpha) and β (beta).

   **Key Methods:**

   .. automethod:: update
   .. automethod:: reset

   **Properties:**

   .. autoproperty:: value
   .. autoproperty:: variance

   **Example:**

   .. code-block:: python

      from lrs.core.precision import PrecisionParameters

      # Initialize with default priors
      precision = PrecisionParameters()
      
      # Update with low prediction error (success)
      new_value = precision.update(prediction_error=0.1)
      # Precision increases: 0.5 → 0.55
      
      # Update with high prediction error (failure)
      new_value = precision.update(prediction_error=0.9)
      # Precision decreases: 0.55 → 0.45

.. autoclass:: lrs.core.precision.HierarchicalPrecision
   :members:
   :special-members: __init__
   :show-inheritance:

   Multi-level precision tracking across hierarchical belief states.

   Tracks precision at three levels:
   
   * **Abstract**: High-level goals and strategies
   * **Planning**: Action sequences and policies
   * **Execution**: Individual tool executions

   Errors propagate upward when they exceed a threshold, with attenuation at each level.

   **Key Methods:**

   .. automethod:: update
   .. automethod:: get_level
   .. automethod:: get_all
   .. automethod:: reset

   **Example:**

   .. code-block:: python

      from lrs.core.precision import HierarchicalPrecision

      hp = HierarchicalPrecision()
      
      # High error at execution level
      updated = hp.update('execution', prediction_error=0.95)
      
      # Error propagates to planning if above threshold
      if 'planning' in updated:
          print("Planning precision also updated!")

Free Energy
-----------

.. automodule:: lrs.core.free_energy
   :members:
   :undoc-members:
   :show-inheritance:

Functions
^^^^^^^^^

.. autofunction:: lrs.core.free_energy.calculate_epistemic_value

   Calculate epistemic value (information gain) of a policy.

   Higher epistemic value indicates the policy will reduce uncertainty about the world.
   Novel tools or uncertain outcomes have high epistemic value.

   :param policy: Sequence of tools to execute
   :param state: Current belief state
   :param historical_stats: Historical tool performance statistics
   :return: Epistemic value (0 to ~2, higher = more information gain)

   **Example:**

   .. code-block:: python

      from lrs.core.free_energy import calculate_epistemic_value

      epistemic = calculate_epistemic_value(
          policy=[novel_tool, uncertain_tool],
          state={},
          historical_stats=None  # No history = high uncertainty
      )
      # Returns ~1.5 (high information gain)

.. autofunction:: lrs.core.free_energy.calculate_pragmatic_value

   Calculate pragmatic value (expected reward) of a policy.

   Higher pragmatic value indicates the policy is likely to achieve​​​​​​​​​​​​​​​​

