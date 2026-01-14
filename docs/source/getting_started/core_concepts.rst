Core Concepts
=============

LRS-Agents is built on principles from Active Inference and Predictive Processing. This guide explains the key concepts.

Active Inference
----------------

What is Active Inference?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Active Inference is a theory from neuroscience that explains how biological agents (like humans) make decisions. The core idea:

   **Agents act to minimize surprises about their sensory observations.**

In LRS-Agents, this means:

* **Precision** (γ): Confidence in predictions
* **Prediction Errors**: Surprises that update beliefs
* **Free Energy**: A measure of surprise to minimize

Why Active Inference for AI?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Traditional agents:

* Need explicit error handling
* Require manual fallback strategies
* Don't learn from failures

Active Inference agents:

* **Adapt automatically** when surprised
* **Balance exploration vs exploitation** naturally
* **Track uncertainty** explicitly

The Free Energy Principle
--------------------------

Expected Free Energy (G)
^^^^^^^^^^^^^^^^^^^^^^^^

LRS agents select actions by minimizing Expected Free Energy:

.. math::

   G = \text{Epistemic Value} - \text{Pragmatic Value}

Where:

* **Epistemic Value**: Information gain (reduces uncertainty)
* **Pragmatic Value**: Expected reward (achieves goals)

Low G = Good Policy
^^^^^^^^^^^^^^^^^^^

Policies with low G are preferred because they:

1. Reduce uncertainty about the world
2. Achieve desired outcomes
3. Balance exploration and exploitation

Example Calculation
^^^^^^^^^^^^^^^^^^^

Consider two policies:

**Policy A**: Try familiar tool

* Epistemic value: 0.2 (low uncertainty)
* Pragmatic value: 4.0 (high success rate)
* **G = 0.2 - 4.0 = -3.8** ✓ (Good!)

**Policy B**: Try novel tool

* Epistemic value: 1.5 (high uncertainty)
* Pragmatic value: 2.0 (unknown success rate)
* **G = 1.5 - 2.0 = -0.5** (Less good)

When precision is high, agent chooses Policy A (exploit).
When precision is low, agent might explore Policy B.

Precision Tracking
------------------

What is Precision?
^^^^^^^^^^^^^^^^^^

Precision (γ) represents the agent's confidence in its world model:

* **High precision** (γ > 0.7): Agent is confident, exploits knowledge
* **Medium precision** (γ ≈ 0.5): Agent balances explore/exploit
* **Low precision** (γ < 0.4): Agent is uncertain, explores

How Precision Updates
^^^^^^^^^^^^^^^^^^^^^

Precision is modeled as a Beta distribution with parameters α and β:

.. math::

   \gamma = \frac{\alpha}{\alpha + \beta}

Updates are asymmetric:

* **Success** (low prediction error): Slow increase
  
  .. math::
  
     \alpha \leftarrow \alpha + \eta_{gain} \cdot (1 - \delta)

* **Failure** (high prediction error): Fast decrease
  
  .. math::
  
     \beta \leftarrow \beta + \eta_{loss} \cdot \delta

where δ is the prediction error.

Example
^^^^^^^

.. code-block:: python

   from lrs.core.precision import PrecisionParameters

   precision = PrecisionParameters()
   print(f"Initial: {precision.value}")  # 0.5

   # Success
   precision.update(0.1)
   print(f"After success: {precision.value}")  # 0.52 (slight increase)

   # Failure
   precision.update(0.9)
   print(f"After failure: {precision.value}")  # 0.42 (larger decrease)

Hierarchical Precision
----------------------

Three Levels
^^^^^^^^^^^^

LRS agents track precision at three hierarchical levels:

1. **Abstract**: High-level goals and strategies
2. **Planning**: Action sequences and policies
3. **Execution**: Individual tool executions

Why Hierarchy?
^^^^^^^^^^^^^^

Different levels have different timescales:

* **Execution**: Changes every step (fast)
* **Planning**: Changes when policies fail (medium)
* **Abstract**: Changes when strategies fail (slow)

Error Propagation
^^^^^^^^^^^^^^^^^

Errors propagate upward when they exceed a threshold:

.. code-block:: python

   from lrs.core.precision import HierarchicalPrecision

   hp = HierarchicalPrecision(propagation_threshold=0.7)

   # High error at execution
   hp.update('execution', prediction_error=0.95)
   
   # Error propagates to planning
   # Planning error is attenuated: 0.95 * 0.5 = 0.475
   # If still above threshold, propagates to abstract

This prevents:

* **Over-reaction**: Not every tool failure requires strategy change
* **Under-reaction**: Persistent failures trigger higher-level adaptation

Prediction Errors
-----------------

What Are They?
^^^^^^^^^^^^^^

Prediction errors measure how surprising an observation is:

.. math::

   \delta = |\text{predicted} - \text{observed}|

In LRS-Agents, each tool execution returns a prediction error in [0, 1]:

* **0.0**: Perfectly predicted (deterministic operations)
* **0.5**: Medium surprise
* **1.0**: Completely unexpected

How to Set Them
^^^^^^^^^^^^^^^

Guidelines for tool implementers:

.. code-block:: python

   def get(self, state):
       try:
           result = execute_operation()
           
           if operation_is_deterministic:
               return ExecutionResult(True, result, None, 0.0)
           elif operation_usually_succeeds:
               return ExecutionResult(True, result, None, 0.1)
           else:
               return ExecutionResult(True, result, None, 0.3)
       
       except ExpectedError:
           return ExecutionResult(False, None, str(e), 0.6)
       except UnexpectedError:
           return ExecutionResult(False, None, str(e), 0.95)

Why They Matter
^^^^^^^^^^^^^^^

Prediction errors drive adaptation:

* **Low errors**: Precision increases → Exploit current strategy
* **High errors**: Precision decreases → Explore alternatives

Tool Lenses
-----------

What is a Lens?
^^^^^^^^^^^^^^^

A lens is a bidirectional interface inspired by functional programming:

* **get()**: Read from environment (forward)
* **set()**: Update belief state (backward)

Why Lenses?
^^^^^^^^^^^

Lenses provide:

1. **Bidirectionality**: Both observe and update
2. **Composability**: Chain tools with ``>>`` operator
3. **Type safety**: Input/output schemas
4. **Statistics**: Automatic call/failure tracking

Example
^^^^^^^

.. code-block:: python

   from lrs.core.lens import ToolLens, ExecutionResult

   class DatabaseTool(ToolLens):
       def get(self, state):
           """Observe: Query database"""
           query = state.get('query')
           result = self.db.execute(query)
           return ExecutionResult(True, result, None, 0.1)
       
       def set(self, state, observation):
           """Update: Store result in belief state"""
           return {**state, 'db_result': observation}

Composition
^^^^^^^^^^^

Tools compose naturally:

.. code-block:: python

   pipeline = FetchTool() >> ParseTool() >> ExtractTool()
   
   result = pipeline.get(state)
   # Executes FetchTool, then ParseTool, then ExtractTool
   # Short-circuits on first failure

Policy Selection
----------------

How Policies are Selected
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Generate** candidate policies (via LLM or exhaustive search)
2. **Evaluate** each policy by calculating G
3. **Select** via precision-weighted softmax:

.. math::

   P(policy_i) = \frac{e^{-\beta \cdot G_i}}{\sum_j e^{-\beta \cdot G_j}}

where β (inverse temperature) depends on precision.

Precision-Weighted Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High Precision (γ > 0.7)
""""""""""""""""""""""""

* Low temperature → Deterministic
* Select policy with lowest G
* **Exploitation**: Use what works

.. code-block:: python

   # High precision
   probs = [0.85, 0.10, 0.05]  # Mostly best policy

Low Precision (γ < 0.4)
"""""""""""""""""""""""

* High temperature → Stochastic
* Uniform-ish selection
* **Exploration**: Try alternatives

.. code-block:: python

   # Low precision
   probs = [0.40, 0.35, 0.25]  # More uniform

Adaptation
----------

When Does It Happen?
^^^^^^^^^^^^^^^^^^^^

Adaptation is triggered when precision drops below threshold (default 0.4).

What Happens?
^^^^^^^^^^^^^

1. **Temperature increases**: More exploration
2. **Epistemic weight increases**: Favor information gain
3. **Alternative tools considered**: Explore registry
4. **Strategy shifts**: From exploit to explore

Example Scenario
^^^^^^^^^^^^^^^^

.. code-block:: text

   Step 1: Try API (precision = 0.5)
           ✓ Success (error = 0.1)
           precision → 0.52
   
   Step 2: Try API (precision = 0.52)
           ✗ Timeout (error = 0.9)
           precision → 0.42
   
   Step 3: Try API (precision = 0.42)
           ✗ Timeout (error = 0.9)
           precision → 0.32  ← Below threshold!
   
   [ADAPTATION TRIGGERED]
   
   Step 4: Explore alternatives
           Temperature: 0.7 → 1.5
           Consider: cache, database, retry_api
   
   Step 5: Try cache (precision = 0.32)
           ✓ Success (error = 0.05)
           precision → 0.38
   
   Step 6: Try cache (precision = 0.38)
           ✓ Success (error = 0.05)
           precision → 0.44  ← Recovery!

Multi-Agent Concepts
--------------------

Social Precision
^^^^^^^^^^^^^^^^

In multi-agent systems, agents track **social precision** (trust) in other agents:

.. code-block:: python

   from lrs.multi_agent.social_precision import SocialPrecisionTracker

   tracker = SocialPrecisionTracker("agent_a")
   tracker.register_agent("agent_b")
   
   # After observing agent_b's action
   tracker.update_social_precision(
       "agent_b",
       predicted_action="fetch",
       observed_action="cache"  # Different!
   )
   # Social precision decreases

Communication as Information-Seeking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Agents communicate when:

* Social precision is low (uncertain about others)
* Environmental precision is high (not a personal problem)

This treats communication as **active sensing** to reduce social uncertainty.

Putting It All Together
-----------------------

The LRS Agent Loop
^^^^^^^^^^^^^^^^^^

.. code-block:: text

   1. Generate policy proposals
      ├─ Via LLM (for 10+ tools)
      └─ Via exhaustive search (for <10 tools)
   
   2. Evaluate each policy
      ├─ Calculate epistemic value
      ├─ Calculate pragmatic value
      └─ Compute G = epistemic - pragmatic
   
   3. Select policy
      ├─ Precision-weighted softmax
      └─ High γ → exploit, Low γ → explore
   
   4. Execute policy
      ├─ Run tools in sequence
      ├─ Observe prediction errors
      └─ Short-circuit on failure
   
   5. Update precision
      ├─ Low error → increase precision
      └─ High error → decrease precision
   
   6. Check adaptation
      ├─ If γ < threshold → adapt
      └─ If goal achieved → terminate
   
   7. Repeat from step 1

Key Takeaways
-------------

1. **Precision drives behavior**

   * High precision → Exploit (use what works)
   * Low precision → Explore (try alternatives)

2. **Prediction errors update precision**

   * Low errors → Increase confidence
   * High errors → Decrease confidence

3. **G balances explore/exploit**

   * Epistemic value: Information gain
   * Pragmatic value: Expected reward

4. **Adaptation is automatic**

   * No manual error handling needed
   * Agent explores when uncertain

5. **Hierarchy prevents over-reaction**

   * Errors attenuate as they propagate
   * Different timescales for different levels

Next Steps
----------

* Try the :doc:`../tutorials/02_understanding_precision` notebook
* Read about :doc:`../theory/active_inference`
* Explore the :doc:`../api/core` API reference

