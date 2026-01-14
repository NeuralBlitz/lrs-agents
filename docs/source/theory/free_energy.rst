Free Energy
===========

A deep dive into Expected Free Energy and its calculation in LRS-Agents.

Overview
--------

**Expected Free Energy** (G) is the core quantity minimized by Active Inference agents. This document explains:

* What G represents
* How it's calculated
* Why it balances exploration and exploitation
* Implementation details in LRS-Agents

What is Expected Free Energy?
------------------------------

Definition
^^^^^^^^^^

Expected Free Energy :math:`G` for a policy :math:`\pi` is:

.. math::

   G(\pi) = \mathbb{E}_{Q(o_\tau | \pi)}[\ln Q(s_\tau | \pi) - \ln P(o_\tau, s_\tau | C)]

where:

* :math:`o_\tau` = Future observations under policy :math:`\pi`
* :math:`s_\tau` = Hidden states
* :math:`C` = Preferences (goals)
* :math:`Q` = Approximate posterior (beliefs)
* :math:`P` = Generative model

Intuitive Explanation
^^^^^^^^^^^^^^^^^^^^^

:math:`G` measures the "badness" of a policy considering:

1. **Uncertainty reduction** (Will I learn something?)
2. **Goal achievement** (Will I get reward?)

Lower :math:`G` = Better policy

A policy with low :math:`G`:

* Reduces uncertainty about the world (epistemic value)
* Achieves desired outcomes (pragmatic value)

Decomposition
^^^^^^^^^^^^^

:math:`G` decomposes into two terms:

.. math::

   G(\pi) = \underbrace{\mathbb{E}[H[P(o|s)]]}_{\text{Epistemic}} - \underbrace{\mathbb{E}[\ln P(o|C)]}_{\text{Pragmatic}}

**Epistemic Value** (Information Gain):

* How much will this policy reduce uncertainty?
* High for novel, uncertain outcomes
* Drives **exploration**

**Pragmatic Value** (Expected Utility):

* How much will this policy achieve my goals?
* High for reliable, rewarding outcomes
* Drives **exploitation**

The Trade-off
^^^^^^^^^^^^^

.. code-block:: text

   High Epistemic, Low Pragmatic → Explore
        (Learn but risky)
        
   Low Epistemic, High Pragmatic → Exploit
        (Safe but boring)
        
   Low Epistemic, Low Pragmatic → Avoid
        (Neither learn nor gain)
        
   High Epistemic, High Pragmatic → Ideal!
        (Learn and gain)

Epistemic Value Calculation
----------------------------

Definition
^^^^^^^^^^

Epistemic value measures information gain:

.. math::

   \text{Epistemic}(\pi) = \mathbb{E}_{Q(s|\pi)}[H[P(o|s)]]

where :math:`H` is entropy (uncertainty).

High entropy → High uncertainty → High information gain

In LRS-Agents
^^^^^^^^^^^^^

For a policy (sequence of tools):

.. math::

   \text{Epistemic} = \sum_{t=1}^{T} H[\text{Tool}_t]

where the entropy of each tool depends on:

1. **Historical reliability**: More failures → More uncertainty
2. **Novelty**: Never used → Maximum uncertainty
3. **Context**: State-dependent uncertainty

.. code-block:: python

   from lrs.core.free_energy import calculate_epistemic_value

   epistemic = calculate_epistemic_value(
       policy=[novel_tool, uncertain_tool],
       state={},
       historical_stats=None  # No history = high uncertainty
   )
   # Returns: ~1.5 (high information gain)

Calculation Details
^^^^^^^^^^^^^^^^^^^

For each tool in the policy:

.. math::

   H[\text{Tool}] = -\sum_i P(\text{outcome}_i) \log P(\text{outcome}_i)

Outcome probabilities from historical statistics:

.. math::

   P(\text{success}) &= \frac{\text{successes}}{\text{total calls}} \\
   P(\text{failure}) &= 1 - P(\text{success})

Binary entropy:

.. math::

   H = -P(\text{success}) \log P(\text{success}) - P(\text{failure}) \log P(\text{failure})

Special cases:

* **No history**: :math:`H = \log 2 \approx 0.69` (maximum uncertainty)
* **Always succeeds**: :math:`H = 0` (no uncertainty)
* **50/50 success**: :math:`H = \log 2` (maximum binary entropy)

Example
^^^^^^^

.. code-block:: python

   from lrs.core.free_energy import calculate_epistemic_value

   # Tool with 70% success rate
   # H = -0.7*log(0.7) - 0.3*log(0.3) ≈ 0.61

   # Tool never used before
   # H = log(2) ≈ 0.69

   # Total epistemic value for policy
   epistemic = 0.61 + 0.69 = 1.30

Pragmatic Value Calculation
----------------------------

Definition
^^^^^^^^^^

Pragmatic value measures expected utility:

.. math::

   \text{Pragmatic}(\pi) = \mathbb{E}_{Q(o|\pi)}[\ln P(o|C)]

where :math:`P(o|C)` represents preferences over outcomes.

In simpler terms:

.. math::

   \text{Pragmatic} = \sum_{t=1}^{T} \gamma^t \left[ P_t(\text{success}) \cdot R_{\text{success}} + P_t(\text{failure}) \cdot R_{\text{failure}} \right]

where:

* :math:`\gamma` = Discount factor (default 0.99)
* :math:`R` = Rewards from preferences
* :math:`P_t` = Success probability at step :math:`t`

In LRS-Agents
^^^^^^^^^^^^^

.. code-block:: python

   from lrs.core.free_energy import calculate_pragmatic_value

   pragmatic = calculate_pragmatic_value(
       policy=[reliable_tool, fast_tool],
       state={},
       preferences={
           'success': 5.0,     # Reward for success
           'error': -3.0,      # Penalty for error
           'step_cost': -0.1   # Small cost per step
       },
       historical_stats=registry.statistics,
       discount_factor=0.99
   )

Calculation Details
^^^^^^^^^^^^^^^^^^^

For each tool at step :math:`t`:

.. math::

   V_t = \gamma^{t-1} \left[ p_{\text{success}} \cdot R_{\text{success}} + (1 - p_{\text{success}}) \cdot R_{\text{error}} \right] + R_{\text{step}}

where:

* :math:`p_{\text{success}}` from historical statistics
* :math:`R_{\text{success}}` from preferences (default 5.0)
* :math:`R_{\text{error}}` from preferences (default -3.0)
* :math:`R_{\text{step}}` = step cost (default -0.1)

Total pragmatic value:

.. math::

   \text{Pragmatic} = \sum_{t=1}^{T} V_t

Example
^^^^^^^

.. code-block:: python

   # Policy: [tool_a, tool_b]
   # tool_a: 80% success
   # tool_b: 90% success

   # Step 1: tool_a
   V_1 = 0.99^0 * [0.8 * 5.0 + 0.2 * (-3.0)] - 0.1
   V_1 = 1.0 * [4.0 - 0.6] - 0.1 = 3.3

   # Step 2: tool_b
   V_2 = 0.99^1 * [0.9 * 5.0 + 0.1 * (-3.0)] - 0.1
   V_2 = 0.99 * [4.5 - 0.3] - 0.1 ≈ 4.06

   # Total pragmatic value
   Pragmatic = 3.3 + 4.06 = 7.36

Total Expected Free Energy
---------------------------

Formula
^^^^^^^

Combining epistemic and pragmatic values:

.. math::

   G = \alpha \cdot \text{Epistemic} - \text{Pragmatic}

where :math:`\alpha` is the epistemic weight (default 1.0).

**Lower G is better** because:

* High epistemic → Higher G (exploration cost)
* High pragmatic → Lower G (exploitation benefit)

The agent balances both by minimizing G.

In LRS-Agents
^^^^^^^^^^^^^

.. code-block:: python

   from lrs.core.free_energy import calculate_expected_free_energy

   G = calculate_expected_free_energy(
       policy=[tool_a, tool_b],
       state={},
       preferences={'success': 5.0, 'error': -3.0},
       historical_stats=registry.statistics,
       epistemic_weight=1.0,
       discount_factor=0.99
   )

Detailed Example
^^^^^^^^^^^^^^^^

Compare two policies:

**Policy A: Reliable tools** [cache_tool, db_tool]

.. code-block:: python

   # Epistemic (low - known tools)
   Epistemic_A = 0.1 + 0.15 = 0.25

   # Pragmatic (high - reliable)
   Pragmatic_A = 4.8 + 4.5 = 9.3

   # G
   G_A = 0.25 - 9.3 = -9.05  # Very negative (good!)

**Policy B: Novel tools** [new_api, experimental_tool]

.. code-block:: python

   # Epistemic (high - uncertain)
   Epistemic_B = 0.69 + 0.69 = 1.38

   # Pragmatic (low - unreliable)
   Pragmatic_B = 2.0 + 1.5 = 3.5

   # G
   G_B = 1.38 - 3.5 = -2.12  # Less negative (worse)

**Result**: Agent prefers Policy A (lower G).

Precision-Weighted Selection
-----------------------------

G alone doesn't determine policy selection. Precision :math:`\gamma` weights the choice.

Softmax Selection
^^^^^^^^^^^^^^^^^

Policies are selected via softmax:

.. math::

   P(\pi_i) = \frac{\exp(-\beta \cdot G_i)}{\sum_j \exp(-\beta \cdot G_j)}

where inverse temperature:

.. math::

   \beta = \frac{1}{T \cdot (1 - \gamma + \epsilon)}

Key insight:

* High :math:`\gamma` → High :math:`\beta` → Deterministic selection (exploit)
* Low :math:`\gamma` → Low :math:`\beta` → Stochastic selection (explore)

Example
^^^^^^^

Three policies with G values:

.. code-block:: python

   G_values = [-9.05, -7.2, -5.1]  # Lower is better

   # High precision (γ = 0.8)
   β_high = 1 / (0.7 * (1 - 0.8 + 0.01)) ≈ 6.8
   P_high = softmax(-6.8 * G_values)
   # Result: [0.85, 0.12, 0.03]  # Exploit best

   # Low precision (γ = 0.3)
   β_low = 1 / (0.7 * (1 - 0.3 + 0.01)) ≈ 2.0
   P_low = softmax(-2.0 * G_values)
   # Result: [0.50, 0.32, 0.18]  # More exploration

Precision-Dependent Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Precision
     - Temperature
     - Behavior
   * - γ > 0.7 (High)
     - Low (deterministic)
     - Exploit: Select best policy
   * - γ ≈ 0.5 (Medium)
     - Medium
     - Balanced: Softmax over policies
   * - γ < 0.3 (Low)
     - High (stochastic)
     - Explore: Try alternatives

Adaptive G Evaluation
----------------------

Epistemic Weight Adaptation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The epistemic weight :math:`\alpha` can adapt with precision:

.. math::

   \alpha(\gamma) = \alpha_{\text{base}} \cdot \left(1 + \frac{1 - \gamma}{\gamma + \epsilon}\right)

Low precision → Higher epistemic weight → More exploration

.. code-block:: python

   def adaptive_epistemic_weight(base_alpha, precision):
       return base_alpha * (1 + (1 - precision) / (precision + 0.01))

   # High precision
   alpha_high = adaptive_epistemic_weight(1.0, 0.8)
   # Result: 1.25 (slightly higher epistemic)

   # Low precision
   alpha_low = adaptive_epistemic_weight(1.0, 0.3)
   # Result: 3.3 (much higher epistemic - explore!)

Context-Dependent G
^^^^^^^^^^^^^^^^^^^^

G can depend on current state:

.. code-block:: python

   def calculate_contextual_G(policy, state, precision):
       # Standard G calculation
       G_base = calculate_expected_free_energy(policy, state, ...)
       
       # Adjust based on context
       if state.get('urgent'):
           # Prioritize pragmatic value when urgent
           G_adjusted = G_base * 0.5  # Favor low-G policies more
       elif state.get('exploratory_phase'):
           # Increase epistemic weight
           G_adjusted = G_base * 2.0  # Allow higher-G exploration
       else:
           G_adjusted = G_base
       
       return G_adjusted

Multiple Objectives
^^^^^^^^^^^^^^^^^^^^

Handle multiple competing objectives:

.. math::

   G_{\text{total}} = \sum_i w_i \cdot G_i

.. code-block:: python

   # Example: Balance speed and accuracy
   G_speed = calculate_G(policy, preferences_speed)
   G_accuracy = calculate_G(policy, preferences_accuracy)
   
   # Weight based on precision
   if precision > 0.7:
       w_speed, w_accuracy = 0.3, 0.7  # Prioritize accuracy
   else:
       w_speed, w_accuracy = 0.6, 0.4  # Try faster approaches
   
   G_total = w_speed * G_speed + w_accuracy * G_accuracy

Hybrid G Evaluation
--------------------

LLM + Mathematical G
^^^^^^^^^^^^^^^^^^^^

LRS-Agents support **hybrid evaluation** combining:

* Mathematical G (precise but limited)
* LLM-estimated G (flexible but noisy)

.. math::

   G_{\text{hybrid}} = (1 - \lambda) \cdot G_{\text{math}} + \lambda \cdot G_{\text{llm}}

where :math:`\lambda = 1 - \gamma` (trust LLM more when uncertain).

.. code-block:: python

   from lrs.inference.evaluator import HybridGEvaluator

   evaluator = HybridGEvaluator()

   eval_result = evaluator.evaluate_hybrid(
       proposal=llm_proposal,
       state={},
       preferences={'success': 5.0},
       precision=0.5,
       historical_stats=registry.statistics
   )

   print(f"G_hybrid: {eval_result.total_G}")
   print(f"G_math: {eval_result.components['G_math']}")
   print(f"G_llm: {eval_result.components['G_llm']}")
   print(f"λ: {eval_result.components['lambda']}")

Why Hybrid?
^^^^^^^^^^^

**Mathematical G**:

* ✓ Precise
* ✓ Consistent
* ✗ Limited to known tools
* ✗ Can't handle novel contexts

**LLM G**:

* ✓ Flexible
* ✓ Handles novel scenarios
* ✗ Noisy
* ✗ Can be overconfident

**Hybrid**:

* ✓ Precise when certain (high γ)
* ✓ Flexible when uncertain (low γ)
* ✓ Best of both worlds

Edge Cases and Special Scenarios
---------------------------------

Empty Policy
^^^^^^^^^^^^

.. math::

   G_{\text{empty}} = 0

No action = No information gain, no reward.

Single Tool
^^^^^^^^^^^

.. math::

   G = H[\text{tool}] - [p \cdot R_{\text{success}} + (1-p) \cdot R_{\text{error}}] - R_{\text{step}}

Long Policies
^^^^^^^^^^^^^

For policies with many steps, discount future contributions:

.. math::

   G = \sum_{t=1}^{T} \gamma^{t-1} [H_t - V_t]

Novel Tools
^^^^^^^^^^^

For tools never seen before:

* Assume maximum entropy: :math:`H = \log 2`
* Assume neutral success probability: :math:`p = 0.5`
* Results in moderate G (neither avoid nor strongly prefer)

Failed Policies
^^^^^^^^^^^^^^^

If a policy fails during execution:

* G becomes irrelevant (policy didn't complete)
* Precision drops based on failure
* Next iteration explores alternatives

Implementation Details
----------------------

Caching
^^^^^^^

G calculations can be expensive. Cache results:

.. code-block:: python

   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def cached_calculate_G(policy_tuple, state_hash, preferences_hash):
       return calculate_expected_free_energy(
           policy=list(policy_tuple),
           state=unhash(state_hash),
           preferences=unhash(preferences_hash),
           ...
       )

Numerical Stability
^^^^^^^^^^^^^^^^^^^

Avoid numerical issues:

.. code-block:: python

   import numpy as np

   def safe_log(x, epsilon=1e-10):
       """Log with numerical stability"""
       return np.log(np.maximum(x, epsilon))

   def safe_entropy(p, epsilon=1e-10):
       """Entropy with stability"""
       p = np.clip(p, epsilon, 1 - epsilon)
       return -p * safe_log(p) - (1 - p) * safe_log(1 - p)

Batch Evaluation
^^^^^^^^^^^^^^^^

Evaluate multiple policies efficiently:

.. code-block:: python

   def evaluate_batch(policies, state, preferences, stats):
       """Vectorized G calculation"""
       epistemics = [calculate_epistemic_value(p, state, stats) 
                     for p in policies]
       pragmatics = [calculate_pragmatic_value(p, state, preferences, stats)
                     for p in policies]
       
       G_values = np.array(epistemics) - np.array(pragmatics)
       return G_values

Validation
^^^^^^^^^^

Sanity checks:

.. code-block:: python

   def validate_G(G, policy):
       """Ensure G is reasonable"""
       assert np.isfinite(G), "G must be finite"
       assert -100 < G < 100, "G out of reasonable range"
       
       # More pragmatic policies should have lower G
       # (all else equal)

Debugging
^^^^^^^^^

Inspect G components:

.. code-block:: python

   from lrs.core.free_energy import evaluate_policy

   eval_obj = evaluate_policy(policy, state, preferences, stats)

   print(f"Total G: {eval_obj.total_G}")
   print(f"Epistemic: {eval_obj.epistemic_value}")
   print(f"Pragmatic: {eval_obj.pragmatic_value}")
   print(f"Per-step breakdown:")
   for i, (e, p) in enumerate(zip(eval_obj.step_epistemics, 
                                    eval_obj.step_pragmatics)):
       print(f"  Step {i+1}: E={e:.2f}, P={p:.2f}, G={e-p:.2f}")

Further Reading
---------------

* :doc:`active_inference` - Theoretical foundations
* :doc:`precision_dynamics` - How precision affects G
* :doc:`../api/core` - API reference for free_energy module
* Friston et al. (2015). "Active inference and epistemic value"

Next Steps
----------

* Understand :doc:`precision_dynamics` for adaptation
* Try :doc:`../tutorials/02_understanding_precision` for hands-on practice
* Read :doc:`../getting_started/core_concepts` for implementation details

