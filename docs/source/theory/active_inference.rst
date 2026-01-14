Active Inference
================

A comprehensive introduction to Active Inference and its application to AI agents.

Overview
--------

**Active Inference** is a unified theory from neuroscience that explains how biological agents perceive, learn, and act. It provides a mathematical framework for building agents that:

* Minimize surprise about their sensory observations
* Balance exploration (learning) vs exploitation (goal achievement)
* Adapt automatically when predictions fail

This document explains the theory and how LRS-Agents implements it.

The Core Principle
------------------

Free Energy Minimization
^^^^^^^^^^^^^^^^^^^^^^^^

Active Inference is based on the **Free Energy Principle**, which states:

   *Biological agents act to minimize their free energy - a measure of surprise about sensory observations.*

In mathematical terms:

.. math::

   F = -\ln P(o | m)

where:

* :math:`F` = Free energy (surprise)
* :math:`o` = Observations
* :math:`m` = Internal model of the world
* :math:`P(o | m)` = Probability of observations given the model

Lower free energy = Better predictions = Less surprise

Two Ways to Minimize Free Energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Agents can minimize free energy in two complementary ways:

1. **Perception** (Update beliefs)
   
   Change the internal model :math:`m` to better explain observations :math:`o`.
   
   *"The world surprised me, so I'll update my beliefs."*

2. **Action** (Change world)
   
   Act to make observations :math:`o` more consistent with expectations.
   
   *"The world surprised me, so I'll act to make it match my predictions."*

This is the fundamental loop of Active Inference:

.. code-block:: text

   ┌─────────────────────────────────────┐
   │                                     │
   │  ┌──────────┐      ┌──────────┐   │
   │  │  BELIEF  │─────→│  ACTION  │   │
   │  │  UPDATE  │      │ SELECTION│   │
   │  └────▲─────┘      └────┬─────┘   │
   │       │                 │          │
   │       │                 ▼          │
   │  ┌────┴─────┐      ┌──────────┐   │
   │  │PREDICTION│◄─────│   WORLD  │   │
   │  │  ERROR   │      │   STATE  │   │
   │  └──────────┘      └──────────┘   │
   │                                     │
   └─────────────────────────────────────┘

Active Inference for AI Agents
-------------------------------

Traditional RL vs Active Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Reinforcement Learning:**

* Maximize expected reward
* Separate exploration and exploitation mechanisms
* No explicit uncertainty tracking
* Requires manual exploration strategies (ε-greedy, etc.)

**Active Inference:**

* Minimize expected free energy
* Exploration and exploitation emerge naturally
* Explicit uncertainty (precision) tracking
* Automatic adaptation when uncertain

Key Insight
^^^^^^^^^^^

In Active Inference, **exploration IS uncertainty reduction**.

An agent explores not randomly, but to gain information that reduces uncertainty about the world. This makes exploration principled and efficient.

Mathematical Framework
----------------------

Generative Model
^^^^^^^^^^^^^^^^

The agent maintains a **generative model** :math:`P(o, s)` that describes:

* :math:`P(o | s)` - How observations arise from hidden states
* :math:`P(s)` - Prior beliefs about states

The agent's goal is to infer hidden states :math:`s` from observations :math:`o`.

Variational Free Energy
^^^^^^^^^^^^^^^^^^^^^^^

Since exact inference is intractable, we use **variational inference**:

.. math::

   F = D_{KL}[Q(s) || P(s|o)] - \ln P(o)

where:

* :math:`Q(s)` = Approximate posterior (agent's beliefs)
* :math:`P(s|o)` = True posterior (unknowable)
* :math:`D_{KL}` = Kullback-Leibler divergence

Minimizing :math:`F` means:

1. Make :math:`Q(s)` close to :math:`P(s|o)` (accurate beliefs)
2. Maximize :math:`\ln P(o)` (make observations likely)

Expected Free Energy
^^^^^^^^^^^^^^^^^^^^

For action selection, agents minimize **Expected Free Energy** :math:`G`:

.. math::

   G(\pi) = \mathbb{E}_{Q(o_\tau | \pi)}[F(o_\tau)] + D_{KL}[Q(s_\tau | \pi) || P(s_\tau | C)]

This decomposes into:

.. math::

   G(\pi) = \underbrace{\mathbb{E}[H[P(o|s)]]}_{\text{Epistemic value}} - \underbrace{\mathbb{E}[Q(s) \ln P(o|s,C)]}_{\text{Pragmatic value}}

where:

* **Epistemic value**: Information gain (exploration)
* **Pragmatic value**: Expected reward (exploitation)

The agent selects policies :math:`\pi` that minimize :math:`G`.

Precision-Weighted Beliefs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not all beliefs are equally certain. Precision :math:`\gamma` weights predictions:

.. math::

   F = \gamma \cdot \text{Prediction Error}

High precision :math:`\gamma` → Trust predictions more (exploit)
Low precision :math:`\gamma` → Trust predictions less (explore)

Precision dynamics are central to adaptation in LRS-Agents.

How LRS-Agents Implements Active Inference
-------------------------------------------

1. Generative Model
^^^^^^^^^^^^^^^^^^^

The agent's generative model consists of:

* **Tools** as actions that change world state
* **Belief state** :math:`s` as a dictionary of key-value pairs
* **Observations** :math:`o` from tool executions

.. code-block:: python

   # Belief state (internal model)
   state = {
       'goal': 'fetch_data',
       'api_available': True,
       'cache_available': True,
       'data': None
   }

   # Tool execution (action) produces observation
   observation = tool.get(state)
   # observation.value, observation.error, observation.prediction_error

2. Precision Tracking
^^^^^^^^^^^^^^^^^^^^^

Precision :math:`\gamma \in [0, 1]` represents confidence:

.. math::

   \gamma = \frac{\alpha}{\alpha + \beta}

Updated via Beta distribution after each observation:

.. math::

   \alpha &\leftarrow \alpha + \eta_{gain} \cdot (1 - \delta) \\
   \beta &\leftarrow \beta + \eta_{loss} \cdot \delta

where :math:`\delta` is the prediction error.

Asymmetric learning rates:

* :math:`\eta_{gain} = 0.1` (slow increase)
* :math:`\eta_{loss} = 0.2` (fast decrease)

This creates **optimism bias**: Easy to become confident, hard to lose it unless strongly surprised.

.. code-block:: python

   from lrs.core.precision import PrecisionParameters

   precision = PrecisionParameters()
   
   # Success: slow increase
   precision.update(prediction_error=0.1)  # γ: 0.5 → 0.52
   
   # Failure: fast decrease
   precision.update(prediction_error=0.9)  # γ: 0.52 → 0.42

3. Expected Free Energy Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each policy (tool sequence), calculate:

**Epistemic Value** (Information gain):

.. math::

   \text{Epistemic} = \sum_{t} H[P(o_t | s_t)]

Higher for novel/uncertain tools.

**Pragmatic Value** (Expected reward):

.. math::

   \text{Pragmatic} = \sum_{t} \gamma^t [P(\text{success}) \cdot R_{\text{success}} + P(\text{fail}) \cdot R_{\text{fail}}]

Higher for reliable tools with good outcomes.

**Total G**:

.. math::

   G = \text{Epistemic} - \text{Pragmatic}

Policies with lower :math:`G` are preferred.

.. code-block:: python

   from lrs.core.free_energy import calculate_expected_free_energy

   G = calculate_expected_free_energy(
       policy=[tool_a, tool_b],
       state=current_state,
       preferences={'success': 5.0, 'error': -3.0},
       historical_stats=registry.statistics
   )

4. Policy Selection
^^^^^^^^^^^^^^^^^^^

Policies are selected via **precision-weighted softmax**:

.. math::

   P(\pi_i) = \frac{\exp(-\beta \cdot G_i)}{\sum_j \exp(-\beta \cdot G_j)}

where inverse temperature:

.. math::

   \beta = \frac{1}{T \cdot (1 - \gamma + \epsilon)}

High :math:`\gamma` → Low temperature → Deterministic (exploit best policy)
Low :math:`\gamma` → High temperature → Stochastic (explore alternatives)

.. code-block:: python

   from lrs.core.free_energy import precision_weighted_selection

   selected_idx = precision_weighted_selection(
       evaluations=[eval_1, eval_2, eval_3],
       precision=0.3  # Low precision → more exploration
   )

5. Hierarchical Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^

LRS-Agents implement **hierarchical Active Inference** with three levels:

* **Abstract**: Long-term goals and strategies
* **Planning**: Action sequences and policies  
* **Execution**: Individual tool executions

Each level has its own precision, updated based on prediction errors:

.. math::

   \delta_{\text{planning}} &= f(\delta_{\text{execution}}) \\
   \delta_{\text{abstract}} &= f(\delta_{\text{planning}})

where :math:`f` applies threshold and attenuation:

.. math::

   f(\delta) = \begin{cases}
   0 & \text{if } \delta < \theta \\
   \alpha \cdot \delta & \text{if } \delta \geq \theta
   \end{cases}

This prevents over-reaction to individual tool failures while allowing persistent errors to propagate upward.

.. code-block:: python

   from lrs.core.precision import HierarchicalPrecision

   hp = HierarchicalPrecision(
       propagation_threshold=0.7,
       attenuation_factor=0.5
   )

   # High execution error
   hp.update('execution', prediction_error=0.95)
   
   # If above threshold, propagates to planning (attenuated)
   # planning_error = 0.95 * 0.5 = 0.475

Active Inference vs Other Approaches
-------------------------------------

Comparison Table
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Property
     - Reinforcement Learning
     - POMDP
     - Active Inference
   * - Objective
     - Maximize reward
     - Maximize expected value
     - Minimize free energy
   * - Exploration
     - Manual (ε-greedy, etc.)
     - Information value
     - Epistemic value
   * - Uncertainty
     - Implicit
     - Belief state
     - Precision (explicit)
   * - Adaptation
     - Fixed learning rate
     - Bayesian update
     - Precision-weighted update
   * - Hierarchy
     - Options framework
     - Hierarchical POMDP
     - Hierarchical inference

Advantages of Active Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Principled Exploration**
   
   Exploration emerges from uncertainty reduction, not random sampling.

2. **Unified Framework**
   
   Perception, learning, and action all minimize free energy.

3. **Explicit Uncertainty**
   
   Precision tracking enables adaptive behavior.

4. **Hierarchical Compositionality**
   
   Natural handling of multi-level planning.

5. **Biological Plausibility**
   
   Matches neural mechanisms (predictive coding, precision-weighting).

Theoretical Foundations
-----------------------

Predictive Processing
^^^^^^^^^^^^^^^^^^^^^

Active Inference builds on **Predictive Processing**:

1. The brain constantly generates predictions
2. Predictions are compared to sensory input
3. Prediction errors update beliefs
4. Actions minimize future prediction errors

.. code-block:: text

   Top-down Predictions
          ↓
   ┌─────────────────┐
   │  Sensory Input  │
   └────────┬────────┘
            ↓
   ┌─────────────────┐
   │ Prediction Error│ → Update Beliefs
   └─────────────────┘

Precision-Weighting
^^^^^^^^^^^^^^^^^^^

Not all prediction errors are equally important. Precision :math:`\gamma` acts as a **gain control**:

.. math::

   \Delta \text{Belief} \propto \gamma \cdot \text{Prediction Error}

High :math:`\gamma` → Large belief updates (trust observations)
Low :math:`\gamma` → Small belief updates (trust prior beliefs)

This is equivalent to **attention** in neuroscience:

* High precision = Attend to observations
* Low precision = Ignore observations (trust model)

Bayesian Brain Hypothesis
^^^^^^^^^^^^^^^^^^^^^^^^^^

Active Inference aligns with the **Bayesian Brain Hypothesis**:

   *The brain performs approximate Bayesian inference to maintain beliefs about the world.*

LRS-Agents implement this through:

* Prior beliefs (initial precision)
* Likelihood (tool reliability statistics)
* Posterior (updated precision after observations)

Markov Blanket
^^^^^^^^^^^^^^

Active Inference respects the **Markov Blanket** - the boundary between agent and environment:

.. code-block:: text

   ┌─────────────────────────────┐
   │         AGENT               │
   │  ┌──────────────────────┐   │
   │  │   Internal States    │   │
   │  │   (Belief State)     │   │
   │  └──────────┬───────────┘   │
   │             │               │
   │  ┌──────────▼───────────┐   │
   │  │   Sensory States     │   │ ← Observations
   │  └──────────────────────┘   │
   │  ┌──────────────────────┐   │
   │  │   Active States      │   │ → Actions
   │  └──────────────────────┘   │
   └─────────────────────────────┘

The Markov Blanket ensures:

* Internal states don't directly access the world
* All interactions mediated by sensory/active states

Real-World Applications
-----------------------

Robotics
^^^^^^^^

Active Inference applied to robot control:

* **Precision-weighted motor control**: More precise movements when confident
* **Exploration of environment**: Robots actively seek information
* **Adaptive grasping**: Adjust grip based on prediction errors

Example: A robot arm learning to grasp objects adjusts its grip force based on tactile prediction errors, with precision determining how much to update based on each touch.

Autonomous Vehicles
^^^^^^^^^^^^^^^^^^^

Self-driving cars use Active Inference principles:

* **Perception**: Update beliefs about road conditions
* **Action**: Steer/brake to minimize surprise
* **Precision**: Higher in clear conditions, lower in fog

Clinical Applications
^^^^^^^^^^^^^^^^^^^^^

Understanding mental health through Active Inference:

* **Anxiety**: Persistently low precision (over-sensitivity to prediction errors)
* **Psychosis**: Precision imbalance (hallucinations as over-confident false beliefs)
* **Autism**: Atypical precision-weighting in social contexts

AI Safety
^^^^^^^^^

Active Inference provides safety benefits:

* **Interpretability**: Explicit beliefs and uncertainty
* **Graceful degradation**: Adapts when uncertain
* **Conservative by default**: Won't act confidently without evidence

Limitations and Open Questions
-------------------------------

Computational Complexity
^^^^^^^^^^^^^^^^^^^^^^^^

Full Active Inference requires:

* Sampling over all possible futures
* Evaluating free energy for each scenario
* Maintaining probability distributions

LRS-Agents addresses this through:

* LLM-based proposal mechanisms (variational sampling)
* Hierarchical decomposition
* Caching and approximations

Model Misspecification
^^^^^^^^^^^^^^^^^^^^^^^

What if the generative model is wrong?

* Agent's beliefs may never converge
* Persistent high prediction errors
* Solution: Model selection, structural learning

In LRS-Agents:

* Tool registry provides alternatives
* Adaptation explores different models
* Human-in-the-loop for model updates

Precision Learning
^^^^^^^^^^^^^^^^^^

How to learn optimal precision parameters?

* Current: Hand-tuned asymmetric learning rates
* Future: Meta-learning from task distributions
* Challenge: Balancing stability vs plasticity

Mathematical Details
--------------------

Variational Message Passing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LRS-Agents implement approximate inference via **variational message passing**:

1. **Forward pass**: Propagate predictions down hierarchy
2. **Backward pass**: Propagate prediction errors up hierarchy
3. **Update**: Adjust beliefs to minimize free energy

.. math::

   Q(s) \leftarrow \arg\min_Q F[Q(s)]

Dynamic Causal Modeling
^^^^^^^^^^^^^^^^^^^^^^^^

Tool execution can be seen as **Dynamic Causal Modeling**:

.. math::

   s_{t+1} &= f(s_t, a_t, \theta) + \omega_t \\
   o_t &= g(s_t, \phi) + \nu_t

where:

* :math:`f` = State transition (tool execution)
* :math:`g` = Observation function (tool output)
* :math:`\theta, \phi` = Parameters
* :math:`\omega_t, \nu_t` = Noise

Prediction errors:

.. math::

   \epsilon_s &= s_{t+1} - f(s_t, a_t, \theta) \\
   \epsilon_o &= o_t - g(s_t, \phi)

Generalized Free Energy
^^^^^^^^^^^^^^^^^^^^^^^^

For continuous time and generalized coordinates:

.. math::

   F = \frac{1}{2} \epsilon' \Pi \epsilon

where:

* :math:`\epsilon` = Prediction error vector
* :math:`\Pi = \text{diag}(\gamma)` = Precision matrix

Further Reading
---------------

Foundational Papers
^^^^^^^^^^^^^^^^^^^

* Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*
* Friston, K. et al. (2015). "Active inference and epistemic value." *Cognitive Neuroscience*
* Parr, T., & Friston, K. (2019). "Generalised free energy and active inference." *Biological Cybernetics*

Books
^^^^^

* Clark, A. (2015). *Surfing Uncertainty: Prediction, Action, and the Embodied Mind*
* Hohwy, J. (2013). *The Predictive Mind*
* Friston, K., & Parr, T. (2021). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*

Implementations
^^^^^^^^^^^^^^^

* `pymdp <https://github.com/infer-actively/pymdp>`_ - Python implementation of Active Inference
* `SPM <https://www.fil.ion.ucl.ac.uk/spm/>`_ - MATLAB toolbox for Active Inference

Next Steps
----------

* Read about :doc:`free_energy` for detailed G calculation
* Understand :doc:`precision_dynamics` for adaptation mechanisms
* See :doc:`../getting_started/core_concepts` for practical implementation
* Explore :doc:`../tutorials/02_understanding_precision` for hands-on examples

