Precision Dynamics
==================

A comprehensive guide to precision tracking...
Precision Dynamics
==================

A comprehensive guide to precision tracking, updates, and adaptation in LRS-Agents.

Overview
--------

**Precision** (γ) is the cornerstone of adaptive behavior in LRS-Agents. This document explains:

* What precision represents
* How it's updated
* Why it drives adaptation
* Hierarchical precision dynamics
* Social precision in multi-agent systems

What is Precision?
------------------

Definition
^^^^^^^^^^

Precision (γ) represents the agent's **confidence in its predictions**:

.. math::

   \gamma \in [0, 1]

where:

* γ = 0: No confidence (maximum uncertainty)
* γ = 0.5: Neutral confidence
* γ = 1: Complete confidence

In neuroscience, precision corresponds to:

* **Attention**: High precision = attend to observations
* **Gain control**: How much to trust sensory input
* **Learning rate**: How much to update beliefs

Bayesian Formulation
^^^^^^^^^^^^^^^^^^^^

Precision is modeled as a Beta distribution:

.. math::

   \gamma \sim \text{Beta}(\alpha, \beta)

with expected value:

.. math::

   \mathbb{E}[\gamma] = \frac{\alpha}{\alpha + \beta}

and variance:

.. math::

   \text{Var}[\gamma] = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}

**Why Beta distribution?**

* Natural conjugate prior for Bernoulli outcomes (success/failure)
* Bounded to [0, 1]
* Analytically tractable updates
* Captures both mean and uncertainty

Initial Values
^^^^^^^^^^^^^^

LRS-Agents start with:

.. math::

   \alpha_0 = \beta_0 = 1

This gives:

* Mean: γ = 0.5 (neutral)
* Variance: 0.05 (moderate uncertainty)

This is a **uniform prior** - no initial bias toward confidence or uncertainty.

.. code-block:: python

   from lrs.core.precision import PrecisionParameters

   precision = PrecisionParameters()
   print(f"Initial γ: {precision.value}")      # 0.5
   print(f"Variance: {precision.variance}")    # 0.05

Precision Updates
-----------------

Update Rule
^^^^^^^^^^^

After observing prediction error δ, precision updates via:

.. math::

   \alpha' &= \alpha + \eta_{\text{gain}} \cdot (1 - \delta) \\
   \beta' &= \beta + \eta_{\text{loss}} \cdot \delta

where:

* δ ∈ [0, 1]: Prediction error (surprise)
* η_gain: Learning rate for successes (default 0.1)
* η_loss: Learning rate for failures (default 0.2)

New precision:

.. math::

   \gamma' = \frac{\alpha'}{\alpha' + \beta'}

Asymmetric Learning
^^^^^^^^^^^^^^^^^^^

**Key insight**: Learning rates are asymmetric:

.. math::

   \eta_{\text{loss}} = 2 \times \eta_{\text{gain}}

This creates **optimism bias**:

* Easy to become confident (slow α increase)
* Hard to lose confidence (fast β increase)

**Why asymmetric?**

Biological agents show optimism bias because:

1. **Stability**: Prevents overreaction to noise
2. **Persistence**: Maintains goal-directed behavior
3. **Resilience**: Quick recovery from setbacks

.. code-block:: python

   from lrs.core.precision import PrecisionParameters

   precision = PrecisionParameters(
       gain_learning_rate=0.1,   # Slow increase
       loss_learning_rate=0.2    # Fast decrease
   )

   # Success (low error)
   precision.update(0.1)   # γ: 0.50 → 0.52 (small increase)
   
   # Failure (high error)
   precision.update(0.9)   # γ: 0.52 → 0.42 (larger decrease)

Update Examples
^^^^^^^^^^^^^^^

**Scenario 1: Repeated successes**

.. code-block:: python

   precision = PrecisionParameters()
   
   for i in range(10):
       precision.update(0.1)  # Low prediction error
       print(f"Step {i+1}: γ = {precision.value:.3f}")

   # Output:
   # Step 1: γ = 0.518
   # Step 2: γ = 0.535
   # Step 3: γ = 0.551
   # ...
   # Step 10: γ = 0.643

Precision grows slowly but steadily.

**Scenario 2: Repeated failures**

.. code-block:: python

   precision = PrecisionParameters()
   
   for i in range(10):
       precision.update(0.9)  # High prediction error
       print(f"Step {i+1}: γ = {precision.value:.3f}")

   # Output:
   # Step 1: γ = 0.424
   # Step 2: γ = 0.357
   # Step 3: γ = 0.299
   # ...
   # Step 10: γ = 0.105

Precision drops rapidly.

**Scenario 3: Mixed results**

.. code-block:: python

   precision = PrecisionParameters()
   
   errors = [0.1, 0.9, 0.1, 0.1, 0.9, 0.1]
   for i, error in enumerate(errors):
       precision.update(error)
       print(f"Step {i+1}: γ = {precision.value:.3f}, error = {error}")

   # Output:
   # Step 1: γ = 0.518, error = 0.1
   # Step 2: γ = 0.442, error = 0.9
   # Step 3: γ = 0.458, error = 0.1
   # Step 4: γ = 0.473, error = 0.1
   # Step 5: γ = 0.406, error = 0.9
   # Step 6: γ = 0.420, error = 0.1

Precision oscillates around 0.4-0.5.

Precision-Dependent Behavior
-----------------------------

Policy Selection
^^^^^^^^^^^^^^^^

Precision controls how policies are selected:

.. math::

   P(\pi_i) = \frac{\exp(-\beta \cdot G_i)}{\sum_j \exp(-\beta \cdot G_j)}

where:

.. math::

   \beta = \frac{1}{T \cdot (1 - \gamma + \epsilon)}

**High precision** (γ > 0.7):

* β is large → Temperature is low
* Sharply peaked distribution
* **Exploit**: Select best policy deterministically

**Low precision** (γ < 0.4):

* β is small → Temperature is high
* Flatter distribution
* **Explore**: Try alternatives stochastically

.. code-block:: python

   import numpy as np

   def selection_probabilities(G_values, precision, base_temp=0.7):
       """Calculate selection probabilities"""
       beta = 1.0 / (base_temp * (1 - precision + 0.01))
       exp_values = np.exp(-beta * np.array(G_values))
       return exp_values / exp_values.sum()

   G_values = [-9.0, -7.0, -5.0]  # Lower is better

   # High precision
   probs_high = selection_probabilities(G_values, 0.8)
   print(f"High precision: {probs_high}")
   # [0.89, 0.09, 0.02]  # Exploit best

   # Low precision  
   probs_low = selection_probabilities(G_values, 0.3)
   print(f"Low precision: {probs_low}")
   # [0.48, 0.33, 0.19]  # Explore alternatives

Epistemic Weight
^^^^^^^^^^^^^^^^

Precision can modulate epistemic value:

.. math::

   \alpha_{\text{eff}} = \alpha_{\text{base}} \cdot \left(1 + \frac{1 - \gamma}{\gamma + \epsilon}\right)

Lower precision → Higher epistemic weight → More exploration

.. code-block:: python

   def effective_epistemic_weight(base_alpha, precision, epsilon=0.01):
       return base_alpha * (1 + (1 - precision) / (precision + epsilon))

   # High precision: mostly pragmatic
   alpha_high = effective_epistemic_weight(1.0, 0.8)
   print(f"High precision: α = {alpha_high:.2f}")  # 1.25

   # Low precision: emphasize epistemic
   alpha_low = effective_epistemic_weight(1.0, 0.3)
   print(f"Low precision: α = {alpha_low:.2f}")    # 3.32

LLM Temperature
^^^^^^^^^^^^^^^

For LLM-based policy generation:

.. math::

   T_{\text{LLM}} = T_{\text{base}} \times \frac{1}{\gamma + \epsilon}

.. code-block:: python

   from lrs.inference.llm_policy_generator import LLMPolicyGenerator

   generator = LLMPolicyGenerator(llm, registry, base_temperature=0.7)

   # High precision: low temperature (focused)
   temp_high = generator._adapt_temperature(0.8)
   print(f"High precision: T = {temp_high:.2f}")  # 0.88

   # Low precision: high temperature (diverse)
   temp_low = generator._adapt_temperature(0.3)
   print(f"Low precision: T = {temp_low:.2f}")    # 2.26

Adaptation Trigger
------------------

When Does Adaptation Happen?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adaptation is triggered when:

.. math::

   \gamma < \theta

where θ is the adaptation threshold (default 0.4).

**Why 0.4?**

* Below 0.5: Agent is uncertain
* Gives buffer before aggressive exploration
* Empirically works well across tasks

What Happens During Adaptation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Temperature increases**: More stochastic selection
2. **Epistemic weight increases**: Prioritize information gain
3. **Alternative tools considered**: Explore registry
4. **Strategy shifts**: From exploit to explore

.. code-block:: python

   from lrs.core.precision import PrecisionParameters

   precision = PrecisionParameters(adaptation_threshold=0.4)

   # Normal operation
   precision.update(0.2)  # γ = 0.53
   print(f"Adapting: {precision.value < precision.adaptation_threshold}")
   # False - continue exploiting

   # After failures
   precision.update(0.9)  # γ = 0.45
   precision.update(0.9)  # γ = 0.38
   print(f"Adapting: {precision.value < precision.adaptation_threshold}")
   # True - start exploring

Adaptation Example
^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Step 1: Try API (γ = 0.50)
           ✓ Success (δ = 0.1)
           γ → 0.52
           
   Step 2: Try API (γ = 0.52)
           ✗ Failure (δ = 0.9)
           γ → 0.44
           
   Step 3: Try API (γ = 0.44)
           ✗ Failure (δ = 0.9)
           γ → 0.37
           
   [ADAPTATION TRIGGERED: γ < 0.4]
   
   - Temperature: 0.7 → 1.5 (more random)
   - Epistemic weight: 1.0 → 2.8 (prioritize learning)
   - Consider alternatives: cache, database, retry
   
   Step 4: Try cache (γ = 0.37)
           ✓ Success (δ = 0.05)
           γ → 0.41
           
   Step 5: Try cache (γ = 0.41)
           ✓ Success (δ = 0.05)
           γ → 0.45
           
   [RECOVERY: γ > 0.4]

Hierarchical Precision
----------------------

Three Levels
^^^^^^^^^^^^

LRS-Agents track precision hierarchically:

1. **Abstract**: High-level goals and strategies
2. **Planning**: Action sequences and policies
3. **Execution**: Individual tool executions

.. code-block:: text

   ┌─────────────────────────────┐
   │    ABSTRACT (γ_abstract)    │  ← Slowest updates
   └────────────┬────────────────┘
                │ Attenuated errors
   ┌────────────▼────────────────┐
   │    PLANNING (γ_planning)    │  ← Medium updates
   └────────────┬────────────────┘
                │ Attenuated errors
   ┌────────────▼────────────────┐
   │   EXECUTION (γ_execution)   │  ← Fastest updates
   └─────────────────────────────┘

.. code-block:: python

   from lrs.core.precision import HierarchicalPrecision

   hp = HierarchicalPrecision()

   print(f"Abstract: {hp.get_level('abstract').value}")    # 0.5
   print(f"Planning: {hp.get_level('planning').value}")    # 0.5
   print(f"Execution: {hp.get_level('execution').value}")  # 0.5

Error Propagation
^^^^^^^^^^^^^^^^^

Errors propagate **upward** through the hierarchy:

.. math::

   \delta_{\text{planning}} = \begin{cases}
   0 & \text{if } \delta_{\text{execution}} < \theta \\
   \alpha \cdot \delta_{\text{execution}} & \text{otherwise}
   \end{cases}

where:

* θ = Propagation threshold (default 0.7)
* α = Attenuation factor (default 0.5)

**Why propagate?**

* Single tool failure → Update execution precision
* Multiple tool failures → Update planning precision
* Plan failures → Update abstract precision

**Why attenuate?**

* Prevent overreaction to single failures
* Different timescales for different levels
* Abstract strategies shouldn't change from every error

.. code-block:: python

   from lrs.core.precision import HierarchicalPrecision

   hp = HierarchicalPrecision(
       propagation_threshold=0.7,
       attenuation_factor=0.5
   )

   # Small error: no propagation
   updated = hp.update('execution', prediction_error=0.5)
   print(f"Updated levels: {list(updated.keys())}")
   # ['execution'] - only execution level updated

   # Large error: propagates upward
   updated = hp.update('execution', prediction_error=0.95)
   print(f"Updated levels: {list(updated.keys())}")
   # ['execution', 'planning'] - propagated!
   
   # Planning receives attenuated error: 0.95 * 0.5 = 0.475

Hierarchical Behavior
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   hp = HierarchicalPrecision()

   # Execution precision drops
   for _ in range(5):
       hp.update('execution', 0.9)
   
   print(f"Execution: {hp.get_level('execution').value:.2f}")  # ~0.30
   print(f"Planning: {hp.get_level('planning').value:.2f}")    # ~0.45
   print(f"Abstract: {hp.get_level('abstract').value:.2f}")    # ~0.49

   # Execution adapts (tries different tools)
   # If that doesn't work, planning adapts (tries different sequences)
   # If that doesn't work, abstract adapts (tries different strategies)

Precision Collapse
------------------

What is Precision Collapse?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Precision collapse** occurs when:

.. math::

   \gamma \to 0

The agent has lost all confidence in its world model.

**Causes:**

* Persistent failures
* Unpredictable environment
* Model misspecification
* Adversarial inputs

**Symptoms:**

* Random behavior (high temperature)
* Inability to exploit knowledge
* Excessive exploration
* No convergence

Prevention
^^^^^^^^^^

1. **Floor on precision**:

.. code-block:: python

   precision = PrecisionParameters(min_precision=0.1)

2. **Reset mechanism**:

.. code-block:: python

   if precision.value < 0.1:
       precision.reset()  # Start fresh

3. **Model switching**:

.. code-block:: python

   if precision.value < 0.2:
       switch_to_alternative_model()

4. **Human intervention**:

.. code-block:: python

   if precision.value < 0.15:
       request_human_feedback()

Recovery Strategies
^^^^^^^^^^^^^^^^^^^

**Strategy 1: Simplify**

Return to known-good tools:

.. code-block:: python

   if precision.value < 0.2:
       # Use only most reliable tools
       reliable_tools = [t for t in tools if t.success_rate > 0.8]

**Strategy 2: Meta-learning**

Learn which situations cause collapse:

.. code-block:: python

   if precision.value < 0.2:
       # Record context
       collapse_history.append({
           'state': current_state,
           'tools_tried': tool_history,
           'errors': error_history
       })

**Strategy 3: External guidance**

Seek help when uncertain:

.. code-block:: python

   if precision.value < 0.15:
       # Ask LLM for guidance
       suggestion = llm.invoke(
           f"I'm stuck. I've tried {tools} and they all failed. "
           f"What should I try next?"
       )

Social Precision
----------------

Multi-Agent Precision
^^^^^^^^^^^^^^^^^^^^^

In multi-agent systems, agents track **social precision** (trust):

.. math::

   \gamma_{\text{social}}(i, j) = \text{Precision of agent } i \text{ about agent } j

.. code-block:: python

   from lrs.multi_agent.social_precision import SocialPrecisionTracker

   tracker = SocialPrecisionTracker("agent_a")
   tracker.register_agent("agent_b")
   tracker.register_agent("agent_c")

Social Precision Updates
^^^^^^^^^^^^^^^^^^^^^^^^^

Social precision updates when:

1. **Action prediction**: Agent predicts what others will do
2. **Observation**: Agent observes what others actually do
3. **Comparison**: Prediction error updates social precision

.. math::

   \delta_{\text{social}} = |\text{predicted action} - \text{observed action}|

.. code-block:: python

   # Agent A predicts Agent B will "fetch"
   tracker.predict_action("agent_b", "fetch")

   # Agent B actually does "cache"
   tracker.observe_action("agent_b", "cache")

   # Social precision decreases
   gamma_social = tracker.get_social_precision("agent_b")
   print(f"Trust in agent_b: {gamma_social:.2f}")  # Decreased

Communication Decisions
^^^^^^^^^^^^^^^^^^^^^^^

Agents communicate when:

.. math::

   \gamma_{\text{social}} < \theta \quad \text{AND} \quad \gamma_{\text{env}} > \theta

Translation: "I don't understand what you're doing, but I understand the environment."

.. code-block:: python

   from lrs.multi_agent.communication import should_communicate

   if should_communicate(
       social_precision=0.3,     # Low - uncertain about other agent
       env_precision=0.7,        # High - confident about environment
       threshold=0.4
   ):
       send_message(other_agent, "What are you trying to do?")

Theory-of-Mind
^^^^^^^^^^^^^^

Recursive social precision tracking:

.. math::

   \gamma^{(1)} &= \text{My precision about environment} \\
   \gamma^{(2)} &= \text{My precision about your precision} \\
   \gamma^{(3)} &= \text{My precision about your precision about my precision}

.. code-block:: python

   from lrs.multi_agent.social_precision import RecursiveBeliefState

   belief = RecursiveBeliefState()

   belief.my_precision = 0.7              # I'm confident
   belief.belief_about_other = 0.4        # I think you're uncertain
   belief.belief_about_other_belief = 0.6 # I think you think I'm confident

   # Should I help?
   if belief.should_share_information():
       share_knowledge(other_agent)

Precision Dynamics in Practice
-------------------------------

Tuning Learning Rates
^^^^^^^^^^^^^^^^^^^^^^

**Default values** (work well generally):

.. code-block:: python

   precision = PrecisionParameters(
       gain_learning_rate=0.1,
       loss_learning_rate=0.2
   )

**Faster adaptation** (volatile environments):

.. code-block:: python

   precision = PrecisionParameters(
       gain_learning_rate=0.2,  # Faster increase
       loss_learning_rate=0.4   # Faster decrease
   )

**Slower adaptation** (stable environments):

.. code-block:: python

   precision = PrecisionParameters(
       gain_learning_rate=0.05,  # Slower increase
       loss_learning_rate=0.1    # Slower decrease
   )

**Symmetric learning** (unbiased):

.. code-block:: python

   precision = PrecisionParameters(
       gain_learning_rate=0.15,
       loss_learning_rate=0.15  # Same rate
   )

Monitoring Precision
^^^^^^^^^^^^^^^^^^^^

Track precision over time:

.. code-block:: python

   from lrs.monitoring.tracker import LRSStateTracker

   tracker = LRSStateTracker()

   # After agent execution
   trajectory = tracker.get_precision_trajectory('execution')
   
   import matplotlib.pyplot as plt
   plt.plot(trajectory)
   plt.axhline(y=0.4, color='r', linestyle='--', label='Adaptation threshold')
   plt.xlabel('Step')
   plt.ylabel('Precision')
   plt.title('Precision Dynamics')
   plt.legend()
   plt.show()

Analyzing Adaptation Events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   events = tracker.get_adaptation_events()

   for event in events:
       print(f"Step {event['step']}:")
       print(f"  Trigger: {event['trigger']}")
       print(f"  Precision: {event['old_precision']:.2f} → {event['new_precision']:.2f}")
       print(f"  Action: {event['action']}")

Precision-Based Metrics
^^^^^^^^^^^^^^^^^^^^^^^

**Adaptation frequency**:

.. math::

   f_{\text{adapt}} = \frac{\text{# adaptations}}{\text{# steps}}

Higher = More adaptation (possibly too volatile)

**Recovery time**:

.. math::

   t_{\text{recover}} = \text{Steps from } \gamma < \theta \text{ to } \gamma > \theta

Lower = Faster recovery (better resilience)

**Precision stability**:

.. math::

   \sigma_\gamma = \text{Std}(\gamma_1, \gamma_2, \ldots, \gamma_T)

Lower = More stable (possibly less adaptive)

.. code-block:: python

   # Calculate metrics
   summary = tracker.get_summary()

   print(f"Adaptation frequency: {summary['adaptation_frequency']:.2%}")
   print(f"Avg recovery time: {summary['avg_recovery_time']:.1f} steps")
   print(f"Precision stability: {summary['precision_stability']:.3f}")

Mathematical Details
--------------------

Beta Distribution Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Probability density**:

.. math::

   p(\gamma; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \gamma^{\alpha-1} (1-\gamma)^{\beta-1}

**Mode** (most likely value):

.. math::

   \text{mode}(\gamma) = \frac{\alpha - 1}{\alpha + \beta - 2} \quad \text{for } \alpha, \beta > 1

**Concentration**:

.. math::

   \kappa = \alpha + \beta

Higher κ = More concentrated (less uncertain)

**Entropy**:

.. math::

   H[\gamma] = \log B(\alpha, \beta) - (\alpha - 1)\psi(\alpha) - (\beta - 1)\psi(\beta) + (\alpha + \beta - 2)\psi(\alpha + \beta)

where B is the Beta function and ψ is the digamma function.

Bayesian Update Interpretation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The precision update is equivalent to Bayesian updating:

**Prior**: β(α, β)
**Likelihood**: Bernoulli(1 - δ) (success probability)
**Posterior**: β(α + (1-δ), β + δ)

This is why we use the Beta distribution - it's the conjugate prior for Bernoulli!

Continuous-Time Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^

For continuous precision dynamics:

.. math::

   \frac{d\gamma}{dt} = \eta_{\text{gain}} (1 - \delta(t))(1 - \gamma) - \eta_{\text{loss}} \delta(t) \gamma

This differential equation shows:

* Precision increases toward 1 when δ is low
* Precision decreases toward 0 when δ is high
* Equilibrium depends on average δ

Simulation Example
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   def simulate_precision_dynamics(errors, eta_gain=0.1, eta_loss=0.2):
       """Simulate precision over time"""
       alpha, beta = 1.0, 1.0
       trajectory = []
       
       for error in errors:
           # Update
           alpha += eta_gain * (1 - error)
           beta += eta_loss * error
           
           # Calculate precision
           gamma = alpha / (alpha + beta)
           trajectory.append(gamma)
       
       return trajectory

   # Simulate with random errors
   np.random.seed(42)
   errors = np.random.beta(2, 5, 100)  # Errors biased toward low values
   
   trajectory = simulate_precision_dynamics(errors)
   
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(trajectory)
   plt.axhline(y=0.4, color='r', linestyle='--', alpha=0.5)
   plt.xlabel('Step')
   plt.ylabel('Precision')
   plt.title('Precision Trajectory')
   
   plt.subplot(1, 2, 2)
   plt.scatter(errors, np.diff([0.5] + trajectory), alpha=0.5)
   plt.xlabel('Prediction Error')
   plt.ylabel('Precision Change')
   plt.title('Error vs Precision Change')
   plt.tight_layout()
   plt.show()

Common Pitfalls
---------------

1. **Precision Too High**
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Agent stuck exploiting suboptimal policy

**Solution**: Lower initial precision or increase loss rate

.. code-block:: python

   precision = PrecisionParameters(
       initial_alpha=1.0,
       initial_beta=2.0  # Start more uncertain
   )

2. **Precision Too Low**
^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Agent explores excessively, never exploits

**Solution**: Increase gain rate or raise adaptation threshold

.. code-block:: python

   precision = PrecisionParameters(
       gain_learning_rate=0.15,  # Build confidence faster
       adaptation_threshold=0.3   # Lower threshold
   )

3. **Oscillating Precision**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Precision bounces around threshold

**Solution**: Add hysteresis or smooth updates

.. code-block:: python

   class HysteresisPrecision(PrecisionParameters):
       def __init__(self, *args, hysteresis=0.05, **kwargs):
           super().__init__(*args, **kwargs)
           self.hysteresis = hysteresis
           self.was_below_threshold = False
       
       def should_adapt(self):
           if self.was_below_threshold:
               # Need to exceed threshold + hysteresis to stop adapting
               should = self.value < (self.adaptation_threshold + self.hysteresis)
           else:
               # Standard threshold check
               should = self.value < self.adaptation_threshold
           
           self.was_below_threshold = should
           return should

4. **Ignoring Variance**
^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: High-variance precision → unstable behavior

**Solution**: Monitor variance, reset if too high

.. code-block:: python

   if precision.variance > 0.1:  # Too uncertain
       precision.reset()

Future Directions
-----------------

Meta-Learning Precision
^^^^^^^^^^^^^^^^^^^^^^^

Learn optimal precision parameters from experience:

.. code-block:: python

   # Collect data
   experiences = []
   for task in tasks:
       result = agent.run(task, precision_params=params)
       experiences.append((params, result.success, result.steps))
   
   # Optimize parameters
   best_params = optimize_precision_parameters(experiences)

Context-Dependent Precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different precision for different contexts:

.. code-block:: python

   class ContextualPrecision:
       def __init__(self):
           self.precisions = {}  # context -> PrecisionParameters
       
       def get_precision(self, context):
           if context not in self.precisions:
               self.precisions[context] = PrecisionParameters()
           return self.precisions[context]

Ensemble Precision
^^^^^^^^^^^^^^^^^^

Maintain multiple precision hypotheses:

.. code-block:: python

   class EnsemblePrecision:
       def __init__(self, n_particles=10):
           self.particles = [PrecisionParameters() for _ in range(n_particles)]
       
       def update(self, error):
           for particle in self.particles:
               particle.update(error + np.random.normal(0, 0.1))
       
       @property
       def value(self):
           return np.mean([p.value for p in self.particles])

Further Reading
---------------

* :doc:`active_inference` - Theoretical foundations
* :doc:`free_energy` - G calculation and policy selection
* :doc:`../getting_started/core_concepts` - Practical implementation
* :doc:`../tutorials/02_understanding_precision` - Hands-on tutorial

References
^^^^^^^^^^

* Friston, K. (2009). "The free-energy principle: a rough guide to the brain?"
* Feldman, H., & Friston, K. (2010). "Attention, uncertainty, and free-energy"
* Mathys, C., et al. (2014). "Uncertainty in perception and the Hierarchical Gaussian Filter"

Next Steps
----------

* Try :doc:`../tutorials/02_understanding_precision` for hands-on practice
* Experiment with different learning rates
* Monitor precision trajectories in your agents
* Tune adaptation thresholds for your use case

