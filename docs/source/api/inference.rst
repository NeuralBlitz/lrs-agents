Inference API
=============

The inference module provides components for LLM-based policy generation and evaluation.

Meta-Cognitive Prompting
------------------------

.. automodule:: lrs.inference.prompts
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.inference.prompts.PromptContext
   :members:
   :show-inheritance:

   Context for generating meta-cognitive prompts.

   **Attributes:**

   * **precision** (float): Current precision value
   * **recent_errors** (List[float]): Recent prediction errors
   * **available_tools** (List[str]): Tools the agent can use
   * **goal** (str): Current task goal
   * **state** (dict): Current belief state
   * **tool_history** (List[dict]): Execution history

.. autoclass:: lrs.inference.prompts.StrategyMode
   :members:
   :show-inheritance:

   Policy generation strategy based on precision.

   * **EXPLOITATION**: High precision → Prioritize reward
   * **EXPLORATION**: Low precision → Prioritize information gain
   * **BALANCED**: Medium precision → Balance both

.. autoclass:: lrs.inference.prompts.MetaCognitivePrompter
   :members:
   :special-members: __init__
   :show-inheritance:

   Generates precision-adaptive prompts for LLM policy generation.

   **Methods:**

   .. automethod:: generate_prompt

   **Example:**

   .. code-block:: python

      from lrs.inference.prompts import MetaCognitivePrompter, PromptContext

      prompter = MetaCognitivePrompter()
      
      context = PromptContext(
          precision=0.3,  # Low precision
          recent_errors=[0.8, 0.9],
          available_tools=['api', 'cache', 'db'],
          goal='Fetch user data',
          state={},
          tool_history=[]
      )
      
      prompt = prompter.generate_prompt(context)
      # Generates exploration-focused prompt

LLM Policy Generator
--------------------

.. automodule:: lrs.inference.llm_policy_generator
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.inference.llm_policy_generator.PolicyProposal
   :members:
   :show-inheritance:

   Single policy proposal from LLM.

   **Attributes:**

   * **policy_id** (int): Unique identifier
   * **tools** (List[str]): Tool names in sequence
   * **estimated_success_prob** (float): LLM's self-assessed success probability
   * **expected_information_gain** (float): Expected epistemic value
   * **strategy** (str): "exploit", "explore", or "balanced"
   * **rationale** (str): Explanation of policy
   * **failure_modes** (List[str]): Potential failure scenarios

.. autoclass:: lrs.inference.llm_policy_generator.PolicyProposalSet
   :members:
   :show-inheritance:

   Set of 3-7 policy proposals from LLM.

   **Attributes:**

   * **proposals** (List[PolicyProposal]): Individual proposals
   * **current_uncertainty** (Optional[float]): LLM's uncertainty estimate
   * **known_unknowns** (List[str]): What the LLM doesn't know

.. autoclass:: lrs.inference.llm_policy_generator.LLMPolicyGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

   LLM-based variational proposal mechanism.

   **Methods:**

   .. automethod:: generate_proposals

   **Temperature Adaptation:**

   Temperature scales with precision:

   .. math::

      T = T_{base} \times \frac{1}{\gamma + 0.1}

   Low precision → High temperature → Diverse proposals

   **Example:**

   .. code-block:: python

      from lrs.inference.llm_policy_generator import LLMPolicyGenerator
      from langchain_anthropic import ChatAnthropic

      llm = ChatAnthropic(model="claude-sonnet-4-20250514")
      generator = LLMPolicyGenerator(llm, registry)
      
      proposals = generator.generate_proposals(
          state={'goal': 'Fetch data'},
          precision=0.5,
          num_proposals=5
      )
      
      for proposal in proposals:
          print(f"{proposal['strategy']}: {proposal['tool_names']}")

Functions
^^^^^^^^^

.. autofunction:: lrs.inference.llm_policy_generator.create_mock_generator

   Create mock generator for testing (doesn't require API key).

Hybrid Evaluator
----------------

.. automodule:: lrs.inference.evaluator
   :members:
   :undoc-members:
   :show-inheritance:

Classes
^^^^^^^

.. autoclass:: lrs.inference.evaluator.HybridGEvaluator
   :members:
   :special-members: __init__
   :show-inheritance:

   Hybrid evaluator combining mathematical G with LLM self-assessment.

   **Formula:**

   .. math::

      G_{hybrid} = (1 - \lambda) \cdot G_{math} + \lambda \cdot G_{llm}

   where :math:`\lambda = 1 - \gamma` (low precision → trust LLM more)

   **Methods:**

   .. automethod:: evaluate_hybrid
   .. automethod:: evaluate_all

   **Example:**

   .. code-block:: python

      from lrs.inference.evaluator import HybridGEvaluator

      evaluator = HybridGEvaluator()
      
      # Evaluate single proposal
      eval_result = evaluator.evaluate_hybrid(
          proposal=proposal_dict,
          state={},
          preferences={'success': 5.0},
          precision=0.5,
          historical_stats=registry.statistics
      )
      
      print(f"G_hybrid: {eval_result.total_G}")
      print(f"G_math: {eval_result.components['G_math']}")
      print(f"G_llm: {eval_result.components['G_llm']}")
      print(f"λ: {eval_result.components['lambda']}")

Functions
^^^^^^^^^

.. autofunction:: lrs.inference.evaluator.compare_math_vs_llm

   Debug utility to compare mathematical vs LLM G values.

