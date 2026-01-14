# Video 5: "LLM Integration: The Variational Proposal Mechanism" (10 minutes)

## Script

[OPENING - 0:00-0:30]
VISUAL: Code showing LLM generating actions directly
VOICEOVER:
"There's a fundamental problem with how most agentic AI systems use LLMs. 
They ask the language model to decide what to do, then execute blindly. 
This creates hallucinated confidence—the LLM says it's 90% sure, but has 
no actual model of tool reliability. Today we're fixing that by making LLMs 
propose, not decide."

[THE PROBLEM: LLM AS ORACLE - 0:30-1:30]
VISUAL: Diagram showing standard LLM agent flow
VOICEOVER:
"Standard approach: Give the LLM your tools, ask it to choose the best one, 
execute whatever it says. The problem? The LLM has no way to know that 
your API is currently rate-limited, or that the file it's trying to read 
was just locked."

CODE EXAMPLE:
```python
# Standard approach (problematic)
action = llm.generate("What should I do next?")
# LLM returns: "Use API tool with 95% confidence"
result = execute(action)
# But API is down! LLM couldn't know.
VOICEOVER: “The LLM is overconfident because it’s making decisions without access to execution history. It doesn’t know precision. It can’t calculate Expected Free Energy. It just guesses.”

[THE SOLUTION: VARIATIONAL PROPOSAL - 1:30-3:00] VISUAL: New diagram showing LLM → proposals → G evaluation → selection VOICEOVER: “LRS flips this. The LLM doesn’t decide—it proposes. We ask it to generate 3-5 diverse policy candidates spanning exploration and exploitation. Then the math evaluates them via Expected Free Energy. The LLM provides the creativity. The math provides the rigor.”

WORKFLOW DIAGRAM:

1. LLM generates proposals ──→ 2. Calculate G for each
                                 ↓
4. Precision-weighted selection ←─ 3. Rank by G value
CODE:

# LRS approach
proposals = llm.generate_policy_samples(state, n=5)  # LLM proposes
G_values = [calculate_G(π) for π in proposals]      # Math evaluates
selected = precision_weighted_selection(proposals, G_values, γ)  # Select
VOICEOVER: “This is variational inference. The LLM approximates the posterior distribution over policies. The G calculation is the objective function. Precision controls the temperature of selection.”

[META-COGNITIVE PROMPTING - 3:00-5:00] VISUAL: Actual prompt shown on screen with highlighting VOICEOVER: “The key is how we prompt the LLM. We don’t just ask for actions—we ask for epistemic metadata. Self-assessed success probability. Expected information gain. Strategy classification.”

PROMPT ON SCREEN:

You are a Bayesian policy generator for an Active Inference agent.

Current Precision (γ): 0.35 (LOW - World model unreliable)

STRATEGIC GUIDANCE: EXPLORATION MODE
Your proposal strategy:
1. Prioritize information - Focus on reducing uncertainty
2. Test assumptions - Include diagnostic actions
3. Accept risk - Exploratory policies may have lower success
4. Question patterns - Previous strategies may be outdated

Generate 5 policy proposals in JSON format:
{
  "proposals": [
    {
      "policy_id": 1,
      "tools": ["tool_a", "tool_b"],
      "estimated_success_prob": 0.7,
      "expected_information_gain": 0.8,
      "strategy": "explore",
      "rationale": "Test alternative approach",
      "failure_modes": ["Tool timeout", "Parse error"]
    }
  ]
}
VOICEOVER: “Notice the precision value at the top. The prompt adapts based on the agent’s confidence. Low precision? We tell the LLM to focus on exploration. High precision? We tell it to exploit known patterns. This is meta-cognitive prompting—making the LLM aware of the agent’s epistemic state.”

[LIVE DEMO - 5:00-7:30] VISUAL: Jupyter notebook with real LLM calls VOICEOVER: “Let’s see this in action. I’ll use Claude Sonnet 4, precision is currently 0.4—medium confidence. Watch what proposals we get.”

NOTEBOOK EXECUTION:

from lrs.inference.llm_policy_generator import LLMPolicyGenerator
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

generator = LLMPolicyGenerator(llm, registry)

proposals = generator.generate_proposals(
    state={'goal': 'Fetch user data'},
    precision=0.4  # Medium precision
)

for p in proposals:
    print(f"\nPolicy {p['policy_id']}: {p['strategy']}")
    print(f"  Tools: {' → '.join(p['tools'])}")
    print(f"  Success prob: {p['estimated_success_prob']}")
    print(f"  Info gain: {p['expected_information_gain']}")
    print(f"  Rationale: {p['rationale']}")
OUTPUT:

Policy 1: exploit
  Tools: api_fetch → parse_json
  Success prob: 0.85
  Info gain: 0.2
  Rationale: Direct approach using proven tools

Policy 2: explore
  Tools: cache_check → api_fetch → validate
  Success prob: 0.65
  Info gain: 0.7
  Rationale: Diagnostic path to test cache state

Policy 3: balanced
  Tools: health_check → api_fetch
  Success prob: 0.75
  Info gain: 0.4
  Rationale: Verify service health before fetching
VOICEOVER: “Three proposals. Notice the diversity. Policy 1 exploits—high success, low info gain. Policy 2 explores—lower success, but tests assumptions. Policy 3 balances. The LLM generated this spread because we prompted for diversity and adapted to medium precision.”

[G EVALUATION - 7:30-8:30] VISUAL: Calculation of G for each proposal VOICEOVER: “Now the math takes over. We calculate Expected Free Energy for each.”

CALCULATION ON SCREEN:

from lrs.core.free_energy import calculate_expected_free_energy

for proposal in proposals:
    G = calculate_expected_free_energy(
        policy=proposal['tools'],
        state=state,
        preferences={'success': 5.0, 'error': -3.0}
    )
    print(f"Policy {proposal['policy_id']}: G = {G:.2f}")
OUTPUT:

Policy 1 (exploit): G = -2.1  ← Lowest G (best)
Policy 2 (explore): G = 0.3
Policy 3 (balanced): G = -1.4
VOICEOVER: “Policy 1 has the lowest G—highest expected reward minus epistemic value. But here’s where precision matters. Watch what happens when we select via softmax.”

CODE:

selected = precision_weighted_selection(proposals, G_values, precision=0.4)
print(f"Selected: Policy {selected}")
OUTPUT:

Selected: Policy 3 (balanced)
VOICEOVER: “With medium precision, the softmax doesn’t just pick the lowest G. It samples probabilistically. This allows exploration even when exploitation looks better. The agent hedges.”

[TEMPERATURE ADAPTATION - 8:30-9:15] VISUAL: Graph showing temperature vs precision VOICEOVER: “One more trick: we adapt the LLM’s temperature based on precision.”

CODE:

def adapt_temperature(precision):
    return base_temp * (1.0 / (precision + 0.1))

# Low precision → high temperature
print(adapt_temperature(0.2))  # 3.5 (very exploratory)

# High precision → low temperature
print(adapt_temperature(0.9))  # 0.7 (focused)
GRAPH SHOWS:

X-axis: Precision (0 to 1)
Y-axis: LLM Temperature
Curve: Hyperbolic decay
VOICEOVER: “Low precision means uncertainty. High temperature means diverse, creative proposals. High precision means confidence. Low temperature means focused, conservative proposals. The entire system adapts together.”

[PRODUCTION EXAMPLE - 9:15-9:45] VISUAL: Real deployment dashboard VOICEOVER: “In production, this means your agent automatically becomes more creative when confused and more focused when confident. No hyperparameter tuning. No manual switches. Just mathematical adaptation.”

EXAMPLE:

Agent encounters new API endpoint (precision drops)
  → Temperature increases
  → LLM generates more diverse proposals
  → Agent explores alternatives
  → Finds working approach
  → Precision recovers
  → Temperature decreases
  → Agent exploits successful pattern
[CLOSING - 9:45-10:00] VISUAL: Summary diagram of full system VOICEOVER: “To recap: LLMs propose, not decide. Prompts adapt to precision. Expected Free Energy evaluates. Precision-weighted selection chooses. And temperature adapts automatically. This is how you integrate language models into mathematically grounded agents.”

[END SCREEN]
