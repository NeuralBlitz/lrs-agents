# Video 8: "Multi-Agent LRS: Social Intelligence Preview" (9 minutes)

## Script

[OPENING - 0:00-0:30]
VISUAL: Single agent vs team of agents working together
VOICEOVER:
"Everything we've covered so far—precision, adaptation, tool composition—
works for single agents. But real-world tasks often require coordination. 
Multiple agents working together. Today we're previewing v0.3.0: multi-agent 
LRS with recursive theory-of-mind. Agents that don't just adapt to their 
environment, but to each other."

[THE PROBLEM: INDEPENDENT AGENTS - 0:30-1:30]
VISUAL: Two agents interfering with each other
VOICEOVER:
"Standard multi-agent systems treat other agents as part of the environment—
unpredictable noise. Agent A doesn't know why Agent B just failed. Agent B 
doesn't know Agent A is confused. They can't coordinate because they can't 
model each other."

EXAMPLE:
Agent A: “I’ll fetch data from API” Agent B: “I’ll also fetch data from API” ← Redundant!

[Both agents hit rate limit]

Agent A: “API failed, I’m confused” Agent B: “API failed, I’m confused”

[No communication, no learning from each other]

VOICEOVER:
"They're working against each other instead of with each other. No shared 
understanding. No trust tracking. No communication."

[THE SOLUTION: SOCIAL PRECISION - 1:30-3:00]
VISUAL: Diagram showing agent with two precision hierarchies
VOICEOVER:
"LRS v0.3.0 introduces social precision—confidence in other agents' models. 
Each agent tracks two types of precision: environmental (how well do I 
understand the world) and social (how well do I understand other agents)."

DIAGRAM:
Agent A’s Precision: ├── Environmental γ_env │ ├── Abstract: 0.8 │ ├── Planning: 0.6 │ └── Execution: 0.5 │ └── Social γ_social ├── Agent B: 0.7 ← High trust ├── Agent C: 0.4 ← Medium trust └── Agent D: 0.2 ← Low trust (unreliable)

VOICEOVER:
"Agent A maintains separate precision values for each other agent. When 
Agent B acts predictably, social precision increases. When Agent B surprises 
Agent A, social precision drops. This is mathematical trust."

[SOCIAL PREDICTION ERRORS - 3:00-4:15]
VISUAL: Code showing social precision updates
VOICEOVER:
"How do agents update social precision? Via social prediction errors—how 
well they predict each other's actions."

CODE:
```python
from lrs.multi_agent.social_precision import SocialPrecisionTracker

# Agent A tracks Agent B
tracker = SocialPrecisionTracker(
    agent_id="agent_a",
    other_agents=["agent_b", "agent_c"]
)

# Agent A predicts Agent B will use cache_fetch
predicted = "cache_fetch"

# Agent B actually uses api_fetch (surprise!)
observed = "api_fetch"

# Update social precision
new_precision = tracker.update_social_precision(
    other_agent_id="agent_b",
    predicted_action=predicted,
    observed_action=observed
)

print(f"Trust in Agent B: {new_precision:.3f}")  # Dropped from 0.7 to 0.5
VOICEOVER: “When Agent B acts unexpectedly, Agent A’s trust drops. This isn’t hardcoded— it’s Bayesian belief updating applied to social cognition.”

[COMMUNICATION AS ACTION - 4:15-5:30] VISUAL: G calculation including communication VOICEOVER: “Here’s the key insight: communication is an information-seeking action. Sending a message reduces social uncertainty. The agent doesn’t communicate because we told it to—it communicates because the math says it’s valuable.”

EQUATION:

G_total = G_env + α · G_social

Where:
G_social = Σ (1 - γ_social[agent_i])  ← Social uncertainty
VOICEOVER: “Total Free Energy includes social uncertainty. When Agent A is confused about Agent B (low γ_social), G_social is high. Sending a message to Agent B reduces G_social. The agent mathematically motivated to communicate.”

CODE EXAMPLE:

# Agent A's decision process
if should_communicate(other_agent="agent_b", threshold=0.5):
    # Social precision low → communicate
    send_message(
        to="agent_b",
        content="What's your current strategy?"
    )
DECISION LOGIC:

def should_communicate(other_agent_id, threshold=0.5):
    social_prec = social_precision[other_agent_id]
    env_prec = env_precision
    
    # Communicate when:
    # 1. Social precision is low (confused about other agent)
    # 2. Environmental precision is high (so problem is social)
    return social_prec < threshold and env_prec > 0.6
VOICEOVER: “Communication happens when Agent A understands the environment but not the other agent. This prevents chatty agents—they only talk when it reduces uncertainty.”

[LIVE DEMO: WAREHOUSE COORDINATION - 5:30-7:30] VISUAL: Simulation of warehouse robots VOICEOVER: “Let’s see this in a warehouse simulation. Three robots: Agent A (picker), Agent B (packer), Agent C (shipper). They need to coordinate package delivery.”

SIMULATION RUNS:

[Initial state: All agents have medium social precision (0.5)]

Step 1:
  Agent A: Picks package #123
  Agent B: Expects Agent A to pick #123 ✓
  Agent C: Observes
  
  Social precision: A↔B increases to 0.6

Step 2:
  Agent B: Starts packing #123
  Agent A: Expects Agent B to pack #123 ✓
  
  Social precision: B↔A increases to 0.7

Step 3:
  Agent C: Tries to ship #456 (different package!)
  Agent A: Expected #123 ✗ SURPRISE
  
  Social precision: A↔C drops to 0.3

Step 4:
  Agent A: should_communicate("agent_c") → True
  Agent A sends: "I picked #123, are you shipping #123?"
  Agent C responds: "Sorry, shipping #456. Will switch to #123."
  
  [Coordination restored via communication]

Step 5-10:
  All agents coordinate on #123
  Social precision recovers to 0.7-0.8
  Package delivered successfully
ANNOTATION OVERLAY:

“Trust builds through successful predictions”
“Surprise triggers communication”
“Communication reduces social Free Energy”
VOICEOVER: “Notice what happened. Agent C acted unexpectedly. Agent A’s social precision dropped. This triggered communication. Agent C explained its action. They coordinated. Social precision recovered. This is emergent collaboration.”

[RECURSIVE THEORY-OF-MIND - 7:30-8:30] VISUAL: Nested belief structure VOICEOVER: “The most powerful part: recursive theory-of-mind. Agent A models Agent B’s model of Agent A.”

DIAGRAM:

Agent A's beliefs:
├── My precision: 0.6
├── My model of Agent B:
│   ├── B's precision: 0.5 (my estimate)
│   └── B's model of me:
│       └── B thinks my precision is: 0.7 (my estimate of B's estimate)
VOICEOVER: “Agent A doesn’t just think ‘Agent B is unreliable.’ Agent A thinks ‘Agent B thinks I’m confident, but I’m actually uncertain. I should communicate my uncertainty to B so B can adjust.’ This is second-order theory-of-mind.”

CODE:

class RecursiveBeliefState:
    my_precision: float
    my_belief_about_other: Dict[str, float]  # B's precision
    my_belief_about_other_belief: Dict[str, float]  # B's belief about my precision
    
    def should_share_uncertainty(self, other_agent):
        # Share if: I'm uncertain, but other thinks I'm confident
        my_actual = self.my_precision
        other_thinks = self.my_belief_about_other_belief[other_agent]
        
        return my_actual < 0.5 and other_thinks > 0.7
[ROADMAP - 8:30-8:50] VISUAL: v0.3.0 feature list VOICEOVER: “This is coming in v0.3.0. We’ve built the social precision tracker. The communication tools. The recursive belief structures. Now we’re integrating it all and running multi-agent benchmarks.”

FEATURES:

v0.3.0 Roadmap:
✅ Social precision tracking
✅ Communication as ToolLens
✅ Shared world state
🚧 Multi-agent coordinator
🚧 Recursive theory-of-mind
🚧 Multi-agent dashboard
📋 Negotiation benchmark
📋 Collaborative task suite
[CLOSING - 8:50-9:00] VISUAL: Teams of agents working in harmony VOICEOVER: “Single-agent LRS gives you adaptation. Multi-agent LRS gives you coordination. From nervous systems to social intelligence. That’s v0.3.0.”

[END SCREEN]


