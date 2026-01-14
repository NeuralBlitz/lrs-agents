
# Video 6: "Real-Time Agent Monitoring with the LRS Dashboard" (6 minutes)

## Script

[OPENING - 0:00-0:20]
VISUAL: Black box agent running with no visibility
VOICEOVER:
"You've deployed an LRS agent to production. It's running. But can you see 
what it's thinking? Standard agents are black boxes—you see actions, but 
not the internal decision-making. The LRS dashboard changes that. Real-time 
precision tracking. G-space visualization. Adaptation timelines. Full 
transparency."

[DASHBOARD LAUNCH - 0:20-0:45]
VISUAL: Terminal showing dashboard startup
VOICEOVER:
"Launching the dashboard is one command."

TERMINAL:
```bash
lrs-monitor --agent-id production_agent_1
BROWSER OPENS to localhost:8501

VOICEOVER: “The dashboard connects to your agent’s state tracker and streams updates in real-time. Let’s walk through each visualization.”

[PRECISION TRAJECTORIES - 0:45-2:00] VISUAL: Three-line chart showing hierarchical precision VOICEOVER: “First, precision trajectories. Remember, LRS tracks precision at three levels: abstract, planning, and execution. This chart shows all three over time.”

CHART SHOWS:

Blue line (abstract): Slow-moving, around 0.7
Orange line (planning): Medium volatility, 0.4-0.6
Green line (execution): High volatility, 0.2-0.8
VOICEOVER: “Notice how they move at different speeds. Abstract level barely changes— that’s your long-term goal confidence. Execution level spikes and drops rapidly—that’s tool-level surprise. And watch what happens when there’s a big failure.”

ANIMATION:

Step 15: Execution line drops sharply (0.7 → 0.3)
Step 16: Planning line drops slightly (0.6 → 0.5)
Abstract line unchanged
VOICEOVER: “High error at execution propagates to planning, but not to abstract. This prevents the agent from abandoning its goal due to a single tool failure. The hierarchical structure provides stability.”

[G-SPACE MAP - 2:00-3:15] VISUAL: Scatter plot with epistemic vs pragmatic axes VOICEOVER: “Second visualization: the G-space map. This shows why the agent chose each policy.”

SCATTER PLOT:

X-axis: Epistemic Value (information gain)
Y-axis: Pragmatic Value (expected reward)
Each point is a candidate policy
Selected policy highlighted with star
VOICEOVER: “Remember, G equals epistemic value minus pragmatic value. Policies in the top-right are ideal—high reward and high information. Bottom-left are terrible—low reward, no learning. Watch what the agent prefers at different precision levels.”

ANIMATION:

High precision (γ=0.8): Selected policy in top-left (high reward, low info)
Low precision (γ=0.3): Selected policy in top-right (high reward, high info)
VOICEOVER: “With high precision, the agent picks high-reward, low-exploration policies. With low precision, it picks policies that gather information even if reward is uncertain. This is the exploration-exploitation trade-off visualized.”

[PREDICTION ERROR STREAM - 3:15-4:15] VISUAL: Area chart showing prediction errors over time VOICEOVER: “Third: prediction error stream. This is the agent’s ‘surprise timeline.’”

AREA CHART:

Baseline around 0.1-0.2 (normal operation)
Spikes to 0.9 at adaptation events
Vertical markers show tool failures
VOICEOVER: “Small errors are normal—tools perform as expected. But look at step 22.”

HIGHLIGHT ON SCREEN:

Step 22: Error spikes to 0.95
Vertical red line: “ADAPTATION EVENT”
Tooltip: “ShellExec failed unexpectedly, precision collapsed, switched to PythonExec”
VOICEOVER: “That spike is a surprise. The tooltip shows exactly what happened: which tool failed, how much precision dropped, and what action the agent took. This is your audit trail for adaptation events.”

[ADAPTATION TIMELINE - 4:15-5:00] VISUAL: Chronological list of adaptation events VOICEOVER: “Fourth: the adaptation timeline. A detailed log of every time the agent changed its mind.”

TIMELINE ENTRIES:

2025-01-14 10:23:45 - ADAPTATION EVENT #1
  Trigger: High prediction error (0.92)
  Tool failed: api_fetch
  Precision: execution 0.75 → 0.41
  Action: Replanned, switched to cache_fetch
  Outcome: Success

2025-01-14 10:24:12 - ADAPTATION EVENT #2
  Trigger: Hierarchical propagation
  Tool failed: parse_json
  Precision: planning 0.68 → 0.52
  Action: Escalated to abstract level, revised goal
  Outcome: In progress
VOICEOVER: “Each entry shows the trigger, the precision change, and the outcome. This is production-grade observability. You can see not just what the agent did, but why it did it.”

[LIVE DEMO - 5:00-5:45] VISUAL: Split screen - agent running on left, dashboard on right VOICEOVER: “Let’s see it live. I’m running the Chaos Scriptorium benchmark. Watch the dashboard update in real-time.”

EXECUTION:

[Agent executes]
Step 3: ✗ ShellExec failed

[Dashboard updates immediately]
- Precision trajectory: Green line drops
- G-space map: New policy selected
- Error stream: Red spike appears
- Timeline: New adaptation entry added
VOICEOVER: “The moment the tool fails, the dashboard reflects it. Precision drops. Error spikes. Adaptation logged. This is real-time transparency.”

[PRODUCTION USE CASE - 5:45-5:55] VISUAL: Production deployment diagram VOICEOVER: “In production, run the dashboard on a separate service. Point it at your agent’s state store. Monitor multiple agents in parallel. Set up alerts for precision thresholds.”

[CLOSING - 5:55-6:00] CODE ON SCREEN:

# Launch for remote agent
lrs-monitor --agent-id prod_agent_1 --state-url redis://...

# Dashboard available at http://localhost:8501
[END SCREEN]
