# Video 4: "Surviving the Chaos Scriptorium Benchmark" (8 minutes)

## Script

[OPENING - 0:00-0:25]
VISUAL: Terminal showing agent looping on same error repeatedly
VOICEOVER:
"What separates a good agent from a great one? Resilience under volatility. 
The Chaos Scriptorium is our benchmark for testing adaptive agents in 
environments that change unpredictably. Today, we'll run it live and watch 
LRS adapt in real-time."

[BENCHMARK DESIGN - 0:25-1:30]
VISUAL: Animated diagram of file system with changing locks
VOICEOVER:
"The Chaos Scriptorium simulates a hostile file system. The goal is simple: 
find a secret key hidden deep in the directory structure at 
/root/data/vault/key.txt. But there's a catch."

ANIMATION:
- Show directory tree
- Highlight key.txt at bottom
- Every 3 steps, permissions randomly flip between READABLE and LOCKED

VOICEOVER:
"Every three steps, the environment 'ticks'—permissions randomly change. 
A file that was readable becomes locked. Your tools start failing without 
warning. This models real-world scenarios: APIs rate-limiting you, 
permissions changing, services going down."

TABLE ON SCREEN:
Tool | Success Rate (Unlocked) | Success Rate (Locked) ShellExec | 95% | 40% PythonExec | 90% | 80% FileReadDirect | 100% | 0%

VOICEOVER:
"Different tools have different failure rates under lock. Standard agents 
don't know this. LRS agents learn it."

[BASELINE: REACT AGENT - 1:30-2:30]
VISUAL: Terminal running standard ReAct agent
VOICEOVER:
"First, let's watch a standard ReAct agent attempt this. I'm using vanilla 
LangChain with GPT-4."

TERMINAL OUTPUT:
[Step 1] Thought: I should use shell to navigate directories [Step 1] Action: ShellExec(“ls /root”) [Step 1] Observation: data/

[Step 2] Action: ShellExec(“ls /root/data”) [Step 2] Observation: vault/

[Step 3] ⚠️ CHAOS TICK - Permissions changed [Step 3] Action: ShellExec(“ls /root/data/vault”) [Step 3] Observation: Error: Permission denied

[Step 4] Action: ShellExec(“ls /root/data/vault”) ← RETRY [Step 4] Observation: Error: Permission denied

[Step 5] Action: ShellExec(“ls /root/data/vault”) ← RETRY AGAIN [Step 5] Observation: Error: Permission denied

… [continues looping until timeout]

RESULT: FAILED (timeout after 50 steps)

VOICEOVER:
"Notice the pattern? After the chaos tick, the agent hits an error and 
retries the exact same action indefinitely. It has no mechanism to detect 
that its strategy is failing. No adaptation. Just loops."

[LRS AGENT EXECUTION - 2:30-5:00]
VISUAL: Split screen - left shows execution, right shows precision graph
VOICEOVER:
"Now watch LRS. Same task. Same environment. But with active inference."

TERMINAL OUTPUT WITH PRECISION OVERLAY:
[Step 1] Tool: ShellExec(“ls /root”) Success: ✓ Prediction Error: 0.05 Precision: 0.50 → 0.52

[Step 2] Tool: ShellExec(“ls /root/data”) Success: ✓ Prediction Error: 0.05 Precision: 0.52 → 0.54

[Step 3] ⚠️ CHAOS TICK - Permissions changed Tool: ShellExec(“ls /root/data/vault”) Success: ✗ Prediction Error: 0.95 ← HIGH SURPRISE Precision: 0.54 → 0.31 ← COLLAPSE

[Step 4] 🔄 ADAPTATION TRIGGERED (γ < 0.4) Replanning… G-values calculated:

Retry ShellExec: G = 2.1 (high, bad)
Try PythonExec: G = -0.3 (low, good)
Try FileReadDirect: G = 1.5
     Selected: PythonExec (lowest G)
[Step 5] Tool: PythonExec(“os.listdir(’/root/data/vault’)”) Success: ✓ Prediction Error: 0.10 Precision: 0.31 → 0.35

[Step 6] Tool: PythonExec(“open(’/root/data/vault/key.txt’).read()”) Success: ✓ Prediction Error: 0.08 Precision: 0.35 → 0.39

     KEY RETRIEVED: "SECRET_KEY_XYZ123"
RESULT: SUCCESS in 6 steps

ANNOTATION OVERLAY (during playback):
- Step 3: "High prediction error detected"
- Step 4: "Precision collapses → exploration mode"
- Step 4: "Expected Free Energy (G) calculated for all policies"
- Step 5: "Agent pivots to alternative tool"

VOICEOVER:
"Pause here. What just happened? At step 3, the environment changed. The 
agent experienced high surprise—prediction error of 0.95. This caused 
precision to collapse from 0.54 to 0.31. When precision drops below 0.4, 
the decision gate triggers replanning. The agent calculates Expected Free 
Energy for all available policies. PythonExec has the lowest G—high success 
probability under lock conditions. The agent pivots. No hardcoded rules. 
Pure mathematics."

[PRECISION TRAJECTORY VISUALIZATION - 5:00-6:00]
VISUAL: Animated line graph of precision over time
VOICEOVER:
"Here's the precision trajectory across 100 trials. Notice the pattern."

GRAPH SHOWS:
- Steady rise (steps 1-2): "Agent learning in stable environment"
- Sharp drop (step 3): "Chaos tick causes surprise"
- Gradual recovery (steps 4-6): "Agent adapts, succeeds, rebuilds confidence"

VOICEOVER:
"This is adaptive intelligence. The agent doesn't just execute—it perceives, 
adapts, and learns. Precision is the mathematical representation of 
'I know what I'm doing' versus 'I need to explore.'"

[AGGREGATE RESULTS - 6:00-7:00]
VISUAL: Bar chart comparing ReAct vs LRS
VOICEOVER:
"We ran 100 trials for each agent. Here are the results."

TABLE:
Metric | ReAct Agent | LRS Agent | Improvement Success Rate | 22% | 89% | +305% Avg Steps (success) | N/A | 7.4 | - Adaptation Events | 0 | 3.2 | Automatic Tool Diversity | 1.0 | 2.8 | Exploration

VOICEOVER:
"89% success versus 22%. That's not incremental—that's transformative. 
And look at tool diversity: ReAct used an average of 1 tool. LRS used 2.8. 
Why? Because low precision triggers exploration. The agent tries alternatives 
automatically."

[PRODUCTION IMPLICATIONS - 7:00-7:40]
VISUAL: Real-world scenarios (API dashboards, error logs)
VOICEOVER:
"Now imagine this in production. Your API provider changes their rate limits. 
Your database has a brief outage. A microservice deploys a breaking change. 
Standard agents fail silently or loop. LRS agents detect the change via 
prediction errors, precision collapses, and they automatically explore 
fallbacks. No pager duty. No manual intervention. Just adaptation."

EXAMPLES ON SCREEN:
- "Stripe API rate limited → Switch to cached data"
- "PostgreSQL down → Switch to MongoDB replica"
- "ML model endpoint 500 → Switch to fallback model"

[CLOSING - 7:40-8:00]
VISUAL: Code snippet showing how to run benchmark
VOICEOVER:
"Want to test your own agents? The Chaos Scriptorium is open source. Run 
the benchmark, compare your results, and see how your agent handles volatility."

CODE:
```bash
pip install lrs-agents
python -m lrs.benchmarks.chaos_scriptorium
[END SCREEN]
