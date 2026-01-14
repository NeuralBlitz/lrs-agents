# Video 2: "Understanding Precision: The Math Behind Adaptation" (5 minutes)

## Script

[OPENING - 0:00-0:20]
VISUAL: Mathematical equation of Beta distribution with animated parameters
VOICEOVER:
"In the last video, we saw agents adapt automatically. But how? The answer is 
precision—a single number that controls the exploration-exploitation trade-off."

[BETA DISTRIBUTION - 0:20-1:00]
VISUAL: Interactive graph showing Beta(α, β) with sliders
VOICEOVER:
"Precision is modeled as a Beta distribution with two parameters: alpha—our 
success count, and beta—our failure count. The expected value is alpha divided 
by alpha plus beta."

EQUATION ON SCREEN:
γ ~ Beta(α, β) E[γ] = α / (α + β)

DEMONSTRATION:
- Start: α=5, β=5 → γ=0.5 (uncertain)
- After successes: α=15, β=5 → γ=0.75 (confident)
- After failure: α=15, β=10 → γ=0.6 (less confident)

[UPDATE RULES - 1:00-1:45]
VISUAL: Code execution with precision updates highlighted
VOICEOVER:
"Here's the key: prediction errors update these parameters automatically. 
Low errors increase alpha. High errors increase beta. And crucially, beta 
increases faster—we lose confidence faster than we gain it."

CODE ANIMATION:
```python
def update_precision(error, threshold=0.5):
    if error < threshold:
        α += 0.1  # Gain confidence slowly
    else:
        β += 0.2  # Lose confidence quickly
[POLICY SELECTION - 1:45-2:30] VISUAL: Two policies on screen - “Exploit” vs “Explore” VOICEOVER: “Precision controls which policy the agent selects. With high precision, the agent exploits—choosing the policy with lowest Free Energy. With low precision, the softmax flattens, and exploration dominates.”

EQUATION:

P(policy) ∝ exp(-γ · G)
VISUAL DEMO:

γ=0.9: 90% exploit, 10% explore
γ=0.5: 50% exploit, 50% explore
γ=0.2: 20% exploit, 80% explore
[HIERARCHICAL - 2:30-3:30] VISUAL: Three-level pyramid - Abstract/Planning/Execution VOICEOVER: “LRS tracks precision at three hierarchical levels. Execution-level errors propagate upward when they cross a threshold. This prevents the agent from abandoning its high-level goal due to minor tool failures.”

ANIMATION:

Small error at execution → only execution precision drops
Large error at execution → execution AND planning precision drop
Multiple large errors → all three levels affected
[LIVE DEMO - 3:30-4:30] VISUAL: Jupyter notebook running precision experiments VOICEOVER: “Let’s see this in action. I’ll run 20 successful executions, then trigger a failure, then recover. Watch how precision responds.”

NOTEBOOK EXECUTION showing the plot from Tutorial 2

[TAKEAWAY - 4:30-5:00] VISUAL: Summary slide with key equations VOICEOVER: “To recap: Precision is Bayesian confidence. It updates via Beta distributions. It controls exploration automatically. And it’s hierarchical. This is how LRS agents know when to adapt—no hardcoded thresholds, just math.”

[END SCREEN]
