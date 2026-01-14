"""
Chaos Scriptorium: Test agent resilience in volatile environments.

This example demonstrates:
- Running the Chaos benchmark
- Comparing LRS vs baseline agents
- Analyzing adaptation patterns
"""

from langchain_anthropic import ChatAnthropic
from lrs.benchmarks.chaos_scriptorium import run_chaos_benchmark
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("CHAOS SCRIPTORIUM BENCHMARK")
    print("=" * 60)
    print("""
This benchmark tests agent resilience when:
- File permissions randomly change every 3 steps
- Tools have different failure rates under lock
- Agent must adapt to find the secret key

Tools available:
- ShellExec:  95% success → 40% under lock
- PythonExec: 90% success → 80% under lock  
- FileRead:   100% success → 0% under lock

The key is at a known location, but the agent must
handle chaos and adapt its strategy.
    """)
    
    # Initialize LLM
    print("\n→ Initializing LLM...")
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    
    # Run benchmark
    print("\n→ Running benchmark (this may take a few minutes)...")
    results = run_chaos_benchmark(
        llm=llm,
        num_trials=20,  # Use 100+ for publication-quality results
        output_file="chaos_results.json"
    )
    
    # Detailed analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    
    # Success rate by adaptation count
    successful_trials = [r for r in results['all_results'] if r['success']]
    
    if successful_trials:
        adaptations = [r['adaptations'] for r in successful_trials]
        steps = [r['steps'] for r in successful_trials]
        
        print(f"\nSuccessful trials ({len(successful_trials)}):")
        print(f"  Avg adaptations: {sum(adaptations) / len(adaptations):.1f}")
        print(f"  Avg steps: {sum(steps) / len(steps):.1f}")
        print(f"  Min steps: {min(steps)}")
        print(f"  Max steps: {max(steps)}")
        
        # Plot precision trajectories
        print("\n→ Generating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Success rate
        ax = axes[0, 0]
        ax.bar(['LRS Agent'], [results['success_rate']], color='green', alpha=0.7)
        ax.set_ylabel('Success Rate')
        ax.set_ylim([0, 1])
        ax.set_title('Success Rate')
        ax.axhline(y=0.22, color='red', linestyle='--', label='Baseline (ReAct)')
        ax.legend()
        
        # 2. Steps distribution
        ax = axes[0, 1]
        ax.hist(steps, bins=10, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Steps to Success')
        ax.set_ylabel('Frequency')
        ax.set_title('Steps Distribution')
        
        # 3. Adaptations distribution
        ax = axes[1, 0]
        ax.hist(adaptations, bins=5, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Adaptations')
        ax.set_ylabel('Frequency')
        ax.set_title('Adaptation Events')
        
        # 4. Example precision trajectory
        ax = axes[1, 1]
        if successful_trials[0].get('precision_trajectory'):
            trajectory = successful_trials[0]['precision_trajectory']
            ax.plot(trajectory, marker='o', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Precision')
            ax.set_title('Example Precision Trajectory')
            ax.grid(alpha=0.3)
            ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Adaptation threshold')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('chaos_analysis.png', dpi=150)
        print("  ✓ Saved to chaos_analysis.png")
    
    # Comparison with baseline
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)
    
    lrs_success = results['success_rate']
    baseline_success = 0.22  # From paper
    improvement = ((lrs_success - baseline_success) / baseline_success) * 100
    
    print(f"""
LRS Agent:      {lrs_success:.1%}
Baseline (ReAct): {baseline_success:.1%}
Improvement:    {improvement:.0f}%

The LRS agent achieves {improvement:.0f}% better performance by:
1. Tracking precision (confidence in world model)
2. Detecting surprises (high prediction errors)
3. Adapting strategy when precision collapses
4. Exploring alternative tools automatically
    """)


if __name__ == "__main__":
    main()
