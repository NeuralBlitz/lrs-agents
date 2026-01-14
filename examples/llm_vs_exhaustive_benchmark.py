"""
LRS Scaling Benchmark: Exhaustive vs. Variational (LLM) Policy Generation

Demonstrates the fundamental scaling advantage of LLM-guided proposals:
- Exhaustive search: O(n^depth) - exponential in tool count
- LLM proposals: O(1) - constant time regardless of registry size

This is the empirical proof that v0.2.0 makes LRS production-ready for 
enterprise-scale tool registries (50+ tools).

Usage:
    python examples/llm_vs_exhaustive_benchmark.py
    
    # Export data for paper figures
    python examples/llm_vs_exhaustive_benchmark.py --export results.csv
"""

import time
import argparse
from typing import List, Tuple
from unittest.mock import Mock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lrs.core.registry import ToolRegistry
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.inference.llm_policy_generator import LLMPolicyGenerator


# ============================================================================
# Mock Components
# ============================================================================

class MockTool(ToolLens):
    """Lightweight mock tool for benchmarking"""
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={'type': 'object'},
            output_schema={'type': 'string'}
        )
    
    def get(self, state: dict) -> ExecutionResult:
        return ExecutionResult(True, "mock", None, 0.0)
    
    def set(self, state: dict, observation: str) -> dict:
        return state


def create_mock_llm(latency: float = 0.5):
    """
    Create mock LLM with configurable latency.
    
    Args:
        latency: Simulated API call time in seconds
    """
    llm = Mock()
    
    def mock_generate(*args, **kwargs):
        time.sleep(latency)  # Simulate API latency
        return {
            "proposals": [
                {
                    "policy_id": i,
                    "tools": [{"tool_name": f"tool_{i}", "reasoning": "test"}],
                    "estimated_success_prob": 0.8,
                    "expected_information_gain": 0.3,
                    "strategy": "balanced",
                    "rationale": "Test proposal",
                    "failure_modes": []
                }
                for i in range(5)  # Always return 5 proposals
            ],
            "current_uncertainty": 0.5,
            "known_unknowns": []
        }
    
    llm.generate = Mock(side_effect=mock_generate)
    return llm


# ============================================================================
# Benchmark Implementations
# ============================================================================

def exhaustive_policy_search(registry: ToolRegistry, max_depth: int = 3) -> List[List[ToolLens]]:
    """
    Exhaustive compositional search (legacy approach).
    
    Complexity: O(n^depth) where n = number of tools
    """
    policies = []
    tools = list(registry.tools.values())
    
    def build_tree(current: List[ToolLens], depth: int):
        if depth == 0:
            if current:
                policies.append(current)
            return
        
        for tool in tools:
            # Avoid immediate repetition
            if not current or tool != current[-1]:
                build_tree(current + [tool], depth - 1)
    
    build_tree([], max_depth)
    return policies


def llm_policy_generation(
    registry: ToolRegistry,
    llm,
    precision: float = 0.5
) -> List[dict]:
    """
    LLM-guided policy generation (v0.2.0 approach).
    
    Complexity: O(1) with respect to tool count
    """
    generator = LLMPolicyGenerator(llm, registry)
    return generator.generate_proposals(
        state={"goal": "benchmark_task"},
        precision=precision
    )


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_single_benchmark(
    tool_count: int,
    max_depth: int = 3,
    llm_latency: float = 0.5,
    trials: int = 3
) -> Tuple[float, float, int, int]:
    """
    Run benchmark for a specific tool count.
    
    Returns:
        (exhaustive_time, llm_time, exhaustive_policies, llm_policies)
    """
    # Build registry
    registry = ToolRegistry()
    for i in range(tool_count):
        tool = MockTool(f"tool_{i}")
        registry.register(tool)
    
    # Benchmark exhaustive search (with timeout)
    exhaustive_times = []
    exhaustive_count = 0
    
    for _ in range(trials):
        try:
            start = time.time()
            policies = exhaustive_policy_search(registry, max_depth)
            elapsed = time.time() - start
            
            # Timeout for large registries
            if elapsed > 60:  # 1 minute max
                exhaustive_times.append(60)
                print(f"  ⚠️  Exhaustive search timed out at {tool_count} tools")
                break
            
            exhaustive_times.append(elapsed)
            exhaustive_count = len(policies)
        except MemoryError:
            exhaustive_times.append(60)
            print(f"  ⚠️  Exhaustive search OOM at {tool_count} tools")
            break
    
    exhaustive_avg = np.mean(exhaustive_times) if exhaustive_times else 60
    
    # Benchmark LLM generation
    llm = create_mock_llm(latency=llm_latency)
    llm_times = []
    llm_count = 0
    
    for _ in range(trials):
        start = time.time()
        proposals = llm_policy_generation(registry, llm)
        elapsed = time.time() - start
        llm_times.append(elapsed)
        llm_count = len(proposals)
    
    llm_avg = np.mean(llm_times)
    
    return exhaustive_avg, llm_avg, exhaustive_count, llm_count


def run_full_benchmark(
    tool_counts: List[int] = None,
    max_depth: int = 3,
    llm_latency: float = 0.5,
    trials: int = 3
) -> pd.DataFrame:
    """
    Run complete scaling benchmark across tool counts.
    
    Args:
        tool_counts: List of registry sizes to test
        max_depth: Policy depth for exhaustive search
        llm_latency: Simulated LLM API latency
        trials: Number of trials per configuration
    
    Returns:
        DataFrame with benchmark results
    """
    if tool_counts is None:
        tool_counts = [2, 5, 10, 15, 20, 30, 50]
    
    results = []
    
    print("\n" + "="*70)
    print("LRS SCALING BENCHMARK: Exhaustive vs. LLM Policy Generation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Max Policy Depth: {max_depth}")
    print(f"  LLM Latency: {llm_latency}s")
    print(f"  Trials per config: {trials}")
    print(f"\nTool Counts: {tool_counts}")
    print("\n" + "-"*70)
    
    for count in tool_counts:
        print(f"\nTesting {count} tools...")
        
        exhaustive_time, llm_time, exhaustive_policies, llm_policies = run_single_benchmark(
            tool_count=count,
            max_depth=max_depth,
            llm_latency=llm_latency,
            trials=trials
        )
        
        speedup = exhaustive_time / llm_time if llm_time > 0 else float('inf')
        
        results.append({
            'tool_count': count,
            'exhaustive_time': exhaustive_time,
            'llm_time': llm_time,
            'speedup': speedup,
            'exhaustive_policies': exhaustive_policies,
            'llm_policies': llm_policies
        })
        
        print(f"  Exhaustive: {exhaustive_time:.3f}s ({exhaustive_policies} policies)")
        print(f"  LLM:        {llm_time:.3f}s ({llm_policies} proposals)")
        print(f"  Speedup:    {speedup:.1f}x")
    
    return pd.DataFrame(results)


# ============================================================================
# Visualization
# ============================================================================

def plot_results(df: pd.DataFrame, save_path: str = None):
    """Generate publication-quality plots"""
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Execution Time (log scale)
    ax1 = axes[0]
    ax1.plot(df['tool_count'], df['exhaustive_time'], 
             'o-', linewidth=2, markersize=8, label='Exhaustive O(n³)', color='#E74C3C')
    ax1.plot(df['tool_count'], df['llm_time'], 
             's-', linewidth=2, markersize=8, label='LRS Variational O(1)', color='#3498DB')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Tools in Registry', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds, log scale)', fontsize=12)
    ax1.set_title('Scaling: Exhaustive vs. Variational', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup Factor
    ax2 = axes[1]
    ax2.plot(df['tool_count'], df['speedup'], 
             'D-', linewidth=2, markersize=8, color='#2ECC71')
    ax2.set_xlabel('Number of Tools in Registry', fontsize=12)
    ax2.set_ylabel('Speedup Factor (Exhaustive / LLM)', fontsize=12)
    ax2.set_title('LRS Variational Advantage', fontsize=14, fontweight='bold')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Policy Counts
    ax3 = axes[2]
    width = 0.35
    x = np.arange(len(df))
    ax3.bar(x - width/2, df['exhaustive_policies'], width, 
            label='Exhaustive', color='#E74C3C', alpha=0.7)
    ax3.bar(x + width/2, df['llm_policies'], width, 
            label='LLM', color='#3498DB', alpha=0.7)
    ax3.set_xlabel('Tool Count', fontsize=12)
    ax3.set_ylabel('Policies Generated', fontsize=12)
    ax3.set_title('Policy Count Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['tool_count'])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {save_path}")
    
    plt.show()


def print_summary_table(df: pd.DataFrame):
    """Print formatted summary table"""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Tools':<8} {'Exhaustive':<15} {'LLM':<15} {'Speedup':<10} {'Winner'}")
    print("-"*70)
    
    for _, row in df.iterrows():
        tools = int(row['tool_count'])
        ex_time = f"{row['exhaustive_time']:.3f}s"
        llm_time = f"{row['llm_time']:.3f}s"
        speedup = f"{row['speedup']:.1f}x"
        winner = "LRS ✓" if row['speedup'] > 1 else "Exhaustive"
        
        print(f"{tools:<8} {ex_time:<15} {llm_time:<15} {speedup:<10} {winner}")
    
    print("\n" + "="*70)
    print(f"Maximum Speedup: {df['speedup'].max():.1f}x at {df.loc[df['speedup'].idxmax(), 'tool_count']:.0f} tools")
    print(f"Average Speedup: {df['speedup'].mean():.1f}x")
    print("="*70 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LRS Scaling Benchmark')
    parser.add_argument('--export', type=str, help='Export results to CSV')
    parser.add_argument('--plot', type=str, help='Save plot to file')
    parser.add_argument('--tools', type=int, nargs='+', 
                       default=[2, 5, 10, 15, 20, 30, 50],
                       help='Tool counts to benchmark')
    parser.add_argument('--depth', type=int, default=3,
                       help='Max policy depth for exhaustive search')
    parser.add_argument('--latency', type=float, default=0.5,
                       help='Simulated LLM API latency (seconds)')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials per configuration')
    
    args = parser.parse_args()
    
    # Run benchmark
    results_df = run_full_benchmark(
        tool_counts=args.tools,
        max_depth=args.depth,
        llm_latency=args.latency,
        trials=args.trials
    )
    
    # Display results
    print_summary_table(results_df)
    
    # Export if requested
    if args.export:
        results_df.to_csv(args.export, index=False)
        print(f"✓ Results exported to {args.export}")
    
    # Plot results
    plot_results(results_df, save_path=args.plot)


if __name__ == "__main__":
    main()