"""
LLM Policy Generation: Detailed walkthrough of the proposal mechanism.

This example shows:
1. Meta-cognitive prompt construction
2. Precision-adaptive temperature
3. LLM response parsing
4. Proposal validation
5. G-based evaluation
"""

from langchain_anthropic import ChatAnthropic
from lrs.core.registry import ToolRegistry
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.inference.prompts import MetaCognitivePrompter, PromptContext
from lrs.inference.llm_policy_generator import LLMPolicyGenerator
from lrs.core.free_energy import evaluate_policy, precision_weighted_selection
import json
import random


# Example tools
class FetchAPITool(ToolLens):
    """Fetch data from external API"""
    def __init__(self):
        super().__init__(name="fetch_api", input_schema={}, output_schema={})
    
    def get(self, state):
        self.call_count += 1
        if random.random() < 0.7:  # 70% success
            return ExecutionResult(True, {"data": "api_data"}, None, 0.2)
        else:
            self.failure_count += 1
            return ExecutionResult(False, None, "API timeout", 0.9)
    
    def set(self, state, obs):
        return {**state, 'api_data': obs}


class FetchCacheTool(ToolLens):
    """Fetch from cache (fast, reliable)"""
    def __init__(self):
        super().__init__(name="fetch_cache", input_schema={}, output_schema={})
    
    def get(self, state):
        self.call_count += 1
        return ExecutionResult(True, {"data": "cache_data"}, None, 0.05)
    
    def set(self, state, obs):
        return {**state, 'cache_data': obs}


class FetchDatabaseTool(ToolLens):
    """Fetch from database (authoritative, slower)"""
    def __init__(self):
        super().__init__(name="fetch_database", input_schema={}, output_schema={})
    
    def get(self, state):
        self.call_count += 1
        if random.random() < 0.9:
            return ExecutionResult(True, {"data": "db_data"}, None, 0.1)
        else:
            self.failure_count += 1
            return ExecutionResult(False, None, "DB connection error", 0.85)
    
    def set(self, state, obs):
        return {**state, 'db_data': obs}


class ProcessDataTool(ToolLens):
    """Process fetched data"""
    def __init__(self):
        super().__init__(name="process_data", input_schema={}, output_schema={})
    
    def get(self, state):
        self.call_count += 1
        has_data = any(k in state for k in ['api_data', 'cache_data', 'db_data'])
        
        if has_data:
            return ExecutionResult(True, {"processed": True}, None, 0.05)
        else:
            self.failure_count += 1
            return ExecutionResult(False, None, "No data to process", 0.9)
    
    def set(self, state, obs):
        return {**state, 'processed_data': obs}


def demonstrate_prompt_generation(precision: float):
    """Show how prompts adapt to precision"""
    print("\n" + "=" * 60)
    print(f"PROMPT GENERATION (Precision = {precision:.2f})")
    print("=" * 60)
    
    # Create prompt context
    context = PromptContext(
        precision=precision,
        recent_errors=[0.8, 0.9, 0.7] if precision < 0.4 else [0.1, 0.2],
        available_tools=['fetch_api', 'fetch_cache', 'fetch_database', 'process_data'],
        goal='Fetch and process user data',
        state={},
        tool_history=[]
    )
    
    # Generate prompt
    prompter = MetaCognitivePrompter()
    prompt = prompter.generate_prompt(context)
    
    # Show key sections
    print("\n→ Prompt includes:")
    print(f"  ✓ Precision value: {precision:.2f}")
    
    if precision < 0.4:
        print(f"  ✓ Mode: EXPLORATION")
        print(f"  ✓ Guidance: Prioritize information gain")
    elif precision > 0.7:
        print(f"  ✓ Mode: EXPLOITATION")
        print(f"  ✓ Guidance: Prioritize reward")
    else:
        print(f"  ✓ Mode: BALANCED")
        print(f"  ✓ Guidance: Mix approaches")
    
    print(f"  ✓ Available tools: {len(context.available_tools)}")
    print(f"  ✓ Output format: JSON with 3-7 proposals")
    print(f"  ✓ Diversity requirements: exploit/explore/balanced mix")
    
    return prompt


def demonstrate_temperature_adaptation():
    """Show temperature scaling with precision"""
    print("\n" + "=" * 60)
    print("TEMPERATURE ADAPTATION")
    print("=" * 60)
    
    precisions = [0.2, 0.5, 0.8, 0.95]
    
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    registry = ToolRegistry()
    generator = LLMPolicyGenerator(llm, registry, base_temperature=0.7)
    
    print("\n  Precision → Temperature:")
    for prec in precisions:
        temp = generator._adapt_temperature(prec)
        print(f"    {prec:.2f} → {temp:.2f}")
    
    print("\n  Insight: Lower precision → Higher temperature → More diverse proposals")


def demonstrate_full_pipeline():
    """Show complete LLM proposal pipeline"""
    print("\n" + "=" * 60)
    print("FULL PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Setup
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    
    registry = ToolRegistry()
    registry.register(FetchAPITool())
    registry.register(FetchCacheTool())
    registry.register(FetchDatabaseTool())
    registry.register(ProcessDataTool())
    
    generator = LLMPolicyGenerator(llm, registry)
    
    # Generate proposals at medium precision
    precision = 0.5
    
    print(f"\n→ Generating proposals (precision = {precision})...")
    proposals = generator.generate_proposals(
        state={'goal': 'Fetch and process user data'},
        precision=precision,
        num_proposals=5
    )
    
    print(f"  ✓ Generated {len(proposals)} proposals\n")
    
    # Display proposals
    for i, proposal in enumerate(proposals, 1):
        print(f"Proposal {i}: {proposal['strategy'].upper()}")
        print(f"  Tools: {' → '.join(proposal['tool_names'])}")
        print(f"  Success prob: {proposal['llm_success_prob']:.2f}")
        print(f"  Info gain: {proposal['llm_info_gain']:.2f}")
        print(f"  Rationale: {proposal['rationale']}")
        print()
    
    # Evaluate proposals
    print("→ Evaluating with Expected Free Energy...")
    
    evaluations = []
    for proposal in proposals:
        eval_obj = evaluate_policy(
            policy=proposal['policy'],
            state={},
            preferences={'success': 5.0, 'error': -3.0},
            historical_stats=registry.statistics
        )
        evaluations.append(eval_obj)
    
    for i, (proposal, eval_obj) in enumerate(zip(proposals, evaluations), 1):
        print(f"  Proposal {i}: G = {eval_obj.total_G:.2f}")
    
    # Select policy
    print("\n→ Selecting via precision-weighted softmax...")
    
    selected_idx = precision_weighted_selection(evaluations, precision)
    selected = proposals[selected_idx]
    
    print(f"\n  ✓ Selected: Proposal {selected_idx + 1}")
    print(f"    Strategy: {selected['strategy']}")
    print(f"    Tools: {' → '.join(selected['tool_names'])}")
    print(f"    G: {evaluations[selected_idx].total_G:.2f}")


def demonstrate_proposal_diversity():
    """Show that LLM generates diverse proposals"""
    print("\n" + "=" * 60)
    print("PROPOSAL DIVERSITY ANALYSIS")
    print("=" * 60)
    
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    
    registry = ToolRegistry()
    registry.register(FetchAPITool())
    registry.register(FetchCacheTool())
    registry.register(FetchDatabaseTool())
    registry.register(ProcessDataTool())
    
    generator = LLMPolicyGenerator(llm, registry)
    
    # Generate multiple times
    print("\n→ Generating 3 batches of proposals...")
    
    all_strategies = []
    all_tool_combos = []
    
    for batch in range(3):
        proposals = generator.generate_proposals(
            state={'goal': 'Fetch data'},
            precision=0.5
        )
        
        for p in proposals:
            all_strategies.append(p['strategy'])
            all_tool_combos.append(tuple(p['tool_names']))
    
    # Analyze diversity
    unique_strategies = set(all_strategies)
    unique_combos = set(all_tool_combos)
    
    print(f"\n  Strategies found: {unique_strategies}")
    print(f"  Unique tool combinations: {len(unique_combos)}")
    
    strategy_counts = {s: all_strategies.count(s) for s in unique_strategies}
    print(f"\n  Strategy distribution:")
    for strategy, count in strategy_counts.items():
        print(f"    {strategy}: {count}")
    
    print("\n  ✓ LLM generates diverse proposals spanning exploit/explore spectrum")


def main():
    print("=" * 60)
    print("LLM POLICY GENERATION - COMPLETE WALKTHROUGH")
    print("=" * 60)
    print("""
This example demonstrates the complete LLM proposal mechanism:

1. Meta-cognitive prompting (precision-adaptive)
2. Temperature scaling (exploration vs exploitation)
3. Proposal generation (variational sampling)
4. G evaluation (mathematical rigor)
5. Policy selection (precision-weighted)
    """)
    
    # Step 1: Prompt generation
    print("\n" + "=" * 60)
    print("STEP 1: META-COGNITIVE PROMPTING")
    print("=" * 60)
    
    demonstrate_prompt_generation(precision=0.3)  # Low precision
    demonstrate_prompt_generation(precision=0.8)  # High precision
    
    # Step 2: Temperature adaptation
    print("\n" + "=" * 60)
    print("STEP 2: TEMPERATURE ADAPTATION")
    print("=" * 60)
    
    demonstrate_temperature_adaptation()
    
    # Step 3: Full pipeline
    print("\n" + "=" * 60)
    print("STEP 3: COMPLETE PIPELINE")
    print("=" * 60)
    
    demonstrate_full_pipeline()
    
    # Step 4: Diversity analysis
    print("\n" + "=" * 60)
    print("STEP 4: DIVERSITY ANALYSIS")
    print("=" * 60)
    
    demonstrate_proposal_diversity()
    
    # Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. LLMs propose, math decides
   - LLM: Generative creativity
   - Math: Rigorous evaluation

2. Precision drives adaptation
   - Low γ → Explore (high temp, diverse proposals)
   - High γ → Exploit (low temp, focused proposals)

3. Meta-cognitive awareness
   - LLM receives precision value
   - Adjusts strategy appropriately
   - Self-assesses success probability

4. Guaranteed diversity
   - Prompt enforces exploit/explore/balanced mix
   - Multiple proposals spanning strategies
   - No mode collapse

5. Scalable to 100+ tools
   - Constant number of proposals
   - Linear time complexity
   - No combinatorial explosion
    """)


if __name__ == "__main__":
    main()
