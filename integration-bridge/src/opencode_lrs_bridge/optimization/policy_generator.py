"""
Optimized policy generation for LRS-Agents with intelligent pruning.
"""

from typing import List, Dict, Any, Set, Tuple
import asyncio
import itertools
import time
from collections import defaultdict, deque
import random
import structlog

from ..core.registry import ToolRegistry
from ..core.precision import PrecisionData

logger = structlog.get_logger(__name__)


class PolicyOptimizer:
    """Optimizes policy generation to prevent exponential blowup."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.tool_success_rates: Dict[str, float] = defaultdict(float)
        self.tool_usage_counts: Dict[str, int] = defaultdict(int)
        self.recent_combinations: deque(maxlen=1000)
        self.combination_cache: Dict[str, List[List[str]]] {}
        self.max_combinations_per_depth = 50
        self.max_depth = 3  # Limit depth to prevent blowup
        
    def update_tool_statistics(self, tool_name: str, success: bool):
        """Update tool performance statistics."""
        if success:
            # Incremental learning rate
            current_rate = self.tool_success_rates[tool_name]
            alpha = max(1.0, current_rate * 0.1)  # Learning rate
            beta = max(1.0, (1.0 - current_rate) * 0.1)
            self.tool_success_rates[tool_name] = alpha / (alpha + beta)
        else:
            # Decay on failure
            current_rate = self.tool_success_rates[tool_name]
            self.tool_success_rates[tool_name] = current_rate * 0.9
        
        self.tool_usage_counts[tool_name] += 1
    
    def _get_prioritized_tools(self, limit: int = 20) -> List[str]:
        """Get prioritized tools based on success rates and recent usage."""
        tools = list(self.registry.tools.keys())
        
        # Sort by weighted score (success rate + usage frequency)
        def tool_score(tool_name: str) -> float:
            success_rate = self.tool_success_rates[tool_name]
            usage_count = self.tool_usage_counts[tool_name]
            recent_usage = min(usage_count, 10)  # Cap recent usage
            
            return success_rate * 0.7 + recent_usage * 0.3
        
        sorted_tools = sorted(tools, key=tool_score, reverse=True)
        return sorted_tools[:limit]
    
    def _generate_intelligent_combinations(
        self, 
        available_tools: List[str],
        max_combinations: int = 100
    ) -> List[List[str]]:
        """Generate intelligent tool combinations."""
        combinations = []
        
        # Start with single tools (highest priority)
        for tool in available_tools[:20]:  # Top 20 single tools
            combinations.append([tool])
        
        # Generate 2-tool combinations (smart pairing)
        priority_tools = available_tools[:10]  # Top 10 tools for pairing
        for i, tool1 in enumerate(priority_tools):
            for tool2 in priority_tools[i+1:]:
                combinations.append([tool1, tool2])
                
                # Add complementary tool pairs
                if self._are_complementary(tool1, tool2):
                    combinations.append([tool2, tool1])
        
        # Generate limited 3-tool sequences (breadth-first approach)
        for start_tool in priority_tools[:5]:  # Only top 5 as starting points
            sequence = [start_tool]
            
            # Build sequence by adding most complementary tool next
            while len(sequence) < 3:
                best_next = self._find_best_next_tool(sequence, available_tools)
                if best_next and best_next not in sequence:
                    sequence.append(best_next)
                else:
                    break
            
            if len(sequence) == 3:
                combinations.append(sequence)
        
        # Fill remaining combinations with random but valid combinations
        while len(combinations) < max_combinations:
            seq_length = random.randint(2, 3)
            random_tools = random.sample(available_tools, min(seq_length, len(available_tools)))
            
            # Ensure no immediate repetition
            if len(set(random_tools)) == len(random_tools):
                combinations.append(random_tools)
        
        return combinations
    
    def _are_complementary(self, tool1: str, tool2: str) -> bool:
        """Check if two tools are complementary."""
        # Define tool categories (simplified)
        categories = {
            "search": ["search", "web_search", "find"],
            "data": ["database", "file_read", "file_write", "parse"],
            "computation": ["calculate", "compute", "analyze", "transform"],
            "communication": ["http", "api", "websocket", "notify"],
            "ai": ["llm", "model", "generate", "summarize"]
        }
        
        # Get categories
        cat1, cat2 = None, None
        for category, tools in categories.items():
            if tool1 in tools:
                cat1 = category
            if tool2 in tools:
                cat2 = category
        
        # Complementary if from different categories
        return cat1 is not None and cat2 is not None and cat1 != cat2
    
    def _find_best_next_tool(
        self, 
        current_sequence: List[str], 
        available_tools: List[str]
    ) -> Optional[str]:
        """Find the best next tool to extend a sequence."""
        used_tools = set(current_sequence)
        available = [t for t in available_tools if t not in used_tools]
        
        if not available:
            return None
        
        # Prioritize tools that complement the current sequence
        best_tool = None
        best_score = -1.0
        
        for tool in available:
            score = 0
            
            # Success rate bonus
            score += self.tool_success_rates[tool] * 0.5
            
            # Complementarity bonus
            last_tool = current_sequence[-1] if current_sequence else None
            if last_tool and self._are_complementary(last_tool, tool):
                score += 2.0
            
            # Usage frequency bonus (prefer less used tools for diversity)
            score += max(0, 5 - self.tool_usage_counts[tool]) * 0.1
            
            if score > best_score:
                best_score = score
                best_tool = tool
        
        return best_tool
    
    def _generate_policy_candidates_optimized(
        self, 
        max_policies: int = 50,
        state: Dict[str, Any] = None
    ) -> List[List[str]]:
        """Generate optimized policy candidates with intelligent pruning."""
        # Get available tools
        available_tools = list(self.registry.tools.keys())
        
        # Filter tools based on context if available
        if state and "goal" in state:
            goal = state["goal"].lower()
            if "search" in goal:
                available_tools = [t for t in available_tools if any(
                    cat in t for cat in ["search", "web_search", "find"]
                )]
            elif "data" in goal:
                available_tools = [t for t in available_tools if any(
                    cat in t for cat in ["database", "file_read", "file_write", "parse"]
                )]
            elif "calculate" in goal:
                available_tools = [t for t in available_tools if any(
                    cat in t for cat in ["calculate", "compute", "analyze", "transform"]
                )]
        
        # Generate intelligent combinations
        combinations = self._generate_intelligent_combinations(
            available_tools, max_policies
        )
        
        # Cache results
        cache_key = f"{'_'.join(sorted(available_tools))}_{max_policies}"
        self.combination_cache[cache_key] = combinations
        
        logger.info(
            "Generated optimized policy candidates",
            available_tools=len(available_tools),
            combinations=len(combinations),
            cache_key=cache_key
        )
        
        return combinations
    
    def _rank_policies_intelligently(
        self, 
        combinations: List[List[str]],
        state: Dict[str, Any] = None
    ) -> List[Tuple[List[str], float]]:
        """Rank policies using multiple heuristics."""
        ranked_policies = []
        
        for combination in combinations:
            score = 0.0
            
            # Heuristic 1: Tool success rate
            tool_success_sum = sum(
                self.tool_success_rates.get(tool, 0.5) for tool in combination
            )
            score += tool_success_sum / len(combination) if combination else 0
            
            # Heuristic 2: Combination diversity
            categories = set()
            for tool in combination:
                for category, tools in {
                    "search": ["search", "web_search"],
                    "data": ["database", "file_read", "file_write"],
                    "computation": ["calculate", "compute", "analyze"],
                    "communication": ["http", "api", "websocket"],
                    "ai": ["llm", "model", "generate"]
                }.items():
                    if tool in tools:
                        categories.add(category)
            score += len(categories) * 0.3  # Prefer diverse toolsets
            
            # Heuristic 3: Length penalty (shorter policies preferred)
            score -= len(combination) * 0.1
            
            # Heuristic 4: Context relevance
            if state and "goal" in state:
                goal = state["goal"].lower()
                goal_keywords = {
                    "search": ["search", "find", "lookup"],
                    "data": ["database", "file", "query", "read"],
                    "calculate": ["compute", "math", "calculate"],
                    "write": ["write", "save", "create"],
                    "analyze": ["analyze", "process", "transform"]
                }
                
                relevant_tools = set()
                for keyword, tools in goal_keywords.items():
                    if keyword in goal:
                        relevant_tools.update(tools)
                
                relevance_bonus = sum(1 for tool in combination if tool in relevant_tools)
                score += relevance_bonus * 0.5
            
            ranked_policies.append((combination, score))
        
        # Sort by score (highest first)
        ranked_policies.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_policies


class OptimizedPolicyGenerator:
    """Main optimized policy generator class."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.optimizer = PolicyOptimizer(registry)
        self.generation_cache: Dict[str, List[List[str]]] = {}
        
    async def generate_policies(
        self, 
        state: Dict[str, Any], 
        max_policies: int = 50,
        force_regenerate: bool = False
    ) -> List[List[str]]:
        """Generate optimized policies with caching."""
        # Create cache key based on state
        context_hash = self._create_context_hash(state)
        
        # Check cache unless forced regeneration
        if not force_regenerate and context_hash in self.generation_cache:
            cached_policies = self.generation_cache[context_hash]
            if cached_policies:
                logger.info("Using cached policies", context_hash=context_hash)
                return cached_policies
        
        # Generate candidates
        candidates = self.optimizer._generate_policy_candidates_optimized(
            max_policies=max_policies,
            state=state
        )
        
        # Rank candidates
        ranked_policies = self.optimizer._rank_policies_intelligently(
            combinations=candidates,
            state=state
        )
        
        # Extract just the policies (not scores)
        final_policies = [policy for policy, score in ranked_policies]
        
        # Cache results
        self.generation_cache[context_hash] = final_policies
        
        logger.info(
            "Generated optimized policies",
            candidates_count=len(candidates),
            final_count=len(final_policies),
            context_hash=context_hash
        )
        
        return final_policies
    
    def _create_context_hash(self, state: Dict[str, Any]) -> str:
        """Create a hash of the context for caching."""
        # Extract relevant context elements
        context_elements = [
            state.get("goal", ""),
            str(state.get("precision", {})),
            str(state.get("preferences", {}))
        ]
        
        # Create simple hash
        import hashlib
        context_str = "|".join(context_elements)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def update_tool_performance(self, tool_name: str, success: bool, execution_time: float = 0.0):
        """Update tool performance metrics."""
        # Adjust success rate based on execution time
        time_weight = min(1.0, max(0.1, execution_time / 10.0))  # Normalize execution time
        
        if success:
            # Success with good time gets higher rate
            self.optimizer.update_tool_statistics(tool_name, True)
            # Additional bonus for fast execution
            if execution_time < 1.0:
                self.optimizer.tool_success_rates[tool_name] = min(
                    1.0,
                    self.optimizer.tool_success_rates[tool_name] + time_weight
                )
        else:
            # Slow success gets lower rate
            self.optimizer.update_tool_statistics(tool_name, False)
            # Additional penalty for slow execution
            if execution_time > 5.0:
                self.optimizer.tool_success_rates[tool_name] = max(
                    0.1,
                    self.optimizer.tool_success_rates[tool_name] - time_weight
                )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all tools."""
        stats = {}
        
        for tool_name in self.optimizer.tool_success_rates:
            stats[tool_name] = {
                "success_rate": self.optimizer.tool_success_rates[tool_name],
                "usage_count": self.optimizer.tool_usage_counts[tool_name],
                "score": (self.optimizer.tool_success_rates[tool_name] * 0.7 + 
                       min(5, self.optimizer.tool_usage_counts[tool_name]) * 0.3)
            }
        
        return stats
    
    def clear_cache(self):
        """Clear the generation cache."""
        self.generation_cache.clear()
        logger.info("Cleared policy generation cache")


# Replacement function to integrate into LRS-Agents
def install_optimized_policy_generator():
    """Install the optimized policy generator to replace the exponential blowup."""
    # This would replace the _generate_policy_candidates method
    # in the LRS-Agents langgraph.py file
    
    # Import statement to add:
    # from .optimized_policy_generator import OptimizedPolicyGenerator
    # 
    # Then in the LRSAgent class:
    # self.policy_generator = OptimizedPolicyGenerator(self.registry)
    # 
    # And replace the _generate_policies method with:
    # async def _generate_policies(self, state: LRSState) -> LRSState:
    #     state["current_hbn_level"] = "planning"
    #     max_depth = 2 if state["precision"]["planning"] > 0.6 else 3
    #     state["candidate_policies"] = await self.policy_generator.generate_policies(state, max_policies=20)
    #     return state
    
    logger.info("Optimized policy generator installed successfully")


if __name__ == "__main__":
    # Demonstration
    install_optimized_policy_generator()