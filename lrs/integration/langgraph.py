"""
LangGraph integration for LRS-Agents.

Provides the main agent builder that creates a LangGraph execution graph
with Active Inference dynamics (precision tracking, G calculation, adaptation).
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

from lrs.core.precision import HierarchicalPrecision
from lrs.core.free_energy import (
    calculate_expected_free_energy,
    evaluate_policy,
    precision_weighted_selection
)
from lrs.core.registry import ToolRegistry
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.inference.llm_policy_generator import LLMPolicyGenerator
from lrs.monitoring.tracker import LRSStateTracker


class LRSState(TypedDict, total=False):
    """
    Complete state schema for LRS agents.
    
    This is the state that flows through the LangGraph execution graph.
    
    Attributes:
        messages: Conversation messages
        belief_state: Agent's current beliefs about the world
        precision: Precision values at each hierarchical level
        prediction_errors: Recent prediction errors
        current_policy: Currently executing policy
        candidate_policies: Policies being considered
        G_values: Expected Free Energy for each candidate
        tool_history: History of tool executions
        adaptation_count: Number of adaptations triggered
        current_hbn_level: Current hierarchical level (abstract/planning/execution)
        next: Next node to execute in graph
    """
    # Core state
    messages: Annotated[List[Dict[str, str]], operator.add]
    belief_state: Dict[str, Any]
    
    # Precision tracking
    precision: Dict[str, float]
    prediction_errors: Dict[str, List[float]]
    
    # Policy state
    current_policy: List[ToolLens]
    candidate_policies: List[Dict[str, Any]]
    G_values: Dict[int, float]
    
    # History
    tool_history: Annotated[List[Dict[str, Any]], operator.add]
    adaptation_count: int
    
    # Hierarchical level
    current_hbn_level: str
    
    # Graph routing
    next: str


class LRSGraphBuilder:
    """
    Builder for LangGraph-based LRS agents.
    
    Creates a StateGraph with nodes for:
    1. Initialize - Set up initial state
    2. Generate policies - Create candidate policies
    3. Evaluate G - Calculate Expected Free Energy
    4. Select policy - Precision-weighted selection
    5. Execute tool - Run selected policy
    6. Update precision - Bayesian belief update
    
    Conditional edges based on precision gates:
    - γ > 0.7 → Execute (confident)
    - 0.4 < γ < 0.7 → Replan (uncertain)
    - γ < 0.4 → Escalate (confused)
    
    Examples:
        >>> from langchain_anthropic import ChatAnthropic
        >>> 
        >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        >>> registry = ToolRegistry()
        >>> # ... register tools ...
        >>> 
        >>> builder = LRSGraphBuilder(llm, registry)
        >>> agent = builder.build()
        >>> 
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "Fetch data"}]
        ... })
    """
    
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        preferences: Optional[Dict[str, float]] = None,
        use_llm_proposals: bool = True,
        tracker: Optional[LRSStateTracker] = None
    ):
        """
        Initialize LRS graph builder.
        
        Args:
            llm: Language model for policy generation
            registry: Tool registry
            preferences: Reward function (default: {'success': 5.0, 'error': -3.0})
            use_llm_proposals: Use LLM for proposals (vs exhaustive search)
            tracker: Optional state tracker for monitoring
        """
        self.llm = llm
        self.registry = registry
        self.preferences = preferences or {
            'success': 5.0,
            'error': -3.0,
            'step_cost': -0.1
        }
        self.use_llm_proposals = use_llm_proposals
        self.tracker = tracker
        
        # Initialize components
        self.hp = HierarchicalPrecision()
        
        if use_llm_proposals:
            self.llm_generator = LLMPolicyGenerator(llm, registry)
    
    def build(self) -> StateGraph:
        """
        Build and compile the LRS agent graph.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Create graph
        workflow = StateGraph(LRSState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize)
        workflow.add_node("generate_policies", self._generate_policies)
        workflow.add_node("evaluate_G", self._evaluate_G)
        workflow.add_node("select_policy", self._select_policy)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("update_precision", self._update_precision)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Add edges
        workflow.add_edge("initialize", "generate_policies")
        workflow.add_edge("generate_policies", "evaluate_G")
        workflow.add_edge("evaluate_G", "select_policy")
        workflow.add_edge("select_policy", "execute_tool")
        workflow.add_edge("execute_tool", "update_precision")
        
        # Add conditional edge from update_precision (precision gate)
        workflow.add_conditional_edges(
            "update_precision",
            self._precision_gate,
            {
                "continue": "generate_policies",  # Continue execution
                "end": END                         # Task complete
            }
        )
        
        # Compile
        return workflow.compile()
    
    # Node implementations
    
    def _initialize(self, state: LRSState) -> LRSState:
        """
        Initialize agent state.
        
        Sets up precision, belief state, and history tracking.
        """
        # Initialize precision if not present
        if not state.get('precision'):
            state['precision'] = self.hp.get_all()
        
        # Initialize belief state
        if not state.get('belief_state'):
            state['belief_state'] = {}
        
        # Initialize history
        if not state.get('tool_history'):
            state['tool_history'] = []
        
        if not state.get('adaptation_count'):
            state['adaptation_count'] = 0
        
        # Set hierarchical level
        state['current_hbn_level'] = 'planning'
        
        return state
    
    def _generate_policies(self, state: LRSState) -> LRSState:
        """
        Generate candidate policies.
        
        Uses LLM proposals (if enabled) or exhaustive search.
        """
        if self.use_llm_proposals:
            # LLM-based generation
            proposals = self.llm_generator.generate_proposals(
                state=state,
                precision=state['precision'].get('planning', 0.5)
            )
        else:
            # Exhaustive search (for small tool sets)
            proposals = self._generate_policy_candidates(max_depth=3)
        
        state['candidate_policies'] = proposals
        return state
    
    def _generate_policy_candidates(
        self,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate policies via exhaustive search.
        
        Only practical for small tool sets (<10 tools).
        
        Args:
            max_depth: Maximum policy length
        
        Returns:
            List of policy candidates
        """
        candidates = []
        tools = list(self.registry.tools.values())
        
        def build_policies(current_policy, depth):
            if depth == 0:
                if current_policy:
                    candidates.append({
                        'policy': current_policy,
                        'strategy': 'unknown'
                    })
                return
            
            # Add single-tool policy
            if current_policy:
                candidates.append({
                    'policy': current_policy,
                    'strategy': 'unknown'
                })
            
            # Extend with each tool
            for tool in tools:
                build_policies(current_policy + [tool], depth - 1)
        
        # Generate all policies up to max_depth
        build_policies([], max_depth)
        
        # Limit to reasonable number
        return candidates[:20]
    
    def _evaluate_G(self, state: LRSState) -> LRSState:
        """
        Calculate Expected Free Energy for all candidate policies.
        """
        G_values = {}
        
        for i, proposal in enumerate(state['candidate_policies']):
            policy = proposal['policy']
            
            # Calculate G
            G = calculate_expected_free_energy(
                policy=policy,
                state=state,
                preferences=self.preferences,
                historical_stats=self.registry.statistics
            )
            
            G_values[i] = G
        
        state['G_values'] = G_values
        return state
    
    def _select_policy(self, state: LRSState) -> LRSState:
        """
        Select policy via precision-weighted softmax.
        """
        if not state['candidate_policies']:
            state['current_policy'] = []
            return state
        
        # Evaluate all policies
        evaluations = []
        for i, proposal in enumerate(state['candidate_policies']):
            policy = proposal['policy']
            G = state['G_values'][i]
            
            eval_obj = evaluate_policy(
                policy=policy,
                state=state,
                preferences=self.preferences,
                historical_stats=self.registry.statistics
            )
            eval_obj.total_G = G  # Override with calculated G
            evaluations.append(eval_obj)
        
        # Precision-weighted selection
        precision = state['precision'].get('planning', 0.5)
        selected_idx = precision_weighted_selection(evaluations, precision)
        
        # Set current policy
        state['current_policy'] = state['candidate_policies'][selected_idx]['policy']
        
        return state
    
    def _execute_tool(self, state: LRSState) -> LRSState:
        """
        Execute the selected policy.
        
        Runs each tool in sequence, tracking results.
        """
        if not state.get('current_policy'):
            return state
        
        for tool in state['current_policy']:
            # Execute tool
            result = tool.get(state['belief_state'])
            
            # Update belief state
            if result.success:
                state['belief_state'] = tool.set(state['belief_state'], result.value)
            
            # Track execution
            execution_entry = {
                'tool': tool.name,
                'success': result.success,
                'prediction_error': result.prediction_error,
                'error': result.error,
                'result': result.value
            }
            
            if 'tool_history' not in state:
                state['tool_history'] = []
            state['tool_history'].append(execution_entry)
            
            # Update registry statistics
            self.registry.update_statistics(
                tool_name=tool.name,
                success=result.success,
                prediction_error=result.prediction_error
            )
            
            # Track with monitor
            if self.tracker:
                self.tracker.track_state(state)
            
            # Stop on failure
            if not result.success:
                break
        
        return state
    
    def _update_precision(self, state: LRSState) -> LRSState:
        """
        Update precision based on prediction errors.
        
        Implements Bayesian belief update via Beta distribution.
        """
        if not state.get('tool_history'):
            return state
        
        # Get latest execution
        latest = state['tool_history'][-1]
        prediction_error = latest['prediction_error']
        
        # Update hierarchical precision
        updated = self.hp.update('execution', prediction_error)
        
        # Store in state
        state['precision'] = self.hp.get_all()
        
        # Check for adaptation
        if state['precision']['execution'] < 0.4:
            state['adaptation_count'] = state.get('adaptation_count', 0) + 1
        
        return state
    
    def _precision_gate(self, state: LRSState) -> str:
        """
        Conditional routing based on precision.
        
        Decides whether to continue execution or end.
        
        Returns:
            "continue" or "end"
        """
        # Check if task is complete
        belief_state = state.get('belief_state', {})
        
        # Simple completion check (can be customized)
        if belief_state.get('completed', False):
            return "end"
        
        # Check tool history
        tool_history = state.get('tool_history', [])
        
        # End if max iterations reached
        max_iterations = state.get('max_iterations', 50)
        if len(tool_history) >= max_iterations:
            return "end"
        
        # End if recent success
        if tool_history and tool_history[-1]['success']:
            # Check if goal appears met
            if 'goal_met' in belief_state or 'data' in belief_state:
                return "end"
        
        # Continue by default
        return "continue"


def create_lrs_agent(
    llm: Any,
    tools: List[ToolLens],
    preferences: Optional[Dict[str, float]] = None,
    use_llm_proposals: bool = True,
    tracker: Optional[LRSStateTracker] = None
) -> StateGraph:
    """
    Create an LRS agent (convenience function).
    
    Args:
        llm: Language model
        tools: List of ToolLens objects
        preferences: Reward function
        use_llm_proposals: Use LLM for policy generation
        tracker: Optional state tracker
    
    Returns:
        Compiled LangGraph agent
    
    Examples:
        >>> from lrs import create_lrs_agent
        >>> from langchain_anthropic import ChatAnthropic
        >>> 
        >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        >>> tools = [MyTool(), AnotherTool()]
        >>> 
        >>> agent = create_lrs_agent(llm, tools)
        >>> 
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "Solve task"}]
        ... })
    """
    # Create registry
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    
    # Build agent
    builder = LRSGraphBuilder(
        llm=llm,
        registry=registry,
        preferences=preferences,
        use_llm_proposals=use_llm_proposals,
        tracker=tracker
    )
    
    return builder.build()
