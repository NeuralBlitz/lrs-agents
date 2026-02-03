"""
LangGraph integration for Lambda-Reflexive Synthesis agents.

Provides drop-in replacement for standard ReAct agents with active inference dynamics.

Usage:
    from lrs.integration.langgraph import create_lrs_agent
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    tools = [...]  # Your tools

    agent = create_lrs_agent(llm, tools)
    result = agent.invoke({"messages": [{"role": "user", "content": "Task"}]})
"""

from typing import Dict, List, Annotated, Literal, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from datetime import datetime
import operator

from lrs.core.precision import HierarchicalPrecision, PrecisionParameters
from lrs.core.free_energy import (
    calculate_expected_free_energy,
    precision_weighted_selection,
    PolicyEvaluation,
)
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.core.registry import ToolRegistry


# ============================================================================
# State Schema
# ============================================================================


class LRSState(TypedDict, total=False):
    """
    Complete state for LRS agent.

    TypedDict with total=False allows optional fields for incremental updates.
    """

    # Standard LangGraph fields
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # Precision tracking
    precision: Dict[str, float]
    precision_history: List[Dict[str, float]]

    # Policy management
    candidate_policies: List[List[ToolLens]]
    policy_evaluations: List[PolicyEvaluation]
    selected_policy: List[ToolLens]
    current_policy_index: int

    # Execution tracking
    tool_history: Annotated[List[Dict[str, Any]], operator.add]

    # Hierarchical state
    current_hbn_level: Literal["abstract", "planning", "execution"]
    belief_state: Dict[str, Any]

    # Adaptation tracking
    adaptation_count: int
    adaptation_events: List[Dict[str, Any]]

    # Goal and preferences
    goal: str
    preferences: Dict[str, float]


# ============================================================================
# Graph Builder
# ============================================================================


class LRSGraphBuilder:
    """
    Constructs a LangGraph with active inference dynamics.

    Architecture:
        propose_policies → evaluate_G → select_policy → execute_tool →
        update_precision → [decision gate] → {continue | replan | end}

    Attributes:
        llm: Language model for policy proposal generation
        registry: ToolRegistry with available tools
        precision_manager: HierarchicalPrecision tracker
        preferences: Goal preferences for pragmatic value calculation
    """

    def __init__(
        self,
        llm,
        registry: ToolRegistry,
        preferences: Optional[Dict[str, float]] = None,
        precision_config: Optional[Dict[str, PrecisionParameters]] = None,
        use_llm_proposals: bool = True,
    ):
        """
        Initialize graph builder.

        Args:
            llm: Language model (must have .invoke() or .generate() method)
            registry: Tool registry with available tools
            preferences: Goal preferences for G calculation.
                Example: {'data_retrieved': 3.0, 'error': -5.0}
            precision_config: Optional custom precision parameters per level
            use_llm_proposals: Whether to use LLM for policy generation (for backward compatibility)
        """
        self.llm = llm
        self.registry = registry
        self.use_llm_proposals = use_llm_proposals
        self.preferences = preferences or {"success": 2.0, "error": -5.0, "execution_time": -0.1}

        # Initialize hierarchical precision
        if precision_config:
            self.hp = HierarchicalPrecision(levels=precision_config)
        else:
            self.hp = HierarchicalPrecision()

    def build(self) -> StateGraph:
        """
        Construct the complete LRS graph.

        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(LRSState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_state)
        workflow.add_node("generate_policies", self._generate_policies)
        workflow.add_node("evaluate_G", self._evaluate_free_energy)
        workflow.add_node("select_policy", self._select_policy)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("update_precision", self._update_precision)
        workflow.add_node("check_goal", self._check_goal_satisfaction)

        # Define flow
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "generate_policies")
        workflow.add_edge("generate_policies", "evaluate_G")
        workflow.add_edge("evaluate_G", "select_policy")
        workflow.add_edge("select_policy", "execute_tool")
        workflow.add_edge("execute_tool", "update_precision")
        workflow.add_edge("update_precision", "check_goal")

        # Conditional branching based on precision and goal state
        workflow.add_conditional_edges(
            "check_goal",
            self._decision_gate,
            {
                "success": END,
                "continue": "execute_tool",
                "replan": "generate_policies",
                "fail": END,
            },
        )

        return workflow.compile()

    # ========================================================================
    # Node Implementations
    # ========================================================================

    def _initialize_state(self, state: LRSState) -> LRSState:
        """
        Initialize agent state from user message.

        Extracts goal, sets initial precision, prepares belief state.
        """
        # Extract goal from messages
        if state.get("messages"):
            latest_message = state["messages"][-1]
            goal = latest_message.get("content", "No goal specified")
        else:
            goal = "No goal specified"

        # Initialize state
        state["goal"] = goal
        state["precision"] = self.hp.get_all()
        state["precision_history"] = [self.hp.get_all()]
        state["current_hbn_level"] = "abstract"
        state["adaptation_count"] = 0
        state["adaptation_events"] = []
        state["tool_history"] = []
        state["current_policy_index"] = 0
        state["belief_state"] = {"goal": goal, "goal_satisfied": False}
        state["preferences"] = self.preferences

        return state

    def _generate_policies(self, state: LRSState) -> LRSState:
        """
        Generate optimized candidate policies using intelligent generation.
        """
        state["current_hbn_level"] = "planning"
        max_depth = 2 if state["precision"]["planning"] > 0.6 else 3
        candidates = self._generate_intelligent_candidates(max_depth=max_depth, state=state)

        state["candidate_policies"] = candidates

        return state

    def _generate_policy_candidates(self, max_depth: int) -> List[List[ToolLens]]:
        """
        Generate all valid tool sequences up to max_depth.

        TODO: Replace with LLM-guided generation for production.
        """
        policies = []

        def build_tree(current: List[ToolLens], depth: int):
            if depth == 0:
                if current:
                    policies.append(current)
                return

            for tool in self.registry.tools.values():
                # Avoid immediate repetition
                if not current or tool != current[-1]:
                    build_tree(current + [tool], depth - 1)

        build_tree([], max_depth)
        return policies

    def _evaluate_free_energy(self, state: LRSState) -> LRSState:
        """
        Calculate Expected Free Energy for all candidate policies.

        Core active inference calculation: G = Epistemic - Pragmatic
        """
        evaluations = []

        for policy in state["candidate_policies"]:
            eval_result = calculate_expected_free_energy(
                policy=policy, registry=self.registry, preferences=state["preferences"]
            )
            evaluations.append(
                PolicyEvaluation(
                    epistemic_value=0.0,
                    pragmatic_value=0.0,
                    total_G=eval_result,
                    expected_success_prob=0.5,
                    components={"policy": [t.name for t in policy]},
                )
            )

        state["policy_evaluations"] = evaluations

        return state

    def _select_policy(self, state: LRSState) -> LRSState:
        """
        Select policy via precision-weighted softmax over G values.

        High precision → exploit (choose lowest G)
        Low precision → explore (flatten distribution)
        """
        selected_idx = precision_weighted_selection(
            evaluations=state["policy_evaluations"], precision=state["precision"]["planning"]
        )

        selected_policy = state["candidate_policies"][selected_idx]
        state["selected_policy"] = selected_policy
        state["current_policy_index"] = 0  # Reset for execution

        return state

    def _execute_tool(self, state: LRSState) -> LRSState:
        """
        Execute next tool in selected policy.

        Updates belief state and records prediction error.
        """
        state["current_hbn_level"] = "execution"

        if not state.get("selected_policy"):
            return state

        policy = state["selected_policy"]
        idx = state["current_policy_index"]

        if idx >= len(policy):
            # Policy exhausted
            return state

        # Execute tool
        tool = policy[idx]
        result = tool.get(state["belief_state"])

        # Record execution
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool.name,
            "success": result.success,
            "prediction_error": result.prediction_error,
            "error_message": result.error,
        }

        state["tool_history"].append(execution_record)

        # Update belief state
        if result.success:
            state["belief_state"] = tool.set(state["belief_state"], result.value)

        # Advance policy index
        state["current_policy_index"] = idx + 1

        return state

    def _update_precision(self, state: LRSState) -> LRSState:
        """
        Update hierarchical precision based on prediction error.

        Implements Bayesian belief revision via Beta distribution updates.
        """
        if not state["tool_history"]:
            return state

        latest_execution = state["tool_history"][-1]
        prediction_error = latest_execution["prediction_error"]

        # Update hierarchical precision
        updated_precisions = self.hp.update(level="execution", prediction_error=prediction_error)

        # Sync to state
        state["precision"].update(updated_precisions)
        state["precision_history"].append(self.hp.get_all())

        # Record adaptation events
        if prediction_error > 0.7:
            state["adaptation_count"] += 1
            state["adaptation_events"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "tool": latest_execution["tool"],
                    "error": prediction_error,
                    "precision_before": state["precision_history"][-2]["planning"]
                    if len(state["precision_history"]) > 1
                    else None,
                    "precision_after": state["precision"]["planning"],
                }
            )

        return state

    def _check_goal_satisfaction(self, state: LRSState) -> LRSState:
        """
        Check if goal has been satisfied.

        In production, this would use more sophisticated goal checking.
        """
        # Simple heuristic: goal satisfied if no errors in last 2 executions
        if len(state["tool_history"]) >= 2:
            recent_success = all(exec["success"] for exec in state["tool_history"][-2:])
            state["belief_state"]["goal_satisfied"] = recent_success

        return state

    # ========================================================================
    # Decision Gate
    # ========================================================================

    def _decision_gate(self, state: LRSState) -> str:
        """
        Determine next action based on goal satisfaction and precision.

        Returns:
            "success": Goal achieved, end execution
            "continue": Continue current policy
            "replan": Precision dropped, generate new policies
            "fail": Systemic failure, cannot proceed
        """
        # Check for goal satisfaction
        if state["belief_state"].get("goal_satisfied", False):
            return "success"

        # Check for systemic failure (all levels have very low precision)
        if all(p < 0.2 for p in state["precision"].values()):
            return "fail"

        # Check if current policy is exhausted
        if state["current_policy_index"] >= len(state.get("selected_policy", [])):
            # Policy done but goal not satisfied → replan
            return "replan"

        # Check if precision collapsed (adaptation needed)
        if state["precision"]["planning"] < 0.4:
            return "replan"

        # Continue with current policy
        return "continue"


# ============================================================================
# Factory Function (Public API)
# ============================================================================


def create_lrs_agent(
    llm, tools: List[ToolLens], preferences: Optional[Dict[str, float]] = None, **kwargs
) -> StateGraph:
    """
    Create an LRS-powered agent as drop-in replacement for create_react_agent.

    Args:
        llm: Language model (Anthropic, OpenAI, etc.)
        tools: List of ToolLens objects or LangChain tools
        preferences: Goal preferences for pragmatic value calculation
        **kwargs: Additional configuration (precision_threshold, etc.)

    Returns:
        Compiled StateGraph with active inference dynamics

    Examples:
        >>> from lrs import create_lrs_agent
        >>> from langchain_anthropic import ChatAnthropic
        >>>
        >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        >>> tools = [ShellTool(), PythonREPLTool()]
        >>>
        >>> agent = create_lrs_agent(llm, tools, preferences={'success': 5.0})
        >>>
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "List files in /tmp"}]
        ... })
    """
    # Create tool registry
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)

    # Build graph
    builder = LRSGraphBuilder(llm=llm, registry=registry, preferences=preferences)

    return builder.build()


# ============================================================================
# Monitoring Integration
# ============================================================================


def create_monitored_lrs_agent(
    llm, tools: List[ToolLens], tracker: "LRSStateTracker", **kwargs
) -> StateGraph:
    """
    Create LRS agent with integrated monitoring.

    Automatically streams state updates to dashboard tracker.

    Args:
        llm: Language model
        tools: Tool lenses
        tracker: LRSStateTracker instance for monitoring
        **kwargs: Additional configuration

    Returns:
        Compiled StateGraph with monitoring hooks
    """
    agent = create_lrs_agent(llm, tools, **kwargs)

    # Wrap invoke to capture state
    original_invoke = agent.invoke

    def monitored_invoke(input_state, **invoke_kwargs):
        result = original_invoke(input_state, **invoke_kwargs)

        # Update tracker with final state
        tracker.update(result)

        return result

    agent.invoke = monitored_invoke

    return agent
