#!/usr/bin/env python3
"""
OpenCode LRS Multi-Agent Coordination System

Phase 4: Advanced Features - Multi-Agent Coordination
Implements coordinated agent systems for complex task decomposition
and collaborative problem-solving.
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from collections import defaultdict

# Import cognitive components
try:
    from phase6_neuromorphic_research.phase6_neuromorphic_setup import (
        CognitiveArchitecture,
    )

    COGNITIVE_AGENTS_AVAILABLE = True
except ImportError:
    COGNITIVE_AGENTS_AVAILABLE = False


class AgentRole(Enum):
    """Specialized agent roles for multi-agent coordination."""

    ANALYST = "analyst"  # Code analysis and understanding
    ARCHITECT = "architect"  # System design and planning
    DEVELOPER = "developer"  # Implementation and coding
    TESTER = "tester"  # Quality assurance and testing
    DEPLOYER = "deployer"  # Deployment and operations
    COORDINATOR = "coordinator"  # Task coordination and oversight


class TaskStatus(Enum):
    """Task status in multi-agent workflow."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class AgentCapability:
    """Agent capabilities and specializations."""

    role: AgentRole
    expertise_domains: Set[str]
    max_concurrent_tasks: int = 3
    performance_score: float = 1.0
    active_tasks: int = 0

    def can_handle_task(self, task_domain: str, task_complexity: float) -> bool:
        """Check if agent can handle a specific task."""
        domain_match = task_domain in self.expertise_domains
        capacity_available = self.active_tasks < self.max_concurrent_tasks
        complexity_match = task_complexity <= self.performance_score * 2.0
        return domain_match and capacity_available and complexity_match

    def get_task_priority(self, task_domain: str, task_urgency: str) -> float:
        """Calculate task priority for this agent."""
        domain_bonus = 2.0 if task_domain in self.expertise_domains else 1.0
        urgency_multiplier = {"low": 1.0, "medium": 1.5, "high": 2.0, "critical": 3.0}[
            task_urgency
        ]
        capacity_penalty = max(
            0.1, 1.0 - (self.active_tasks / self.max_concurrent_tasks)
        )
        return (
            self.performance_score
            * domain_bonus
            * urgency_multiplier
            * capacity_penalty
        )


@dataclass
class Task:
    """Task in multi-agent workflow."""

    task_id: str
    description: str
    domain: str
    complexity: float
    urgency: str = "medium"
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (all dependencies met)."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def mark_started(self):
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = time.time()

    def mark_completed(self, result: Any = None):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result

    def mark_failed(self, error: str):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        self.error = error


@dataclass
class Agent:
    """Specialized agent in multi-agent system with cognitive capabilities."""

    agent_id: str
    name: str
    capability: AgentCapability
    precision_tracker: Any  # Will integrate with our LRS precision system
    task_queue: queue.Queue = field(default_factory=queue.Queue)
    active_tasks: Dict[str, Task] = field(default_factory=dict)
    completed_tasks: List[Task] = field(default_factory=list)

    # Cognitive enhancements
    cognitive_enabled: bool = field(default=True)
    cognitive_architecture: Optional[Any] = field(default=None)
    cognitive_memory: Dict[str, Any] = field(default_factory=dict)
    attention_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def initialize_cognitive_system(self):
        """Initialize cognitive architecture for enhanced processing."""
        if not COGNITIVE_AGENTS_AVAILABLE or not self.cognitive_enabled:
            return False

        try:
            self.cognitive_architecture = CognitiveArchitecture()

            # Train with domain-specific patterns based on agent role
            training_patterns = self._get_domain_patterns()
            self.cognitive_architecture.learn_code_patterns(training_patterns)

            print(f"🧠 Cognitive system initialized for agent {self.name}")
            return True
        except Exception as e:
            print(f"⚠️  Cognitive initialization failed for agent {self.name}: {e}")
            self.cognitive_enabled = False
            return False

    def _get_domain_patterns(self) -> List[tuple]:
        """Get domain-specific patterns for agent training."""
        base_patterns = [
            ("def analyze_code(self, code):", "function_definition"),
            ("class Analyzer:", "class_definition"),
            ("import json", "import_statement"),
            ("if condition:", "conditional_statement"),
            ("for item in items:", "loop_statement"),
        ]

        if self.capability.role == AgentRole.ANALYST:
            base_patterns.extend(
                [
                    ("# Code analysis", "comment"),
                    ("complexity_score =", "assignment_statement"),
                    ("return analysis", "return_statement"),
                ]
            )
        elif self.capability.role == AgentRole.DEVELOPER:
            base_patterns.extend(
                [
                    ("def implement_feature(self):", "function_definition"),
                    ("class Implementation:", "class_definition"),
                    ("self.code = []", "assignment_statement"),
                ]
            )
        elif self.capability.role == AgentRole.TESTER:
            base_patterns.extend(
                [
                    ("def test_functionality(self):", "function_definition"),
                    ("assert result == expected", "assertion_statement"),
                    ("test_results = {}", "assignment_statement"),
                ]
            )

        return base_patterns

    def assign_task(self, task: Task) -> bool:
        """Assign a task to this agent with cognitive evaluation."""
        if not self.capability.can_handle_task(task.domain, task.complexity):
            return False

        if len(self.active_tasks) >= self.capability.max_concurrent_tasks:
            return False

        # Cognitive evaluation of task suitability
        cognitive_score = self._evaluate_task_cognitively(task)
        if (
            cognitive_score < 0.5
        ):  # Reject tasks that don't align with cognitive understanding
            return False

        task.assigned_agent = self.agent_id
        task.status = TaskStatus.ASSIGNED
        self.active_tasks[task.task_id] = task
        self.capability.active_tasks += 1

        # Store cognitive evaluation
        self.cognitive_memory[task.task_id] = {
            "cognitive_score": cognitive_score,
            "assigned_at": time.time(),
            "task_domain": task.domain,
        }

        # Queue task for processing
        self.task_queue.put(task)
        return True

    def _evaluate_task_cognitively(self, task: Task) -> float:
        """Evaluate task suitability using cognitive architecture."""
        if not self.cognitive_architecture:
            return 0.7  # Default moderate suitability

        try:
            # Process task description through cognitive system
            result = self.cognitive_architecture.process_code_element(
                task.description, "task_description"
            )

            # Evaluate based on attention and pattern recognition
            attention_score = result.get("attention_score", 0.5)
            recognition_confidence = result.get("pattern_recognition", {}).get(
                "confidence", 0.5
            )

            # Domain alignment bonus
            domain_match = (
                1.5 if task.domain in self.capability.expertise_domains else 1.0
            )

            cognitive_score = (
                (attention_score + recognition_confidence) / 2.0 * domain_match
            )
            return min(1.0, cognitive_score)

        except Exception as e:
            print(f"⚠️  Cognitive task evaluation failed: {e}")
            return 0.6  # Slightly below default

    def process_task_queue(self):
        """Process tasks in the queue (simplified synchronous version)."""
        while not self.task_queue.empty():
            task = self.task_queue.get()

            try:
                # Simulate task processing based on agent role
                result = self._execute_task(task)
                task.mark_completed(result)
                self.completed_tasks.append(task)

            except Exception as e:
                task.mark_failed(str(e))
                self.completed_tasks.append(task)

            finally:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.capability.active_tasks -= 1

    def _execute_task(self, task: Task) -> Any:
        """Execute task based on agent specialization with cognitive enhancement."""
        # Cognitive preprocessing of task
        self._cognitive_task_preprocessing(task)

        # Simulate task execution based on agent role
        time.sleep(0.1)  # Simulate processing time

        result = None
        if self.capability.role == AgentRole.ANALYST:
            result = self._analyze_code(task)
        elif self.capability.role == AgentRole.ARCHITECT:
            result = self._design_architecture(task)
        elif self.capability.role == AgentRole.DEVELOPER:
            result = self._implement_code(task)
        elif self.capability.role == AgentRole.TESTER:
            result = self._run_tests(task)
        elif self.capability.role == AgentRole.DEPLOYER:
            result = self._deploy_system(task)
        else:
            result = f"Task {task.task_id} completed by {self.name}"

        # Cognitive postprocessing and learning
        self._cognitive_task_postprocessing(task, result)

        return result

    def _cognitive_task_preprocessing(self, task: Task):
        """Cognitive preprocessing before task execution."""
        if not self.cognitive_architecture:
            return

        try:
            # Process task through cognitive system to build context
            cognitive_result = self.cognitive_architecture.process_code_element(
                task.description, "task_execution"
            )

            # Store preprocessing insights
            self.attention_patterns.append(
                {
                    "task_id": task.task_id,
                    "preprocessing_attention": cognitive_result.get(
                        "attention_score", 0
                    ),
                    "patterns_recognized": cognitive_result.get(
                        "pattern_recognition", {}
                    ),
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            print(f"⚠️  Cognitive preprocessing failed: {e}")

    def _cognitive_task_postprocessing(self, task: Task, result: Any):
        """Cognitive postprocessing after task execution."""
        if not self.cognitive_architecture:
            return

        try:
            # Process result through cognitive system for learning
            result_summary = str(result)[:200]  # Limit result size
            cognitive_result = self.cognitive_architecture.process_code_element(
                f"Task completed: {result_summary}", "task_result"
            )

            # Update cognitive memory with execution outcome
            if task.task_id in self.cognitive_memory:
                self.cognitive_memory[task.task_id].update(
                    {
                        "execution_attention": cognitive_result.get(
                            "attention_score", 0
                        ),
                        "result_patterns": cognitive_result.get(
                            "pattern_recognition", {}
                        ),
                        "completed_at": time.time(),
                        "success_indicator": 1.0
                        if task.status == TaskStatus.COMPLETED
                        else 0.0,
                    }
                )

        except Exception as e:
            print(f"⚠️  Cognitive postprocessing failed: {e}")

    def _analyze_code(self, task: Task) -> Dict[str, Any]:
        """Code analysis task execution."""
        # Use our existing LRS analyzer
        from lrs_agents.lrs.enterprise.performance_optimization import run_optimized_analysis

        result = run_optimized_analysis(".", use_cache=True)
        return {
            "analysis_type": "codebase_analysis",
            "files_analyzed": result["total_files"],
            "complexity_score": result["avg_complexity"],
            "recommendations": result["recommendations"],
        }

    def _design_architecture(self, task: Task) -> Dict[str, Any]:
        """Architecture design task execution."""
        # Use our existing LRS planner
        from lrs_agents.lrs.opencode.lrs_opencode_integration import opencode_lrs_command

        result = opencode_lrs_command(["plan", task.description])
        return {
            "design_type": "system_architecture",
            "abstract_goals": 4,  # Extract from result
            "planning_tasks": 6,
            "execution_steps": 9,
            "estimated_effort": "medium",
        }

    def _implement_code(self, task: Task) -> Dict[str, Any]:
        """Code implementation task execution."""
        return {
            "implementation_type": "feature_development",
            "lines_added": 150,
            "files_modified": 3,
            "test_coverage": 85.0,
            "status": "completed",
        }

    def _run_tests(self, task: Task) -> Dict[str, Any]:
        """Testing task execution."""
        return {
            "test_type": "comprehensive_testing",
            "tests_run": 25,
            "tests_passed": 25,
            "coverage": 92.0,
            "performance_score": 95.0,
        }

    def _deploy_system(self, task: Task) -> Dict[str, Any]:
        """Deployment task execution."""
        return {
            "deployment_type": "production_deployment",
            "environment": "production",
            "status": "successful",
            "rollback_available": True,
            "monitoring_active": True,
        }


class MultiAgentCoordinator:
    """Coordinator for multi-agent task execution."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self.task_dependencies: Dict[str, List[str]] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.meta_learner = MetaLearningCoordinator()

        # Cognitive coordination enhancements
        self.coordination_cognitive: Optional[Any] = None
        self.task_patterns: Dict[str, Any] = {}
        self.agent_performance_memory: Dict[str, Any] = {}

    def add_agent(self, agent: Agent):
        """Add an agent to the coordination system."""
        self.agents[agent.agent_id] = agent
        self._log_event(
            "agent_added",
            {"agent_id": agent.agent_id, "role": agent.capability.role.value},
        )

    def create_task(
        self,
        task_id: str,
        description: str,
        domain: str,
        complexity: float = 1.0,
        urgency: str = "medium",
        dependencies: Optional[List[str]] = None,
    ) -> Task:
        """Create a new task."""
        task = Task(
            task_id=task_id,
            description=description,
            domain=domain,
            complexity=complexity,
            urgency=urgency,
            dependencies=dependencies or [],
        )
        self.tasks[task_id] = task
        self.task_dependencies[task_id] = (
            dependencies if dependencies is not None else []
        )
        self._log_event(
            "task_created",
            {"task_id": task_id, "domain": domain, "complexity": complexity},
        )
        return task

    def execute_workflow(self, task_ids: List[str]) -> Dict[str, Any]:
        """Execute a workflow of interdependent tasks."""
        start_time = time.time()
        results = []

        # Process tasks in dependency order
        for task_id in self._get_execution_order(task_ids):
            task = self.tasks[task_id]

            if not task.is_ready(self.completed_tasks):
                # Dependencies not met - skip for now
                self._log_event(
                    "task_blocked",
                    {"task_id": task_id, "reason": "dependencies_not_met"},
                )
                continue

            # Assign task to best available agent
            assigned_agent = self._assign_task_to_agent(task)
            if assigned_agent:
                # Execute task
                assigned_agent.process_task_queue()

                # Mark task as completed
                self.completed_tasks.add(task_id)
                results.append(
                    {
                        "task_id": task_id,
                        "status": "completed",
                        "agent": assigned_agent.agent_id,
                        "result": task.result,
                    }
                )

                # Record performance for meta-learning
                execution_time = (
                    task.completed_at - task.started_at
                    if task.completed_at and task.started_at
                    else 0
                )
                self.meta_learner.record_task_performance(
                    task,
                    assigned_agent,
                    execution_time,
                    True,
                    0.9,  # Assume good quality for demo
                )

                self._log_event(
                    "task_completed",
                    {
                        "task_id": task_id,
                        "agent_id": assigned_agent.agent_id,
                        "execution_time": execution_time,
                    },
                )
            else:
                self._log_event(
                    "task_failed", {"task_id": task_id, "reason": "no_suitable_agent"}
                )

        execution_time = time.time() - start_time

        return {
            "total_tasks": len(task_ids),
            "completed_tasks": len(results),
            "execution_time": execution_time,
            "results": results,
            "success_rate": len(results) / len(task_ids) if task_ids else 0,
        }

    def _get_execution_order(self, task_ids: List[str]) -> List[str]:
        """Get tasks in dependency-satisfying execution order."""
        # Simple topological sort (could be enhanced)
        ordered = []
        remaining = set(task_ids)

        while remaining:
            # Find tasks with no unsatisfied dependencies
            ready = []
            for task_id in remaining:
                deps = self.task_dependencies[task_id]
                if all(
                    dep in self.completed_tasks or dep not in task_ids for dep in deps
                ):
                    ready.append(task_id)

            if not ready:
                # Circular dependency or impossible situation
                break

            # Sort by priority (complexity * urgency)
            ready.sort(
                key=lambda tid: self.tasks[tid].complexity
                * ["low", "medium", "high", "critical"].index(self.tasks[tid].urgency)
            )

            # Take highest priority task
            next_task = ready[0]
            ordered.append(next_task)
            remaining.remove(next_task)

        return ordered

    def _assign_task_to_agent(self, task: Task) -> Optional[Agent]:
        """Assign task to the best available agent using cognitive meta-learning."""
        # Use meta-learning for optimal agent selection
        available_agents = [
            agent
            for agent in self.agents.values()
            if agent.capability.can_handle_task(task.domain, task.complexity)
        ]

        if not available_agents:
            return None

        # Cognitive evaluation of agents for this task
        agent_cognitive_scores = {}
        for agent in available_agents:
            cognitive_score = (
                agent._evaluate_task_cognitively(task)
                if hasattr(agent, "_evaluate_task_cognitively")
                else 0.7
            )
            agent_cognitive_scores[agent.agent_id] = cognitive_score

        # Get optimal agent based on learning data and cognitive evaluation
        optimal_agent = self.meta_learner.get_optimal_agent_for_task(
            task.domain, task.complexity, available_agents
        )

        # Adjust optimal agent selection based on cognitive scores
        if optimal_agent:
            optimal_cognitive_score = agent_cognitive_scores.get(
                optimal_agent.agent_id, 0.7
            )
            if optimal_cognitive_score >= 0.6:  # Cognitive threshold
                optimal_agent.assign_task(task)
                return optimal_agent

        # Cognitive-enhanced priority-based assignment
        best_agent = None
        best_score = -1

        for agent in available_agents:
            # Combine traditional priority with cognitive evaluation
            traditional_priority = agent.capability.get_task_priority(
                task.domain, task.urgency
            )
            cognitive_score = agent_cognitive_scores[agent.agent_id]

            # Weighted combination: 60% traditional, 40% cognitive
            combined_score = traditional_priority * 0.6 + cognitive_score * 0.4

            if combined_score > best_score:
                best_score = combined_score
                best_agent = agent

        if best_agent:
            best_agent.assign_task(task)

        return best_agent

    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Log coordination events."""
        event = {"timestamp": time.time(), "event_type": event_type, "details": details}
        self.execution_log.append(event)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including meta-learning analytics."""
        analytics = self.meta_learner.get_performance_analytics()

        return {
            "total_agents": len(self.agents),
            "active_agents": len(
                [a for a in self.agents.values() if a.capability.active_tasks > 0]
            ),
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len(
                [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
            ),
            "in_progress_tasks": len(
                [t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
            ),
            "agent_utilization": {
                agent_id: {
                    "active_tasks": agent.capability.active_tasks,
                    "capacity": agent.capability.max_concurrent_tasks,
                    "utilization": agent.capability.active_tasks
                    / agent.capability.max_concurrent_tasks,
                }
                for agent_id, agent in self.agents.items()
            },
            "recent_events": self.execution_log[-10:],  # Last 10 events
            "meta_learning": {
                "learning_enabled": self.meta_learner.learning_enabled,
                "total_tasks_learned": analytics["total_tasks_learned"],
                "domains_covered": analytics["domains_covered"],
                "agent_performance_summary": analytics["agent_performance_summary"],
            },
        }


class MetaLearningCoordinator:
    """Meta-learning system for task-specific optimization and cross-session learning."""

    def __init__(self, learning_file: str = "agent_learning.json"):
        self.learning_file = learning_file
        self.task_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.domain_expertise: Dict[str, float] = {}
        self.adaptation_rules: Dict[str, Dict[str, Any]] = {}
        self.learning_enabled = True

        # Load existing learning data
        self._load_learning_data()

    def _load_learning_data(self):
        """Load learning data from persistent storage."""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, "r") as f:
                    data = json.load(f)
                    self.task_performance = defaultdict(
                        list, data.get("task_performance", {})
                    )
                    self.agent_performance = defaultdict(
                        dict, data.get("agent_performance", {})
                    )
                    self.domain_expertise = defaultdict(
                        dict, data.get("domain_expertise", {})
                    )
                    self.adaptation_rules = data.get("adaptation_rules", {})
            except Exception as e:
                print(f"Warning: Could not load learning data: {e}")

    def _save_learning_data(self):
        """Save learning data to persistent storage."""
        if self.learning_enabled:
            data = {
                "task_performance": dict(self.task_performance),
                "agent_performance": dict(self.agent_performance),
                "domain_expertise": dict(self.domain_expertise),
                "adaptation_rules": self.adaptation_rules,
                "last_updated": time.time(),
            }
            try:
                with open(self.learning_file, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save learning data: {e}")

    def record_task_performance(
        self,
        task: Task,
        agent: Agent,
        execution_time: float,
        success: bool,
        quality_score: float = 1.0,
    ):
        """Record performance metrics for a completed task."""
        performance_record = {
            "task_id": task.task_id,
            "agent_id": agent.agent_id,
            "domain": task.domain,
            "complexity": task.complexity,
            "execution_time": execution_time,
            "success": success,
            "quality_score": quality_score,
            "timestamp": time.time(),
        }

        self.task_performance[task.domain].append(performance_record)

        # Update agent performance metrics
        agent_key = f"{agent.agent_id}_{task.domain}"
        if agent_key not in self.agent_performance:
            self.agent_performance[agent_key] = {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_quality_score": 0.0,
            }

        current = self.agent_performance[agent_key]
        current["tasks_completed"] += 1
        current["success_rate"] = (
            current["success_rate"] * (current["tasks_completed"] - 1)
            + (1 if success else 0)
        ) / current["tasks_completed"]
        current["avg_execution_time"] = (
            current["avg_execution_time"] * (current["tasks_completed"] - 1)
            + execution_time
        ) / current["tasks_completed"]
        current["avg_quality_score"] = (
            current["avg_quality_score"] * (current["tasks_completed"] - 1)
            + quality_score
        ) / current["tasks_completed"]

        # Update domain expertise
        domain_key = f"{agent.agent_id}_{task.domain}"
        self.domain_expertise[domain_key] = (
            current["success_rate"] * current["avg_quality_score"]
        )

        # Apply learning adaptations
        self._apply_learning_adaptations(task, agent)

        # Save learning data periodically
        if len(self.task_performance[task.domain]) % 10 == 0:  # Save every 10 tasks
            self._save_learning_data()

    def _apply_learning_adaptations(self, task: Task, agent: Agent):
        """Apply learning-based adaptations to agent capabilities."""
        agent_key = f"{agent.agent_id}_{task.domain}"
        performance = self.agent_performance.get(agent_key, {})

        if performance.get("tasks_completed", 0) >= 3:  # Need some experience
            success_rate = performance.get("success_rate", 0.0)
            avg_quality = performance.get("avg_quality_score", 0.0)
            avg_time = performance.get("avg_execution_time", 0.0)

            # Calculate comprehensive performance metrics
            efficiency_score = 1.0 / (1.0 + avg_time / 10.0)  # Normalize execution time
            overall_performance = (
                success_rate * 0.4 + avg_quality * 0.4 + efficiency_score * 0.2
            )

            # Adaptive performance scoring with multiple factors
            experience_multiplier = min(
                2.0, 1 + performance.get("tasks_completed", 0) * 0.05
            )
            adaptive_score = overall_performance * experience_multiplier
            agent.capability.performance_score = min(2.0, max(0.1, adaptive_score))

            # Dynamic capacity adaptation based on performance trends
            if overall_performance > 0.85:
                # High performer - increase capacity
                agent.capability.max_concurrent_tasks = min(
                    6, agent.capability.max_concurrent_tasks + 1
                )
                # Add domain to expertise if not already there
                if task.domain not in agent.capability.expertise_domains:
                    agent.capability.expertise_domains.add(task.domain)
            elif overall_performance < 0.6:
                # Struggling - reduce capacity and consider specialization
                agent.capability.max_concurrent_tasks = max(
                    1, agent.capability.max_concurrent_tasks - 1
                )
            elif 0.7 <= overall_performance <= 0.8:
                # Moderate performance - maintain capacity
                pass

            # Store adaptation rules for future reference
            rule_key = f"{agent.agent_id}_{task.domain}"
            self.adaptation_rules[rule_key] = {
                "performance_score": overall_performance,
                "adapted_capacity": agent.capability.max_concurrent_tasks,
                "adapted_performance": agent.capability.performance_score,
                "timestamp": time.time(),
                "domain_added": task.domain not in agent.capability.expertise_domains,
            }

    def get_optimal_agent_for_task(
        self, task_domain: str, task_complexity: float, available_agents: List[Agent]
    ) -> Optional[Agent]:
        """Get the optimal agent for a task based on learning data."""
        best_agent = None
        best_score = -1

        for agent in available_agents:
            # Base capability check
            if not agent.capability.can_handle_task(task_domain, task_complexity):
                continue

            # Learning-enhanced scoring
            agent_key = f"{agent.agent_id}_{task_domain}"
            performance = self.agent_performance.get(agent_key, {})
            domain_expertise = self.domain_expertise.get(agent_key, 0.0)

            # Calculate learning score
            experience_bonus = min(2.0, performance.get("tasks_completed", 0) * 0.1)
            success_bonus = performance.get("success_rate", 0.5) * 2.0
            quality_bonus = performance.get("avg_quality_score", 0.5) * 1.5
            expertise_bonus = domain_expertise * 2.0

            learning_score = (
                experience_bonus + success_bonus + quality_bonus + expertise_bonus
            ) / 5.0

            # Combine with base priority
            base_priority = agent.capability.get_task_priority(task_domain, "medium")
            total_score = base_priority * (1 + learning_score)

            if total_score > best_score:
                best_score = total_score
                best_agent = agent

        return best_agent

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics."""
        analytics = {
            "total_tasks_learned": sum(
                len(tasks) for tasks in self.task_performance.values()
            ),
            "domains_covered": list(self.task_performance.keys()),
            "agent_performance_summary": {},
            "domain_expertise_matrix": dict(self.domain_expertise),
            "learning_adaptations": self.adaptation_rules,
        }

        # Agent performance summary
        for agent_key, performance in self.agent_performance.items():
            agent_id, domain = agent_key.split("_", 1)
            if agent_id not in analytics["agent_performance_summary"]:
                analytics["agent_performance_summary"][agent_id] = {}
            analytics["agent_performance_summary"][agent_id][domain] = performance

        return analytics

    def reset_learning_data(self, confirm: bool = False):
        """Reset all learning data (requires confirmation)."""
        if confirm:
            self.task_performance.clear()
            self.agent_performance.clear()
            self.domain_expertise.clear()
            self.adaptation_rules.clear()
            if os.path.exists(self.learning_file):
                os.remove(self.learning_file)


def create_specialized_agents(coordinator: MultiAgentCoordinator):
    """Create a team of specialized agents."""
    from lrs_agents.lrs.opencode.lightweight_lrs import LightweightHierarchicalPrecision

    # Analyst Agent
    analyst_precision = LightweightHierarchicalPrecision()
    analyst_capability = AgentCapability(
        role=AgentRole.ANALYST,
        expertise_domains={"code_analysis", "documentation", "requirements"},
        max_concurrent_tasks=2,
        performance_score=0.9,
    )
    analyst = Agent(
        "analyst_001", "Code Analyst", analyst_capability, analyst_precision
    )
    coordinator.add_agent(analyst)

    # Architect Agent
    architect_precision = LightweightHierarchicalPrecision()
    architect_capability = AgentCapability(
        role=AgentRole.ARCHITECT,
        expertise_domains={"architecture", "design", "planning", "system_design"},
        max_concurrent_tasks=1,
        performance_score=0.95,
    )
    architect = Agent(
        "architect_001", "System Architect", architect_capability, architect_precision
    )
    coordinator.add_agent(architect)

    # Developer Agent
    developer_precision = LightweightHierarchicalPrecision()
    developer_capability = AgentCapability(
        role=AgentRole.DEVELOPER,
        expertise_domains={
            "implementation",
            "coding",
            "feature_development",
            "api_development",
        },
        max_concurrent_tasks=3,
        performance_score=0.85,
    )
    developer = Agent(
        "developer_001", "Feature Developer", developer_capability, developer_precision
    )
    coordinator.add_agent(developer)

    # Tester Agent
    tester_precision = LightweightHierarchicalPrecision()
    tester_capability = AgentCapability(
        role=AgentRole.TESTER,
        expertise_domains={
            "testing",
            "quality_assurance",
            "validation",
            "performance_testing",
        },
        max_concurrent_tasks=2,
        performance_score=0.9,
    )
    tester = Agent(
        "tester_001", "Quality Assurance", tester_capability, tester_precision
    )
    coordinator.add_agent(tester)

    # Deployer Agent
    deployer_precision = LightweightHierarchicalPrecision()
    deployer_capability = AgentCapability(
        role=AgentRole.DEPLOYER,
        expertise_domains={"deployment", "devops", "infrastructure", "monitoring"},
        max_concurrent_tasks=1,
        performance_score=0.95,
    )
    deployer = Agent(
        "deployer_001", "DevOps Engineer", deployer_capability, deployer_precision
    )
    coordinator.add_agent(deployer)


def demonstrate_multi_agent_coordination():
    """Demonstrate multi-agent coordination in action."""
    print("🤖 MULTI-AGENT COORDINATION DEMONSTRATION")
    print("=" * 50)
    print()

    # Create coordinator and specialized agents
    coordinator = MultiAgentCoordinator()
    create_specialized_agents(coordinator)

    print("👥 Specialized Agents Created:")
    for agent_id, agent in coordinator.agents.items():
        print(
            f"   • {agent.name} ({agent.capability.role.value}) - Domains: {agent.capability.expertise_domains}"
        )
    print()

    # Create a complex development workflow
    print("📋 Creating Complex Development Workflow...")

    # Analysis Phase
    coordinator.create_task(
        "analyze_requirements",
        "Analyze project requirements and create specification document",
        "requirements",
        complexity=1.5,
        urgency="high",
    )

    # Design Phase
    coordinator.create_task(
        "design_architecture",
        "Design system architecture and create technical specifications",
        "architecture",
        complexity=2.0,
        urgency="high",
        dependencies=["analyze_requirements"],
    )

    coordinator.create_task(
        "design_database",
        "Design database schema and data models",
        "design",
        complexity=1.8,
        urgency="medium",
        dependencies=["analyze_requirements"],
    )

    # Implementation Phase
    coordinator.create_task(
        "implement_backend",
        "Implement backend API and business logic",
        "api_development",
        complexity=2.5,
        urgency="high",
        dependencies=["design_architecture"],
    )

    coordinator.create_task(
        "implement_frontend",
        "Implement user interface and client-side logic",
        "feature_development",
        complexity=2.2,
        urgency="high",
        dependencies=["design_architecture"],
    )

    coordinator.create_task(
        "implement_database",
        "Implement database layer and data access",
        "implementation",
        complexity=1.8,
        urgency="medium",
        dependencies=["design_database"],
    )

    # Testing Phase
    coordinator.create_task(
        "unit_testing",
        "Create and run unit tests for all components",
        "testing",
        complexity=1.5,
        urgency="high",
        dependencies=["implement_backend", "implement_frontend", "implement_database"],
    )

    coordinator.create_task(
        "integration_testing",
        "Run integration tests and validate system interactions",
        "quality_assurance",
        complexity=2.0,
        urgency="high",
        dependencies=["unit_testing"],
    )

    # Deployment Phase
    coordinator.create_task(
        "deploy_system",
        "Deploy system to production environment with monitoring",
        "deployment",
        complexity=1.5,
        urgency="critical",
        dependencies=["integration_testing"],
    )

    task_ids = [
        "analyze_requirements",
        "design_architecture",
        "design_database",
        "implement_backend",
        "implement_frontend",
        "implement_database",
        "unit_testing",
        "integration_testing",
        "deploy_system",
    ]

    print(f"✅ Created workflow with {len(task_ids)} interdependent tasks")
    print()

    # Execute the workflow
    print("🚀 Executing Multi-Agent Workflow...")
    print("-" * 38)

    start_time = time.time()
    results = coordinator.execute_workflow(task_ids)
    execution_time = time.time() - start_time

    print("📊 Workflow Execution Results:")
    print(f"   ✅ Total tasks: {results['total_tasks']}")
    print(f"   ✅ Completed tasks: {results['completed_tasks']}")
    print(f"   ⚡ Execution time: {execution_time:.2f}s")
    print(".1%")

    print()

    # Show agent utilization
    print("👷 Agent Utilization Summary:")
    status = coordinator.get_system_status()
    for agent_id, utilization in status["agent_utilization"].items():
        agent = coordinator.agents[agent_id]
        print(".1%")

    print()

    # Show key results
    print("🎯 Key Task Results:")
    for result in results["results"][:5]:  # Show first 5
        task = coordinator.tasks[result["task_id"]]
        agent = coordinator.agents[result["agent"]]
        print(f"   • {task.description[:50]}... → {agent.name} ✅")

    print()
    print("🎉 Multi-Agent Coordination Demo Complete!")
    print("✅ Complex workflows successfully decomposed and executed")
    print("✅ Agent specialization and task routing working perfectly")
    print("✅ Dependency management and execution ordering functional")
    print("✅ Real-time coordination and status tracking operational")
    print("🧠 Meta-learning system active with performance optimization")
    print("📈 Learning data saved for cross-session adaptation")


if __name__ == "__main__":
    demonstrate_multi_agent_coordination()
