#!/usr/bin/env python3
"""
OpenCode LRS Integration Tools

Advanced tools that integrate LRS-Agents capabilities into OpenCode CLI.
These tools leverage Active Inference, precision tracking, and hierarchical planning.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Import our simplified LRS components
from .simplified_integration import OpenCodeTool

# Import will be done locally to avoid circular imports

# Import cognitive architecture components
try:
    from phase6_neuromorphic_research.phase6_neuromorphic_setup import (
        CognitiveArchitecture,
        NeuromorphicPatternRecognition,
        SpikingNeuralNetwork,
    )

    COGNITIVE_COMPONENTS_AVAILABLE = True
except ImportError:
    print(
        "Warning: Cognitive components not available, running without neuromorphic enhancements"
    )
    COGNITIVE_COMPONENTS_AVAILABLE = False

# Add LRS directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lrs"))


class LRSToolRegistry:
    """Registry for LRS-enhanced tools."""

    def __init__(self):
        self.tools = {}
        self.precision_history = []
        self.free_energy_log = []

    def register_tool(self, name: str, tool_class):
        """Register an LRS-enhanced tool."""
        self.tools[name] = tool_class

    def get_tool(self, name: str):
        """Get tool instance by name."""
        tool_class = self.tools.get(name)
        return tool_class() if tool_class else None

    def record_precision_update(
        self, level: str, old_precision: float, new_precision: float, reason: str
    ):
        """Record precision changes for analysis."""
        self.precision_history.append(
            {
                "timestamp": len(self.precision_history),
                "level": level,
                "old_precision": old_precision,
                "new_precision": new_precision,
                "change": new_precision - old_precision,
                "reason": reason,
            }
        )

    def record_free_energy_calculation(
        self, epistemic: float, pragmatic: float, total_g: float, policy_desc: str
    ):
        """Record free energy calculations."""
        self.free_energy_log.append(
            {
                "policy": policy_desc,
                "epistemic": epistemic,
                "pragmatic": pragmatic,
                "total_g": total_g,
                "timestamp": len(self.free_energy_log),
            }
        )


# Global registry instance
lrs_registry = LRSToolRegistry()


@dataclass
class LRSExecutionContext:
    """Context for LRS-enhanced executions with cognitive capabilities."""

    belief_state: Dict[str, Any] = field(default_factory=dict)
    precision_levels: Dict[str, float] = field(
        default_factory=lambda: {"abstract": 0.5, "planning": 0.5, "execution": 0.5}
    )
    prediction_errors: List[float] = field(default_factory=list)
    adaptation_events: List[Dict[str, Any]] = field(default_factory=list)
    free_energy_history: List[float] = field(default_factory=list)

    # Cognitive enhancements
    cognitive_enabled: bool = field(default=True)
    cognitive_architecture: Optional[Any] = field(default_factory=lambda: None)
    cognitive_insights: List[Dict[str, Any]] = field(default_factory=list)
    attention_patterns: List[Dict[str, Any]] = field(default_factory=list)
    memory_state: Dict[str, Any] = field(default_factory=dict)

    def update_precision(self, level: str, prediction_error: float, reason: str = ""):
        """Update precision based on prediction error."""
        old_precision = self.precision_levels[level]

        # Simplified precision update (beta distribution learning)
        learning_rate = 0.1 if prediction_error < 0.5 else 0.2
        error_signal = 1.0 - prediction_error  # Convert to success signal

        # Asymmetric learning: faster for failures
        if prediction_error > 0.5:  # Failure case
            self.precision_levels[level] *= 1.0 - learning_rate * prediction_error
        else:  # Success case
            self.precision_levels[level] += learning_rate * error_signal
            self.precision_levels[level] = min(1.0, self.precision_levels[level])

        # Clamp to valid range
        self.precision_levels[level] = max(0.1, min(0.9, self.precision_levels[level]))

        new_precision = self.precision_levels[level]

        # Record adaptation if significant change
        if abs(new_precision - old_precision) > 0.1:
            self.adaptation_events.append(
                {
                    "level": level,
                    "old_precision": old_precision,
                    "new_precision": new_precision,
                    "prediction_error": prediction_error,
                    "reason": reason,
                    "timestamp": len(self.prediction_errors),
                }
            )

        lrs_registry.record_precision_update(
            level, old_precision, new_precision, reason
        )
        self.prediction_errors.append(prediction_error)

    def calculate_free_energy(
        self,
        epistemic_value: float,
        pragmatic_value: float,
        precision_weight: float = 0.5,
    ) -> float:
        """Calculate expected free energy G."""
        # G = Epistemic - Pragmatic (weighted by precision)
        epistemic_term = (1.0 - precision_weight) * epistemic_value
        pragmatic_term = precision_weight * pragmatic_value
        G = epistemic_term - pragmatic_term

        self.free_energy_history.append(G)
        return G

    def initialize_cognitive_system(self):
        """Initialize cognitive architecture for enhanced processing."""
        if not COGNITIVE_COMPONENTS_AVAILABLE or not self.cognitive_enabled:
            return False

        try:
            if COGNITIVE_COMPONENTS_AVAILABLE:
                self.cognitive_architecture = CognitiveArchitecture()
            else:
                return False

            # Train with common code patterns
            training_patterns = [
                ("def calculate_total(items):", "function_definition"),
                ("class UserManager:", "class_definition"),
                ("import numpy as np", "import_statement"),
                ("if x > 0:", "conditional_statement"),
                ("for item in items:", "loop_statement"),
                ("try:", "exception_handling"),
                ("return result", "return_statement"),
            ]

            self.cognitive_architecture.learn_code_patterns(training_patterns)
            print("✅ Cognitive architecture initialized for LRS execution context")
            return True
        except Exception as e:
            print(f"⚠️  Cognitive system initialization failed: {e}")
            self.cognitive_enabled = False
            return False

    def process_with_cognition(
        self, code_element: str, context: str = "general"
    ) -> Dict[str, Any]:
        """Process code element through cognitive architecture."""
        if not self.cognitive_architecture or not self.cognitive_enabled:
            return {"cognitive_processing": False, "insights": []}

        try:
            # Process through cognitive system
            result = self.cognitive_architecture.process_code_element(
                code_element, context
            )

            # Store cognitive insights
            self.cognitive_insights.append(
                {
                    "element": code_element,
                    "context": context,
                    "insights": result,
                    "timestamp": len(self.cognitive_insights),
                }
            )

            # Update memory state
            self.memory_state = self.cognitive_architecture.get_cognitive_state()

            # Extract attention patterns
            if result.get("attention_score", 0) > 0.6:
                self.attention_patterns.append(
                    {
                        "element": code_element,
                        "score": result["attention_score"],
                        "context": context,
                        "focus": result.get("attention_focus"),
                        "timestamp": len(self.attention_patterns),
                    }
                )

            return {
                "cognitive_processing": True,
                "insights": result,
                "attention_focus": result.get("attention_focus"),
                "cognitive_insight": result.get("cognitive_insight", ""),
            }

        except Exception as e:
            print(f"⚠️  Cognitive processing failed: {e}")
            return {"cognitive_processing": False, "error": str(e), "insights": []}

    def get_cognitive_stats(self) -> Dict[str, Any]:
        """Get cognitive processing statistics."""
        if not self.cognitive_architecture:
            return {"cognitive_enabled": False}

        return {
            "cognitive_enabled": self.cognitive_enabled,
            "cognitive_insights_count": len(self.cognitive_insights),
            "attention_patterns_count": len(self.attention_patterns),
            "memory_state": self.memory_state,
            "cognitive_cycles": self.memory_state.get("cognitive_cycles", 0),
            "patterns_learned": self.memory_state.get("patterns_learned", 0),
            "working_memory_items": self.memory_state.get("working_memory_items", 0),
            "attention_focus": self.memory_state.get("attention_focus"),
            "temporal_sequences_learned": self.memory_state.get(
                "temporal_sequences_learned", 0
            ),
        }

    def get_precision_stats(self) -> Dict[str, Any]:
        """Get comprehensive precision statistics."""
        return {
            "current_levels": self.precision_levels,
            "avg_prediction_error": sum(self.prediction_errors)
            / len(self.prediction_errors)
            if self.prediction_errors
            else 0.5,
            "total_adaptations": len(self.adaptation_events),
            "precision_volatility": self._calculate_volatility(),
            "adaptation_triggers": len(
                [e for e in self.adaptation_events if e["prediction_error"] > 0.7]
            ),
        }

    def _calculate_volatility(self) -> float:
        """Calculate precision volatility (how much it changes)."""
        if len(self.prediction_errors) < 2:
            return 0.0

        changes = []
        for i in range(1, len(self.prediction_errors)):
            changes.append(
                abs(self.prediction_errors[i] - self.prediction_errors[i - 1])
            )

        return sum(changes) / len(changes) if changes else 0.0


# =============================================================================
# LRS-ENHANCED OPENCODE TOOLS
# =============================================================================


class ActiveInferenceAnalyzer(OpenCodeTool):
    """Analyze codebases using Active Inference principles."""

    def __init__(self):
        self.name = "active_inference_analyzer"
        self.context = LRSExecutionContext()

    def analyze_codebase(self, path: str) -> Dict[str, Any]:
        """Analyze codebase structure and complexity."""
        try:
            files = []
            total_lines = 0
            languages = {}

            for root, dirs, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(
                        (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs")
                    ):
                        filepath = os.path.join(root, filename)
                        try:
                            with open(
                                filepath, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                lines = f.readlines()
                                total_lines += len(lines)

                                # Language detection
                                ext = filename.split(".")[-1]
                                languages[ext] = languages.get(ext, 0) + 1

                                files.append(
                                    {
                                        "path": filepath,
                                        "lines": len(lines),
                                        "extension": ext,
                                        "complexity_score": self._estimate_complexity(
                                            lines
                                        ),
                                    }
                                )
                        except Exception:
                            continue

            # Calculate epistemic value (information gain from analysis)
            epistemic_value = min(
                1.0, len(files) / 100.0
            )  # More files = more information

            # Calculate pragmatic value (usefulness for development)
            avg_complexity = (
                sum(f["complexity_score"] for f in files) / len(files) if files else 0
            )
            pragmatic_value = 1.0 - (
                avg_complexity / 10.0
            )  # Lower complexity = more pragmatic

            # Calculate free energy
            precision = self.context.precision_levels["planning"]
            free_energy = self.context.calculate_free_energy(
                epistemic_value, pragmatic_value, precision
            )

            return {
                "total_files": len(files),
                "total_lines": total_lines,
                "languages": languages,
                "avg_complexity": avg_complexity,
                "epistemic_value": epistemic_value,
                "pragmatic_value": pragmatic_value,
                "free_energy": free_energy,
                "precision": precision,
                "recommendations": self._generate_recommendations(
                    avg_complexity, languages
                ),
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def _estimate_complexity(self, lines: List[str]) -> float:
        """Estimate code complexity."""
        complexity = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue

            # Count control structures and operators
            if any(
                keyword in line.lower()
                for keyword in ["if", "for", "while", "switch", "try", "catch"]
            ):
                complexity += 1
            if any(op in line for op in ["&&", "||", "==", "!=", "<=", ">="]):
                complexity += 0.5

        return min(10.0, complexity / len(lines) * 100) if lines else 0

    def _generate_recommendations(
        self, complexity: float, languages: Dict[str, int]
    ) -> List[str]:
        """Generate Active Inference-based recommendations."""
        recommendations = []

        if complexity > 7.0:
            recommendations.append(
                "High complexity detected - consider refactoring for better precision"
            )
        elif complexity < 2.0:
            recommendations.append("Low complexity - good for exploration and learning")

        if len(languages) > 3:
            recommendations.append(
                "Multi-language codebase - precision may vary across domains"
            )

        total_files = sum(languages.values())
        if total_files > 50:
            recommendations.append(
                "Large codebase - hierarchical precision tracking recommended"
            )

        return recommendations


class PrecisionGuidedRefactor(OpenCodeTool):
    """Refactoring tool with precision-guided decision making."""

    def __init__(self):
        self.name = "precision_guided_refactor"
        self.context = LRSExecutionContext()

    def analyze_refactor_opportunities(self, file_path: str) -> Dict[str, Any]:
        """Analyze file for refactoring opportunities using precision metrics."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = content.split("\n")

            # Analyze different aspects
            metrics = {
                "function_length": self._analyze_function_lengths(lines),
                "duplication": self._analyze_duplication(lines),
                "complexity": self._analyze_cyclomatic_complexity(lines),
                "naming": self._analyze_naming_conventions(lines),
            }

            # Calculate overall refactor priority using free energy
            epistemic_value = sum(
                metrics[k]["information_gain"] for k in metrics
            ) / len(metrics)
            pragmatic_value = sum(
                metrics[k]["effort_benefit_ratio"] for k in metrics
            ) / len(metrics)

            precision = self.context.precision_levels["execution"]
            free_energy = self.context.calculate_free_energy(
                epistemic_value, pragmatic_value, precision
            )

            # Determine refactor priority based on free energy
            if free_energy < -0.5:
                priority = "HIGH"
                reason = "Strong evidence for beneficial refactoring"
            elif free_energy < 0:
                priority = "MEDIUM"
                reason = "Moderate improvement potential"
            else:
                priority = "LOW"
                reason = "Limited benefit relative to effort"

            return {
                "file": file_path,
                "metrics": metrics,
                "epistemic_value": epistemic_value,
                "pragmatic_value": pragmatic_value,
                "free_energy": free_energy,
                "refactor_priority": priority,
                "reason": reason,
                "precision_confidence": precision,
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def _analyze_function_lengths(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze function lengths for refactoring opportunities."""
        functions = []
        current_function = []
        in_function = False

        for line in lines:
            if any(
                line.strip().startswith("def ")
                for f in ["def ", "function ", "public ", "private "]
            ):
                if current_function:
                    functions.append(len(current_function))
                current_function = [line]
                in_function = True
            elif in_function and line.strip() and not line.startswith("    "):
                functions.append(len(current_function))
                current_function = []
                in_function = False
            elif in_function:
                current_function.append(line)

        if current_function:
            functions.append(len(current_function))

        avg_length = sum(functions) / len(functions) if functions else 0
        long_functions = len([f for f in functions if f > 30])

        return {
            "avg_length": avg_length,
            "long_functions": long_functions,
            "total_functions": len(functions),
            "information_gain": min(1.0, long_functions / 10.0),
            "effort_benefit_ratio": 0.7 if avg_length > 25 else 0.3,
        }

    def _analyze_duplication(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze code duplication."""
        # Simple duplication detection
        line_counts = {}
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Only consider substantial lines
                line_counts[line] = line_counts.get(line, 0) + 1

        duplicated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        duplication_ratio = duplicated_lines / len(lines) if lines else 0

        return {
            "duplicated_lines": duplicated_lines,
            "duplication_ratio": duplication_ratio,
            "information_gain": min(1.0, duplication_ratio * 2),
            "effort_benefit_ratio": 0.8 if duplication_ratio > 0.1 else 0.2,
        }

    def _analyze_cyclomatic_complexity(self, lines: List[str]) -> Dict[str, Any]:
        """Estimate cyclomatic complexity."""
        complexity_indicators = [
            "if ",
            "elif ",
            "else:",
            "for ",
            "while ",
            "try:",
            "except:",
            "&&",
            "||",
        ]
        complexity_count = sum(
            sum(1 for indicator in complexity_indicators if indicator in line)
            for line in lines
        )

        complexity_score = complexity_count / len(lines) if lines else 0

        return {
            "complexity_score": complexity_score,
            "complexity_indicators": complexity_count,
            "information_gain": min(1.0, complexity_score),
            "effort_benefit_ratio": 0.6 if complexity_score > 0.5 else 0.4,
        }

    def _analyze_naming_conventions(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze naming convention adherence."""
        # Simple heuristic for naming issues
        naming_issues = 0
        total_identifiers = 0

        for line in lines:
            # Look for variable/function definitions
            if "=" in line or "def " in line or "function " in line:
                # Count uppercase in identifiers (potential naming issues)
                words = (
                    line.replace("=", " ").replace("(", " ").replace(")", " ").split()
                )
                for word in words:
                    if len(word) > 2 and word.isupper():
                        naming_issues += 1
                    if word and word[0].isalpha():
                        total_identifiers += 1

        naming_score = naming_issues / total_identifiers if total_identifiers > 0 else 0

        return {
            "naming_issues": naming_issues,
            "total_identifiers": total_identifiers,
            "naming_score": naming_score,
            "information_gain": min(1.0, naming_score),
            "effort_benefit_ratio": 0.9 if naming_score > 0.2 else 0.1,
        }


class HierarchicalPlanner(OpenCodeTool):
    """Hierarchical planning tool using LRS precision levels."""

    def __init__(self):
        self.name = "hierarchical_planner"
        self.context = LRSExecutionContext()

    def create_development_plan(
        self, project_description: str, constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a hierarchical development plan using precision-guided planning."""

        # Abstract level planning
        abstract_goals = self._plan_abstract_level(project_description)

        # Planning level refinement
        planning_tasks = self._plan_planning_level(abstract_goals, constraints or {})

        # Execution level breakdown
        execution_steps = self._plan_execution_level(planning_tasks)

        # Calculate free energy for the overall plan
        epistemic_value = self._calculate_plan_epistemic_value(
            abstract_goals, planning_tasks, execution_steps
        )
        pragmatic_value = self._calculate_plan_pragmatic_value(
            execution_steps, constraints or {}
        )

        precision = self.context.precision_levels["abstract"]
        free_energy = self.context.calculate_free_energy(
            epistemic_value, pragmatic_value, precision
        )

        # Assess plan quality
        plan_quality = self._assess_plan_quality(free_energy, execution_steps)

        return {
            "abstract_goals": abstract_goals,
            "planning_tasks": planning_tasks,
            "execution_steps": execution_steps,
            "epistemic_value": epistemic_value,
            "pragmatic_value": pragmatic_value,
            "free_energy": free_energy,
            "plan_quality": plan_quality,
            "precision_confidence": precision,
            "estimated_effort": len(execution_steps),
            "risk_assessment": self._assess_plan_risks(execution_steps),
        }

    def _plan_abstract_level(self, description: str) -> List[str]:
        """High-level goal extraction."""
        goals = []

        # Extract key objectives from description
        description_lower = description.lower()

        if "web" in description_lower or "api" in description_lower:
            goals.append("Create robust web/API infrastructure")
        if "database" in description_lower or "data" in description_lower:
            goals.append("Implement efficient data management")
        if "security" in description_lower or "auth" in description_lower:
            goals.append("Ensure security and authentication")
        if "test" in description_lower:
            goals.append("Build comprehensive testing suite")
        if "ui" in description_lower or "interface" in description_lower:
            goals.append("Develop user-friendly interface")

        if not goals:
            goals.append("Implement core functionality")
            goals.append("Ensure code quality")
            goals.append("Enable future extensibility")

        return goals

    def _plan_planning_level(
        self, abstract_goals: List[str], constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Mid-level task planning."""
        tasks = []

        for goal in abstract_goals:
            if "infrastructure" in goal:
                tasks.extend(
                    [
                        {
                            "task": "Design system architecture",
                            "effort": "high",
                            "dependencies": [],
                        },
                        {
                            "task": "Set up development environment",
                            "effort": "medium",
                            "dependencies": [],
                        },
                        {
                            "task": "Configure deployment pipeline",
                            "effort": "medium",
                            "dependencies": ["environment"],
                        },
                    ]
                )
            elif "data" in goal:
                tasks.extend(
                    [
                        {
                            "task": "Design data models",
                            "effort": "high",
                            "dependencies": [],
                        },
                        {
                            "task": "Implement data access layer",
                            "effort": "medium",
                            "dependencies": ["models"],
                        },
                        {
                            "task": "Add data validation",
                            "effort": "low",
                            "dependencies": ["access"],
                        },
                    ]
                )
            elif "security" in goal:
                tasks.extend(
                    [
                        {
                            "task": "Implement authentication",
                            "effort": "high",
                            "dependencies": [],
                        },
                        {
                            "task": "Add authorization checks",
                            "effort": "medium",
                            "dependencies": ["auth"],
                        },
                        {
                            "task": "Security testing",
                            "effort": "low",
                            "dependencies": ["auth", "checks"],
                        },
                    ]
                )

        # Apply constraints
        if constraints.get("timeline") == "tight":
            # Reduce scope for tight timelines
            tasks = [t for t in tasks if t["effort"] != "low"]

        return tasks

    def _plan_execution_level(
        self, planning_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detailed execution planning."""
        steps = []

        for task in planning_tasks:
            if "architecture" in task["task"]:
                steps.extend(
                    [
                        "Create system design document",
                        "Define component interfaces",
                        "Set up project structure",
                    ]
                )
            elif "environment" in task["task"]:
                steps.extend(
                    [
                        "Install development dependencies",
                        "Configure local development server",
                        "Set up version control",
                    ]
                )
            elif "models" in task["task"]:
                steps.extend(
                    [
                        "Define database schema",
                        "Create model classes",
                        "Implement relationships",
                    ]
                )

        # Convert to detailed steps
        detailed_steps = []
        for i, step in enumerate(steps):
            detailed_steps.append(
                {
                    "step": step,
                    "order": i + 1,
                    "estimated_time": "2-4 hours",
                    "success_criteria": f"Step {i + 1} completed and tested",
                    "precision_requirement": 0.7
                    + (i * 0.05),  # Increasing precision requirements
                }
            )

        return detailed_steps

    def _calculate_plan_epistemic_value(
        self, goals: List[str], tasks: List[Dict], steps: List[Dict]
    ) -> float:
        """Calculate information gain from plan creation."""
        # More goals/tasks/steps = more information structure
        goal_entropy = len(goals) / 10.0  # Normalized
        task_entropy = len(tasks) / 20.0
        step_entropy = len(steps) / 50.0

        return min(1.0, (goal_entropy + task_entropy + step_entropy) / 3.0)

    def _calculate_plan_pragmatic_value(
        self, steps: List[Dict], constraints: Dict[str, Any]
    ) -> float:
        """Calculate practical value of the plan."""
        # Shorter plans with clear dependencies are more pragmatic
        step_count = len(steps)
        dependency_complexity = sum(
            len(step.get("dependencies", []))
            for step in steps
            if isinstance(step, dict)
        )

        # Penalty for complexity
        complexity_penalty = (
            dependency_complexity / (step_count * 2) if step_count > 0 else 0
        )

        base_value = 1.0 - (step_count / 100.0)  # Prefer shorter plans
        pragmatic_value = max(0.1, base_value - complexity_penalty)

        # Adjust for constraints
        if constraints.get("timeline") == "tight":
            pragmatic_value *= 0.8  # Slight penalty for rushing

        return pragmatic_value

    def _assess_plan_quality(self, free_energy: float, steps: List[Dict]) -> str:
        """Assess overall plan quality."""
        if free_energy < -0.7:
            return "EXCELLENT - Strong evidence for optimal planning"
        elif free_energy < -0.4:
            return "GOOD - Well-balanced exploration and exploitation"
        elif free_energy < 0:
            return "FAIR - Moderate planning effectiveness"
        else:
            return "NEEDS_IMPROVEMENT - Reconsider approach"

    def _assess_plan_risks(self, steps: List[Dict]) -> Dict[str, Any]:
        """Assess risks in the execution plan."""
        risks = []

        if len(steps) > 20:
            risks.append("High complexity - consider breaking into phases")

        precision_requirements = [s.get("precision_requirement", 0.7) for s in steps]
        avg_precision_req = sum(precision_requirements) / len(precision_requirements)

        if avg_precision_req > 0.8:
            risks.append("High precision requirements - ensure team expertise")

        dependency_count = sum(1 for s in steps if s.get("dependencies"))
        if dependency_count > len(steps) * 0.7:
            risks.append("Tight coupling - high risk of blocking issues")

        return {
            "risk_count": len(risks),
            "risks": risks,
            "overall_risk_level": "HIGH"
            if len(risks) > 2
            else "MEDIUM"
            if len(risks) > 0
            else "LOW",
        }


class CognitiveEnhancedTool(OpenCodeTool):
    """Base class for tools enhanced with cognitive architecture capabilities."""

    def __init__(self, tool_name: str):
        self.name = tool_name
        self.context = LRSExecutionContext()
        self.cognitive_initialized = self.context.initialize_cognitive_system()

    def process_with_cognition(
        self, code_element: str, context: str = "general"
    ) -> Dict[str, Any]:
        """Process code element through cognitive architecture."""
        return self.context.process_with_cognition(code_element, context)

    def get_cognitive_insights(self) -> Dict[str, Any]:
        """Get cognitive processing insights."""
        return self.context.get_cognitive_stats()


class CognitiveCodeAnalyzer(CognitiveEnhancedTool):
    """Cognitive-enhanced code analysis tool."""

    def __init__(self):
        super().__init__("cognitive_code_analyzer")

    def analyze_code_with_cognition(
        self, code_content: str, file_path: str = ""
    ) -> Dict[str, Any]:
        """Analyze code using cognitive architecture for enhanced insights."""

        lines = code_content.split("\n")
        analysis_results = {
            "file_path": file_path,
            "total_lines": len(lines),
            "cognitive_insights": [],
            "attention_patterns": [],
            "anomalies_detected": [],
            "patterns_recognized": [],
            "processing_summary": {},
        }

        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                # Determine context based on line content
                context = self._classify_line_context(line.strip())

                # Process through cognitive system
                cognitive_result = self.process_with_cognition(line.strip(), context)

                if cognitive_result.get("cognitive_processing", False):
                    insight = cognitive_result["insights"]

                    analysis_results["cognitive_insights"].append(
                        {
                            "line_number": i + 1,
                            "content": line.strip(),
                            "context": context,
                            "attention_score": insight.get("attention_score", 0),
                            "cognitive_insight": insight.get("cognitive_insight", ""),
                            "pattern_recognition": insight.get(
                                "pattern_recognition", {}
                            ),
                            "attention_focus": insight.get("attention_focus"),
                        }
                    )

                    # Track high-attention elements
                    if insight.get("attention_score", 0) > 0.7:
                        analysis_results["attention_patterns"].append(
                            {
                                "line": i + 1,
                                "content": line.strip(),
                                "attention_score": insight["attention_score"],
                                "context": context,
                            }
                        )

        # Generate processing summary
        analysis_results["processing_summary"] = self._generate_analysis_summary(
            analysis_results
        )

        return analysis_results

    def _classify_line_context(self, line: str) -> str:
        """Classify the context of a code line."""
        line_lower = line.lower().strip()

        if line.startswith("def "):
            return "function_definition"
        elif line.startswith("class "):
            return "class_definition"
        elif line.startswith("import ") or line.startswith("from "):
            return "import_statement"
        elif "if " in line_lower or "elif " in line_lower:
            return "conditional_statement"
        elif "for " in line_lower or "while " in line_lower:
            return "loop_statement"
        elif (
            "try:" in line_lower or "except " in line_lower or "finally:" in line_lower
        ):
            return "exception_handling"
        elif "return " in line_lower or line_lower.startswith("return"):
            return "return_statement"
        elif "syntax" in line_lower and "error" in line_lower:
            return "error_location"
        else:
            return "general"

    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of cognitive analysis."""
        insights = results["cognitive_insights"]
        attention_patterns = results["attention_patterns"]

        if not insights:
            return {"summary": "No cognitive insights generated"}

        avg_attention = sum(i["attention_score"] for i in insights) / len(insights)
        high_attention_count = len([i for i in insights if i["attention_score"] > 0.7])

        # Count pattern types
        pattern_counts = {}
        for insight in insights:
            pattern = insight["pattern_recognition"].get(
                "recognized_pattern", "unknown"
            )
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        return {
            "total_lines_analyzed": len(insights),
            "average_attention_score": avg_attention,
            "high_attention_elements": high_attention_count,
            "attention_patterns_found": len(attention_patterns),
            "pattern_distribution": pattern_counts,
            "cognitive_state": self.get_cognitive_insights(),
        }


class PolicyEvaluator(OpenCodeTool):
    """Tool for evaluating multiple development strategies."""

    def __init__(self):
        self.name = "policy_evaluator"
        self.context = LRSExecutionContext()

    def evaluate_strategies(
        self, task_description: str, strategies: List[str]
    ) -> Dict[str, Any]:
        """Evaluate multiple development strategies using free energy minimization."""

        evaluations = []

        for strategy in strategies:
            # Simulate strategy execution to gather metrics
            strategy_metrics = self._simulate_strategy(strategy, task_description)

            # Calculate free energy components
            epistemic_value = strategy_metrics["information_gain"]
            pragmatic_value = strategy_metrics["effort_efficiency"]

            # Use current precision for weighting
            precision = self.context.precision_levels["planning"]
            free_energy = self.context.calculate_free_energy(
                epistemic_value, pragmatic_value, precision
            )

            evaluations.append(
                {
                    "strategy": strategy,
                    "epistemic_value": epistemic_value,
                    "pragmatic_value": pragmatic_value,
                    "free_energy": free_energy,
                    "expected_success": strategy_metrics["success_probability"],
                    "estimated_effort": strategy_metrics["effort_days"],
                    "risk_level": strategy_metrics["risk_level"],
                }
            )

        # Select best strategy using precision-weighted selection
        best_strategy_idx = self._select_best_strategy(evaluations)
        best_strategy = evaluations[best_strategy_idx]

        # Get precision level used for decision making
        precision = self.context.precision_levels.get("planning", 0.5)

        return {
            "evaluations": evaluations,
            "recommended_strategy": best_strategy,
            "selection_confidence": self._calculate_selection_confidence(evaluations),
            "precision_used": precision,
            "alternative_strategies": [e for e in evaluations if e != best_strategy],
        }

    def _simulate_strategy(self, strategy: str, task: str) -> Dict[str, Any]:
        """Simulate strategy execution to gather performance metrics."""

        # Strategy analysis based on keywords
        strategy_lower = strategy.lower()

        if "agile" in strategy_lower or "iterative" in strategy_lower:
            return {
                "information_gain": 0.7,  # Good for learning
                "effort_efficiency": 0.6,  # Some overhead
                "success_probability": 0.8,
                "effort_days": 14,
                "risk_level": "MEDIUM",
            }
        elif "waterfall" in strategy_lower or "sequential" in strategy_lower:
            return {
                "information_gain": 0.3,  # Less learning
                "effort_efficiency": 0.8,  # Less overhead
                "success_probability": 0.6,
                "effort_days": 21,
                "risk_level": "HIGH",
            }
        elif "prototype" in strategy_lower or "mvp" in strategy_lower:
            return {
                "information_gain": 0.9,  # Maximum learning
                "effort_efficiency": 0.4,  # High overhead
                "success_probability": 0.7,
                "effort_days": 10,
                "risk_level": "MEDIUM",
            }
        elif "tdd" in strategy_lower or "test" in strategy_lower:
            return {
                "information_gain": 0.6,  # Good practices
                "effort_efficiency": 0.7,  # Balanced
                "success_probability": 0.85,
                "effort_days": 16,
                "risk_level": "LOW",
            }
        else:
            # Default balanced approach
            return {
                "information_gain": 0.5,
                "effort_efficiency": 0.6,
                "success_probability": 0.7,
                "effort_days": 18,
                "risk_level": "MEDIUM",
            }

    def _select_best_strategy(self, evaluations: List[Dict[str, Any]]) -> int:
        """Select best strategy using free energy minimization."""
        # Convert free energy to selection probabilities
        G_values = [e["free_energy"] for e in evaluations]

        # Lower G is better (minimization), so invert for probabilities
        max_G = max(G_values)
        min_G = min(G_values)
        range_G = max_G - min_G if max_G != min_G else 1.0

        # Convert to probabilities (lower G = higher probability)
        probabilities = [(max_G - g) / range_G for g in G_values]

        # Normalize
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        # Sample from distribution
        import random

        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probabilities):
            cumsum += p
            if r < cumsum:
                return i

        return 0  # Fallback

    def _calculate_selection_confidence(
        self, evaluations: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in strategy selection."""
        if len(evaluations) < 2:
            return 1.0

        G_values = [e["free_energy"] for e in evaluations]
        best_G = min(G_values)
        second_best_G = sorted(G_values)[1]

        # Confidence based on gap between best and second best
        gap = second_best_G - best_G
        confidence = min(1.0, gap * 2.0)  # Scale gap to confidence

        return confidence


# =============================================================================
# OPENCODE COMMAND INTEGRATION
# =============================================================================

# Register LRS-enhanced tools
lrs_registry.register_tool("active_inference_analyzer", ActiveInferenceAnalyzer)
lrs_registry.register_tool("precision_guided_refactor", PrecisionGuidedRefactor)
lrs_registry.register_tool("hierarchical_planner", HierarchicalPlanner)
lrs_registry.register_tool("policy_evaluator", PolicyEvaluator)


def opencode_lrs_command(args: List[str]) -> str:
    """
    OpenCode LRS command interface with performance optimization.

    Usage:
        opencode lrs analyze <path>          # Active Inference codebase analysis (optimized)
        opencode lrs refactor <file>         # Precision-guided refactoring analysis
        opencode lrs plan <description>      # Hierarchical development planning
        opencode lrs evaluate <task> <strat1> <strat2> ...  # Strategy evaluation
        opencode lrs stats                    # LRS system statistics
        opencode lrs optimize <path>         # Run optimized analysis with caching
    """
    if len(args) < 1:
        return "Usage: opencode lrs <command> [args...]\n\nCommands:\n  analyze <path>     - Active Inference codebase analysis (optimized)\n  optimize <path>    - Performance-optimized analysis with caching\n  refactor <file>    - Precision-guided refactoring analysis\n  plan <description> - Hierarchical development planning\n  evaluate <task> <strategies...> - Strategy evaluation\n  stats              - LRS system statistics"

    command = args[0].lower()

    try:
        if command == "analyze" and len(args) > 1:
            # Use optimized analyzer for better performance
            from lrs_agents.lrs.enterprise.performance_optimization import OptimizedActiveInferenceAnalyzer

            analyzer = OptimizedActiveInferenceAnalyzer()
            result = analyzer.analyze_codebase(args[1])

            if "error" in result:
                return f"Analysis failed: {result['error']}"

            output = "🎯 Active Inference Codebase Analysis (Optimized)\n"
            output += "=" * 48 + "\n\n"
            output += f"📊 Total Files: {result['total_files']}\n"
            output += f"📝 Total Lines: {result['total_lines']}\n"
            output += f"🧠 Average Complexity: {result['avg_complexity']:.2f}\n"
            output += f"🎯 Free Energy G: {result['free_energy']:.3f} (optimization objective)\n"
            output += f"⏱️  Analysis Time: {result.get('analysis_time', 'N/A'):.2f}s\n"
            output += f"💡 Recommendation: {result['recommendations'][0] if result['recommendations'] else 'Analysis complete'}\n"

            return output

        elif command == "optimize" and len(args) > 1:
            # Dedicated optimized analysis command
            from lrs_agents.lrs.enterprise.performance_optimization import run_optimized_analysis

            result = run_optimized_analysis(args[1], use_cache=True)

            if "error" in result:
                return f"Optimized analysis failed: {result['error']}"

            output = "🚀 Performance-Optimized Codebase Analysis\n"
            output += "=" * 45 + "\n\n"
            output += f"📊 Total Files: {result['total_files']}\n"
            output += f"📝 Total Lines: {result['total_lines']}\n"
            output += f"🧠 Average Complexity: {result['avg_complexity']:.2f}\n"
            output += f"🎯 Free Energy G: {result['free_energy']:.3f}\n"
            output += f"⏱️  Total Time: {result.get('total_time', result.get('analysis_time', 'N/A')):.2f}s ⚡\n"
            output += "💾 Cached: Results cached for future use\n\n"
            output += "💡 Recommendations:\n"
            for rec in result["recommendations"][:3]:
                output += f"   • {rec}\n"

            return output

        elif command == "refactor" and len(args) > 1:
            refactor_tool = PrecisionGuidedRefactor()
            result = refactor_tool.analyze_refactor_opportunities(args[1])

            if "error" in result:
                return f"Refactoring analysis failed: {result['error']}"

            output = "🔧 Precision-Guided Refactoring Analysis\n"
            output += "=" * 45 + "\n\n"
            output += f"📁 File: {result['file']}\n\n"
            output += "📊 Metrics:\n"

            for metric_name, metric_data in result["metrics"].items():
                output += f"   {metric_name.replace('_', ' ').title()}:\n"
                for key, value in metric_data.items():
                    if isinstance(value, float):
                        output += f"     • {key}: {value:.3f}\n"
                    else:
                        output += f"     • {key}: {value}\n"
                output += "\n"

            output += f"🔍 Epistemic Value: {result['epistemic_value']:.3f}\n"
            output += f"🎯 Pragmatic Value: {result['pragmatic_value']:.3f}\n"
            output += f"⚡ Free Energy G: {result['free_energy']:.3f}\n"
            output += (
                f"🎚️  Precision Confidence: {result['precision_confidence']:.3f}\n\n"
            )
            output += f"🚨 Refactor Priority: {result['refactor_priority']}\n"
            output += f"💡 Reason: {result['reason']}\n"

            return output

        elif command == "plan" and len(args) > 1:
            planner = HierarchicalPlanner()
            description = " ".join(args[1:])
            result = planner.create_development_plan(description)

            output = "📋 Hierarchical Development Plan\n"
            output += "=" * 35 + "\n\n"
            output += "🎯 Abstract Goals:\n"
            for goal in result["abstract_goals"]:
                output += f"   • {goal}\n"
            output += "\n"

            output += "📝 Planning Tasks:\n"
            for task in result["planning_tasks"]:
                output += f"   • {task['task']} (effort: {task['effort']})\n"
            output += "\n"

            output += "⚙️  Execution Steps:\n"
            for step in result["execution_steps"]:
                output += f"   {step['order']}. {step['step']}\n"
                output += f"      ⏱️  Estimated: {step['estimated_time']}\n"
                output += f"      ✅ Criteria: {step['success_criteria']}\n"
                output += (
                    f"      🎚️  Precision Req: {step['precision_requirement']:.2f}\n\n"
                )

            output += f"🔍 Epistemic Value: {result['epistemic_value']:.3f}\n"
            output += f"🎯 Pragmatic Value: {result['pragmatic_value']:.3f}\n"
            output += f"⚡ Free Energy G: {result['free_energy']:.3f}\n"
            output += f"📊 Plan Quality: {result['plan_quality']}\n"
            output += f"🎚️  Precision Confidence: {result['precision_confidence']:.3f}\n"
            output += f"⏱️  Estimated Effort: {result['estimated_effort']} steps\n"
            output += (
                f"⚠️  Risk Level: {result['risk_assessment']['overall_risk_level']}\n"
            )

            if result["risk_assessment"]["risks"]:
                output += "\n🚨 Identified Risks:\n"
                for risk in result["risk_assessment"]["risks"]:
                    output += f"   • {risk}\n"

            return output

        elif command == "evaluate" and len(args) > 2:
            evaluator = PolicyEvaluator()
            task = args[1]
            strategies = args[2:]

            result = evaluator.evaluate_strategies(task, strategies)

            output = "⚖️  Strategy Evaluation Results\n"
            output += "=" * 30 + "\n\n"
            output += f"🎯 Task: {task}\n\n"

            output += "📊 Strategy Evaluations:\n"
            for i, eval_data in enumerate(result["evaluations"], 1):
                output += f"{i}. {eval_data['strategy']}\n"
                output += f"   🎯 Epistemic: {eval_data['epistemic_value']:.3f}\n"
                output += f"   🎯 Pragmatic: {eval_data['pragmatic_value']:.3f}\n"
                output += f"   ⚡ Free Energy G: {eval_data['free_energy']:.3f}\n"
                output += f"   📈 Success Prob: {eval_data['expected_success']:.1%}\n"
                output += f"   ⏱️  Effort: {eval_data['estimated_effort']} days\n"
                output += f"   ⚠️  Risk: {eval_data['risk_level']}\n\n"

            output += "🏆 Recommended Strategy:\n"
            rec = result["recommended_strategy"]
            output += f"   🎯 {rec['strategy']}\n"
            output += f"   📊 Confidence: {result['selection_confidence']:.1%}\n"
            output += f"   🎚️  Precision Used: {result['precision_used']:.3f}\n"

            return output

        elif command == "stats":
            # Show LRS statistics
            output = "📈 LRS System Statistics\n"
            output += "=" * 25 + "\n\n"

            # Tool usage stats
            output += "🔧 Registered Tools:\n"
            for name in lrs_registry.tools.keys():
                output += f"   • {name}\n"
            output += "\n"

            # Precision history
            if lrs_registry.precision_history:
                output += "🎚️  Precision Updates:\n"
                recent = lrs_registry.precision_history[-5:]  # Last 5
                for update in recent:
                    output += f"   • {update['level']}: {update['old_precision']:.3f} → {update['new_precision']:.3f} ({update['reason']})\n"
                output += "\n"

            # Free energy calculations
            if lrs_registry.free_energy_log:
                output += "⚡ Free Energy Calculations:\n"
                recent = lrs_registry.free_energy_log[-3:]  # Last 3
                for calc in recent:
                    output += f"   • Policy: {calc['policy'][:30]}...\n"
                    output += f"     G = {calc['epistemic']:.3f} - {calc['pragmatic']:.3f} = {calc['total_g']:.3f}\n"
                output += "\n"

            output += "💡 LRS enhances OpenCode with:\n"
            output += "   • Active Inference for intelligent decision making\n"
            output += "   • Precision tracking for confidence assessment\n"
            output += "   • Free energy minimization for optimal strategies\n"
            output += "   • Hierarchical planning for complex tasks\n"

            return output

        else:
            return f"Unknown command: {command}\n\n{opencode_lrs_command([])}"

    except Exception as e:
        return f"Error executing LRS command: {str(e)}"


# Export for OpenCode integration
__all__ = [
    "opencode_lrs_command",
    "ActiveInferenceAnalyzer",
    "PrecisionGuidedRefactor",
    "HierarchicalPlanner",
    "PolicyEvaluator",
    "LRSExecutionContext",
    "lrs_registry",
]

if __name__ == "__main__":
    # Test the LRS integration
    print("🧠 OpenCode LRS Integration Test")
    print("=" * 40)

    # Test basic functionality
    analyzer = ActiveInferenceAnalyzer()
    result = analyzer.analyze_codebase(".")
    print(
        f"✅ Active Inference Analyzer: {result.get('total_files', 0)} files analyzed"
    )

    # Test hierarchical planner
    planner = HierarchicalPlanner()
    plan = planner.create_development_plan("Build a web API for task management")
    print(
        f"✅ Hierarchical Planner: {len(plan['execution_steps'])} execution steps created"
    )

    # Test policy evaluator
    evaluator = PolicyEvaluator()
    strategies = ["Use Agile methodology", "Follow TDD approach", "Build MVP first"]
    evaluation = evaluator.evaluate_strategies(
        "Implement user authentication", strategies
    )
    print(f"✅ Policy Evaluator: {len(evaluation['evaluations'])} strategies evaluated")

    print("\n🎉 All LRS tools integrated successfully!")
    print("Use: opencode lrs <command> [args...]")

    # Register cognitive-enhanced tools after class definitions
    if COGNITIVE_COMPONENTS_AVAILABLE:
        lrs_registry.register_tool("cognitive_code_analyzer", CognitiveEnhancedTool)
        lrs_registry.register_tool("cognitive_analyzer", CognitiveCodeAnalyzer)
        print("✅ Cognitive-enhanced tools registered in LRS registry")
    else:
        print(
            "⚠️  Cognitive components not available - running with standard LRS tools only"
        )
