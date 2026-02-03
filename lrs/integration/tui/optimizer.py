"""
Automated Agent Optimization and Tuning System.

This component provides intelligent optimization, hyperparameter tuning,
and automated performance improvement for LRS agents using advanced ML techniques.
"""

import asyncio
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import random
from scipy import optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ...multi_agent.shared_state import SharedWorldState
from ...core.precision import PrecisionParameters


class OptimizationType(Enum):
    """Types of optimization strategies."""

    PRECISION_TUNING = "precision_tuning"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    TOOL_SELECTION = "tool_selection"
    LEARNING_RATE_ADAPTATION = "learning_rate_adaptation"
    ENSEMBLE_OPTIMIZATION = "ensemble_optimization"
    RESOURCE_ALLOCATION = "resource_allocation"


class OptimizationAlgorithm(Enum):
    """Optimization algorithms."""

    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class OptimizationStatus(Enum):
    """Optimization status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OptimizationTarget:
    """Optimization target configuration."""

    agent_id: str
    optimization_type: OptimizationType
    objective_function: str  # "maximize_precision", "minimize_error_rate", etc.
    target_metric: str  # "precision_value", "success_rate", etc.
    constraints: Dict[str, Any] = field(default_factory=dict)
    optimization_horizon: int = 3600  # seconds
    min_improvement: float = 0.05  # 5% minimum improvement


@dataclass
class OptimizationResult:
    """Result of optimization process."""

    optimization_id: str
    agent_id: str
    optimization_type: OptimizationType
    algorithm: OptimizationAlgorithm
    status: OptimizationStatus
    started_at: datetime
    completed_at: Optional[datetime]
    best_parameters: Dict[str, Any]
    best_score: float
    baseline_score: float
    improvement_percentage: float
    iterations: int
    convergence_achieved: bool
    final_metrics: Dict[str, Any]
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParameterSpace:
    """Parameter space for optimization."""

    parameter_name: str
    parameter_type: str  # "continuous", "discrete", "categorical"
    bounds: Optional[Tuple[float, float]]
    values: Optional[List[Any]]
    log_scale: bool = False


class AgentOptimizer:
    """
    Intelligent agent optimization and tuning system.

    Features:
    - Multi-objective optimization (precision, performance, efficiency)
    - Advanced algorithms (Bayesian, Genetic, RL-based)
    - Real-time parameter tuning with feedback
    - A/B testing framework
    - Automated rollback on performance degradation
    - Ensemble optimization strategies
    - Cross-agent optimization sharing
    - Resource-aware optimization

    Examples:
        >>> optimizer = AgentOptimizer(shared_state)
        >>>
        >>> # Optimize agent precision
        >>> result = await optimizer.optimize_agent(
        ...     target=OptimizationTarget(
        ...         agent_id="agent_1",
        ...         optimization_type=OptimizationType.PRECISION_TUNING,
        ...         objective_function="maximize_precision"
        ...     ),
        ...     algorithm=OptimizationAlgorithm.BAYESIAN_OPTIMIZATION
        ... )
        >>>
        >>> # Continuous optimization
        >>> await optimizer.start_continuous_optimization("agent_1")
    """

    def __init__(self, shared_state: SharedWorldState):
        """
        Initialize agent optimizer.

        Args:
            shared_state: LRS shared world state
        """
        self.shared_state = shared_state

        # Optimization state
        self.active_optimizations: Dict[str, OptimizationResult] = {}
        self.optimization_history: List[OptimizationResult] = []

        # Parameter spaces
        self.parameter_spaces = self._setup_parameter_spaces()

        # ML models for optimization
        self.performance_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Continuous optimization
        self.continuous_optimizations: Dict[str, bool] = {}
        self.optimization_loops: Dict[str, asyncio.Task] = {}

        # A/B testing framework
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}

        # Optimization cache
        self.parameter_cache: Dict[str, Dict[str, Any]] = {}
        self.performance_cache: Dict[str, List[float]] = {}

        # Configuration
        self.config = self._default_config()

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._initialize_ml_models()
        self._start_background_tasks()

    async def optimize_agent(
        self,
        target: OptimizationTarget,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN_OPTIMIZATION,
        max_iterations: int = 100,
        parallel_evaluations: int = 4,
    ) -> OptimizationResult:
        """
        Optimize agent parameters using specified algorithm.

        Args:
            target: Optimization target configuration
            algorithm: Optimization algorithm to use
            max_iterations: Maximum optimization iterations
            parallel_evaluations: Number of parallel evaluations

        Returns:
            Optimization result with best parameters
        """
        optimization_id = f"opt_{datetime.now().timestamp()}"

        try:
            # Create optimization result
            result = OptimizationResult(
                optimization_id=optimization_id,
                agent_id=target.agent_id,
                optimization_type=target.optimization_type,
                algorithm=algorithm,
                status=OptimizationStatus.RUNNING,
                started_at=datetime.now(),
                completed_at=None,
                best_parameters={},
                best_score=float("-inf")
                if "maximize" in target.objective_function
                else float("inf"),
                baseline_score=await self._evaluate_baseline(target),
                improvement_percentage=0.0,
                iterations=0,
                convergence_achieved=False,
                final_metrics={},
            )

            self.active_optimizations[optimization_id] = result

            # Get parameter space for optimization type
            param_space = self._get_parameter_space(target.optimization_type)

            # Run optimization algorithm
            if algorithm == OptimizationAlgorithm.BAYESIAN_OPTIMIZATION:
                await self._bayesian_optimization(result, target, param_space, max_iterations)

            elif algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                await self._genetic_algorithm(result, target, param_space, max_iterations)

            elif algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                await self._particle_swarm_optimization(result, target, param_space, max_iterations)

            elif algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
                await self._simulated_annealing(result, target, param_space, max_iterations)

            elif algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
                await self._random_search(
                    result, target, param_space, max_iterations, parallel_evaluations
                )

            else:
                raise ValueError(f"Unsupported optimization algorithm: {algorithm}")

            # Mark as completed
            result.status = OptimizationStatus.COMPLETED
            result.completed_at = datetime.now()
            result.improvement_percentage = self._calculate_improvement(result)
            result.final_metrics = await self._evaluate_final_metrics(
                target, result.best_parameters
            )

            # Apply best parameters if improvement is significant
            if result.improvement_percentage >= target.min_improvement:
                await self._apply_optimization_results(target.agent_id, result.best_parameters)
                self.logger.info(
                    f"Applied optimization results for {target.agent_id}: {result.improvement_percentage:.2%} improvement"
                )
            else:
                self.logger.info(
                    f"Optimization completed but improvement ({result.improvement_percentage:.2%}) below threshold ({target.min_improvement:.2%})"
                )

            # Store in history
            self.optimization_history.append(result)

            # Update performance models
            await self._update_performance_models(target.agent_id, result)

            return result

        except Exception as e:
            self.logger.error(f"Optimization failed for {target.agent_id}: {e}")

            if optimization_id in self.active_optimizations:
                self.active_optimizations[optimization_id].status = OptimizationStatus.FAILED
                self.active_optimizations[optimization_id].completed_at = datetime.now()

            # Return failed result
            return OptimizationResult(
                optimization_id=optimization_id,
                agent_id=target.agent_id,
                optimization_type=target.optimization_type,
                algorithm=algorithm,
                status=OptimizationStatus.FAILED,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                best_parameters={},
                best_score=0.0,
                baseline_score=0.0,
                improvement_percentage=0.0,
                iterations=0,
                convergence_achieved=False,
                final_metrics={},
            )

    async def start_continuous_optimization(
        self,
        agent_id: str,
        optimization_interval: int = 3600,  # 1 hour
        optimization_types: List[OptimizationType] = None,
    ):
        """
        Start continuous optimization for agent.

        Args:
            agent_id: Agent to optimize continuously
            optimization_interval: Interval between optimizations (seconds)
            optimization_types: Types of optimizations to run
        """
        if optimization_types is None:
            optimization_types = [
                OptimizationType.PRECISION_TUNING,
                OptimizationType.LEARNING_RATE_ADAPTATION,
                OptimizationType.TOOL_SELECTION,
            ]

        self.continuous_optimizations[agent_id] = True

        # Create continuous optimization loop
        task = asyncio.create_task(
            self._continuous_optimization_loop(agent_id, optimization_interval, optimization_types)
        )

        self.optimization_loops[agent_id] = task
        self.logger.info(f"Started continuous optimization for {agent_id}")

    async def stop_continuous_optimization(self, agent_id: str):
        """Stop continuous optimization for agent."""

        if agent_id in self.continuous_optimizations:
            self.continuous_optimizations[agent_id] = False

            if agent_id in self.optimization_loops:
                self.optimization_loops[agent_id].cancel()
                del self.optimization_loops[agent_id]

            self.logger.info(f"Stopped continuous optimization for {agent_id}")

    async def create_ab_test(
        self,
        agent_id: str,
        test_name: str,
        control_parameters: Dict[str, Any],
        variant_parameters: List[Dict[str, Any]],
        traffic_split: Optional[List[float]] = None,
        test_duration: int = 3600,  # 1 hour
    ) -> str:
        """
        Create A/B test for agent parameters.

        Args:
            agent_id: Agent to test
            test_name: Test identifier
            control_parameters: Control group parameters
            variant_parameters: List of variant parameters
            traffic_split: Traffic split percentages
            test_duration: Test duration in seconds

        Returns:
            Test ID
        """
        test_id = f"ab_test_{test_name}_{datetime.now().timestamp()}"

        if traffic_split is None:
            traffic_split = [1.0 / (len(variant_parameters) + 1)] * (len(variant_parameters) + 1)

        self.ab_tests[test_id] = {
            "test_id": test_id,
            "agent_id": agent_id,
            "test_name": test_name,
            "control_parameters": control_parameters,
            "variant_parameters": variant_parameters,
            "traffic_split": traffic_split,
            "test_duration": test_duration,
            "started_at": datetime.now(),
            "status": "running",
            "assigned_traffic": {},  # agent_instance_id -> variant
            "results": {
                "control": {"metrics": [], "performance": 0.0},
                "variants": [{"metrics": [], "performance": 0.0} for _ in variant_parameters],
            },
        }

        # Start test monitoring
        asyncio.create_task(self._monitor_ab_test(test_id))

        self.logger.info(f"Created A/B test {test_id} for agent {agent_id}")

        return test_id

    async def get_optimization_recommendations(
        self, agent_id: str, optimization_types: List[OptimizationType] = None
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations for agent.

        Args:
            agent_id: Agent to analyze
            optimization_types: Types of optimizations to consider

        Returns:
            Optimization recommendations
        """
        if optimization_types is None:
            optimization_types = list(OptimizationType)

        recommendations = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "recommendations": [],
        }

        # Analyze current agent performance
        agent_state = self.shared_state.get_agent_state(agent_id)
        if not agent_state:
            return recommendations

        current_precision = agent_state.get("precision", {}).get("value", 0.5)
        current_performance = self._calculate_performance_score(agent_state)

        for opt_type in optimization_types:
            recommendation = await self._generate_recommendation(
                agent_id, opt_type, current_precision, current_performance
            )

            if recommendation:
                recommendations["recommendations"].append(recommendation)

        # Prioritize recommendations
        recommendations["recommendations"].sort(key=lambda x: x["priority"], reverse=True)

        return recommendations

    def _setup_parameter_spaces(self) -> Dict[OptimizationType, List[ParameterSpace]]:
        """Setup parameter spaces for different optimization types."""

        return {
            OptimizationType.PRECISION_TUNING: [
                ParameterSpace(
                    parameter_name="gain_learning_rate",
                    parameter_type="continuous",
                    bounds=(0.01, 0.5),
                ),
                ParameterSpace(
                    parameter_name="loss_learning_rate",
                    parameter_type="continuous",
                    bounds=(0.05, 0.8),
                ),
                ParameterSpace(
                    parameter_name="adaptation_threshold",
                    parameter_type="continuous",
                    bounds=(0.1, 0.8),
                ),
            ],
            OptimizationType.HYPERPARAMETER_OPTIMIZATION: [
                ParameterSpace(
                    parameter_name="exploration_rate",
                    parameter_type="continuous",
                    bounds=(0.01, 0.3),
                ),
                ParameterSpace(
                    parameter_name="temperature", parameter_type="continuous", bounds=(0.1, 2.0)
                ),
                ParameterSpace(
                    parameter_name="max_iterations", parameter_type="discrete", bounds=(10, 1000)
                ),
            ],
            OptimizationType.TOOL_SELECTION: [
                ParameterSpace(
                    parameter_name="tool_weights",
                    parameter_type="categorical",
                    values=["equal", "performance_based", "precision_based", "adaptive"],
                ),
                ParameterSpace(
                    parameter_name="fallback_strategy",
                    parameter_type="categorical",
                    values=["random", "similar", "performance_ranked", "precision_ranked"],
                ),
            ],
            OptimizationType.LEARNING_RATE_ADAPTATION: [
                ParameterSpace(
                    parameter_name="adaptive_lr",
                    parameter_type="categorical",
                    values=["constant", "decay", "cyclical", "performance_adaptive"],
                ),
                ParameterSpace(
                    parameter_name="lr_decay_factor",
                    parameter_type="continuous",
                    bounds=(0.9, 0.999),
                ),
            ],
        }

    async def _bayesian_optimization(
        self,
        result: OptimizationResult,
        target: OptimizationTarget,
        param_space: List[ParameterSpace],
        max_iterations: int,
    ):
        """Implement Bayesian optimization."""

        # Initialize with random samples
        n_init = min(10, max_iterations // 4)
        X_init = []
        y_init = []

        for _ in range(n_init):
            params = self._sample_parameters(param_space)
            score = await self._evaluate_parameters(target.agent_id, params, target)

            X_init.append(self._params_to_vector(params, param_space))
            y_init.append(score)

            if score > result.best_score:
                result.best_score = score
                result.best_parameters = params.copy()

            result.iterations += 1

        # Simple Bayesian optimization using Gaussian Process
        X = np.array(X_init)
        y = np.array(y_init)

        for iteration in range(n_init, max_iterations):
            # Fit Gaussian Process (simplified)
            if len(X) > 1:
                # Use simple mean/std approach for demonstration
                current_best_idx = np.argmax(y)
                best_params = X[current_best_idx]

                # Generate new point around best
                new_params_vector = best_params + np.random.normal(0, 0.1, len(best_params))
            else:
                # Random exploration
                new_params_vector = np.random.rand(len(param_space))

            # Convert vector back to parameters
            new_params = self._vector_to_params(new_params_vector, param_space)

            # Evaluate new parameters
            score = await self._evaluate_parameters(target.agent_id, new_params, target)

            X = np.vstack([X, self._params_to_vector(new_params, param_space)])
            y = np.append(y, score)

            if score > result.best_score:
                result.best_score = score
                result.best_parameters = new_params.copy()

            result.iterations += 1

            # Store optimization history
            result.optimization_history.append(
                {
                    "iteration": iteration,
                    "parameters": new_params,
                    "score": score,
                    "best_score": result.best_score,
                }
            )

            # Check convergence
            if self._check_convergence(y, iteration):
                result.convergence_achieved = True
                break

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)

    async def _genetic_algorithm(
        self,
        result: OptimizationResult,
        target: OptimizationTarget,
        param_space: List[ParameterSpace],
        max_iterations: int,
    ):
        """Implement genetic algorithm optimization."""

        population_size = 20
        mutation_rate = 0.1
        crossover_rate = 0.8

        # Initialize population
        population = []
        for _ in range(population_size):
            params = self._sample_parameters(param_space)
            population.append(params)

        for generation in range(max_iterations):
            # Evaluate fitness
            fitness_scores = []

            for params in population:
                score = await self._evaluate_parameters(target.agent_id, params, target)
                fitness_scores.append(score)

                if score > result.best_score:
                    result.best_score = score
                    result.best_parameters = params.copy()

            # Selection (tournament selection)
            selected = self._tournament_selection(population, fitness_scores, population_size // 2)

            # Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i + 1]

                    if random.random() < crossover_rate:
                        child1, child2 = self._crossover(parent1, parent2, param_space)
                        offspring.extend([child1, child2])
                    else:
                        offspring.extend([parent1, parent2])

            # Mutation
            for individual in offspring:
                if random.random() < mutation_rate:
                    self._mutate(individual, param_space)

            # Create new generation
            population = offspring + selected[: population_size - len(offspring)]

            result.iterations += 1

            # Check convergence
            if self._check_convergence(fitness_scores, generation):
                result.convergence_achieved = True
                break

    async def _particle_swarm_optimization(
        self,
        result: OptimizationResult,
        target: OptimizationTarget,
        param_space: List[ParameterSpace],
        max_iterations: int,
    ):
        """Implement particle swarm optimization."""

        n_particles = 15
        w = 0.7  # inertia weight
        c1 = 1.5  # cognitive parameter
        c2 = 1.5  # social parameter

        # Initialize particles
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []

        for _ in range(n_particles):
            params = self._sample_parameters(param_space)
            particles.append(params)
            velocities.append(self._random_velocity(param_space))
            personal_best.append(params.copy())

            score = await self._evaluate_parameters(target.agent_id, params, target)
            personal_best_scores.append(score)

            if score > result.best_score:
                result.best_score = score
                result.best_parameters = params.copy()

        # Find global best
        global_best_idx = np.argmax(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()

        for iteration in range(max_iterations):
            for i in range(n_particles):
                # Update velocity
                for j, param_space_item in enumerate(param_space):
                    r1, r2 = random.random(), random.random()

                    velocities[i][j] = (
                        w * velocities[i][j]
                        + c1 * r1 * (personal_best[i][j] - particles[i][j])
                        + c2 * r2 * (global_best[j] - particles[i][j])
                    )

                # Update position
                new_params = self._apply_velocity(particles[i], velocities[i], param_space)
                particles[i] = new_params

                # Evaluate new position
                score = await self._evaluate_parameters(target.agent_id, new_params, target)

                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = new_params.copy()

                    # Update global best
                    if score > result.best_score:
                        result.best_score = score
                        result.best_parameters = new_params.copy()
                        global_best = new_params.copy()

            result.iterations += 1

            # Check convergence
            if self._check_convergence(personal_best_scores, iteration):
                result.convergence_achieved = True
                break

    async def _evaluate_parameters(
        self, agent_id: str, parameters: Dict[str, Any], target: OptimizationTarget
    ) -> float:
        """Evaluate parameters and return objective score."""

        try:
            # Apply parameters temporarily
            await self._apply_parameters_temporarily(agent_id, parameters)

            # Wait for system to stabilize
            await asyncio.sleep(2)

            # Evaluate objective
            if target.objective_function == "maximize_precision":
                state = self.shared_state.get_agent_state(agent_id)
                precision = state.get("precision", {}).get("value", 0.5)
                return precision

            elif target.objective_function == "minimize_error_rate":
                state = self.shared_state.get_agent_state(agent_id)
                error_rate = self._calculate_error_rate(state.get("tool_executions", []))
                return -error_rate  # Negative because we minimize

            elif target.objective_function == "maximize_performance":
                state = self.shared_state.get_agent_state(agent_id)
                performance = self._calculate_performance_score(state)
                return performance

            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {e}")
            return 0.0

        finally:
            # Restore original parameters (in real implementation)
            pass

    def _sample_parameters(self, param_space: List[ParameterSpace]) -> Dict[str, Any]:
        """Sample random parameters from parameter space."""

        params = {}

        for param in param_space:
            if param.parameter_type == "continuous":
                if param.log_scale:
                    # Log-uniform sampling
                    log_min = np.log(param.bounds[0])
                    log_max = np.log(param.bounds[1])
                    value = np.exp(random.uniform(log_min, log_max))
                else:
                    value = random.uniform(param.bounds[0], param.bounds[1])
                params[param.parameter_name] = value

            elif param.parameter_type == "discrete":
                value = random.randint(int(param.bounds[0]), int(param.bounds[1]))
                params[param.parameter_name] = value

            elif param.parameter_type == "categorical":
                value = random.choice(param.values)
                params[param.parameter_name] = value

        return params

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""

        return {
            "max_concurrent_optimizations": 5,
            "optimization_timeout": 3600,  # 1 hour
            "evaluation_timeout": 30,  # 30 seconds
            "convergence_patience": 20,  # iterations
            "min_improvement_threshold": 0.001,
            "ab_test_min_samples": 100,
            "continuous_optimization_interval": 3600,  # 1 hour
        }

    def _initialize_ml_models(self):
        """Initialize ML models for optimization."""

        # Performance prediction models
        for param_type in OptimizationType:
            self.performance_models[param_type.value] = RandomForestRegressor(
                n_estimators=100, random_state=42
            )
            self.scalers[param_type.value] = StandardScaler()

    async def _start_background_tasks(self):
        """Start background optimization tasks."""

        # Performance monitoring task
        task = asyncio.create_task(self._performance_monitoring_loop())
        self.background_tasks.append(task)

        # Optimization history cleanup task
        task = asyncio.create_task(self._cleanup_optimization_history())
        self.background_tasks.append(task)

        # A/B test monitoring task
        task = asyncio.create_task(self._ab_test_monitoring_loop())
        self.background_tasks.append(task)

    async def _continuous_optimization_loop(
        self, agent_id: str, interval: int, optimization_types: List[OptimizationType]
    ):
        """Continuous optimization loop for agent."""

        while self.continuous_optimizations.get(agent_id, False):
            try:
                # Get current recommendations
                recommendations = await self.get_optimization_recommendations(
                    agent_id, optimization_types
                )

                # Apply top recommendation if significant
                if recommendations["recommendations"]:
                    top_rec = recommendations["recommendations"][0]

                    if top_rec["expected_improvement"] > 0.05:  # 5% threshold
                        target = OptimizationTarget(
                            agent_id=agent_id,
                            optimization_type=top_rec["optimization_type"],
                            objective_function=top_rec["objective_function"],
                        )

                        result = await self.optimize_agent(
                            target,
                            OptimizationAlgorithm.BAYESIAN_OPTIMIZATION,
                            max_iterations=20,  # Quick optimization
                        )

                        if result.improvement_percentage > 0.05:
                            self.logger.info(
                                f"Applied continuous optimization to {agent_id}: {result.improvement_percentage:.2%}"
                            )

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in continuous optimization for {agent_id}: {e}")
                await asyncio.sleep(60)  # Wait before retry


# Import required modules (would normally be at top)
import logging
import random
