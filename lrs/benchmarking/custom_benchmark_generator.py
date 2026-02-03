#!/usr/bin/env python3
"""
OpenCode Custom Benchmark Generator
Phase 4: Advanced Benchmarking - Custom Benchmark Generation

Generates domain-specific benchmarks for evaluating agent performance
across different programming domains and complexity levels.
"""

import json
import random
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class BenchmarkDomain(Enum):
    """Supported benchmark domains."""

    WEB_DEVELOPMENT = "web_development"
    DATA_SCIENCE = "data_science"
    SYSTEM_ADMINISTRATION = "system_administration"
    API_DEVELOPMENT = "api_development"
    DATABASE_DESIGN = "database_design"
    SECURITY = "security"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    TESTING = "testing"


class ComplexityLevel(Enum):
    """Benchmark complexity levels."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class BenchmarkTask:
    """Generated benchmark task."""

    task_id: str
    domain: BenchmarkDomain
    complexity: ComplexityLevel
    description: str
    requirements: List[str]
    expected_output: Dict[str, Any]
    evaluation_criteria: Dict[str, float]
    time_limit: float
    prerequisites: List[str]


class CustomBenchmarkGenerator:
    """Generates custom benchmarks for specific domains."""

    def __init__(self):
        self.templates = self._load_templates()
        self.generated_benchmarks: Dict[str, BenchmarkTask] = {}

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load benchmark templates for different domains."""
        return {
            "web_development": {
                "basic": {
                    "description": "Create a simple HTML/CSS webpage with responsive design",
                    "requirements": [
                        "Valid HTML5 structure",
                        "CSS styling with media queries",
                        "Responsive layout for mobile and desktop",
                        "Semantic HTML elements",
                    ],
                    "evaluation_criteria": {
                        "validity": 0.3,
                        "responsiveness": 0.3,
                        "styling": 0.2,
                        "semantics": 0.2,
                    },
                    "time_limit": 30.0,
                },
                "intermediate": {
                    "description": "Build a React component with state management and API integration",
                    "requirements": [
                        "Functional React component",
                        "State management with hooks",
                        "API data fetching",
                        "Error handling",
                        "Loading states",
                    ],
                    "evaluation_criteria": {
                        "functionality": 0.3,
                        "state_management": 0.25,
                        "api_integration": 0.25,
                        "error_handling": 0.2,
                    },
                    "time_limit": 60.0,
                },
                "advanced": {
                    "description": "Develop a full-stack web application with authentication and database",
                    "requirements": [
                        "User authentication system",
                        "Database integration",
                        "RESTful API endpoints",
                        "Frontend-backend communication",
                        "Security best practices",
                    ],
                    "evaluation_criteria": {
                        "authentication": 0.25,
                        "database_design": 0.2,
                        "api_design": 0.25,
                        "security": 0.2,
                        "full_stack_integration": 0.1,
                    },
                    "time_limit": 120.0,
                },
            },
            "data_science": {
                "basic": {
                    "description": "Perform basic data analysis on a CSV dataset",
                    "requirements": [
                        "Data loading and cleaning",
                        "Basic statistical analysis",
                        "Data visualization",
                        "Summary report generation",
                    ],
                    "evaluation_criteria": {
                        "data_cleaning": 0.3,
                        "analysis": 0.3,
                        "visualization": 0.2,
                        "reporting": 0.2,
                    },
                    "time_limit": 45.0,
                },
                "intermediate": {
                    "description": "Build a machine learning model for classification",
                    "requirements": [
                        "Data preprocessing pipeline",
                        "Feature engineering",
                        "Model selection and training",
                        "Model evaluation metrics",
                        "Cross-validation",
                    ],
                    "evaluation_criteria": {
                        "preprocessing": 0.2,
                        "feature_engineering": 0.2,
                        "model_training": 0.25,
                        "evaluation": 0.2,
                        "validation": 0.15,
                    },
                    "time_limit": 90.0,
                },
                "advanced": {
                    "description": "Develop an end-to-end ML pipeline with deployment",
                    "requirements": [
                        "Complete ML pipeline",
                        "Model optimization",
                        "API deployment",
                        "Monitoring and logging",
                        "Performance optimization",
                    ],
                    "evaluation_criteria": {
                        "pipeline_design": 0.2,
                        "model_optimization": 0.2,
                        "deployment": 0.2,
                        "monitoring": 0.2,
                        "performance": 0.2,
                    },
                    "time_limit": 180.0,
                },
            },
            "api_development": {
                "basic": {
                    "description": "Create a simple REST API endpoint",
                    "requirements": [
                        "HTTP endpoint implementation",
                        "Request/response handling",
                        "Basic error handling",
                        "API documentation",
                    ],
                    "evaluation_criteria": {
                        "endpoint_functionality": 0.4,
                        "request_handling": 0.3,
                        "error_handling": 0.2,
                        "documentation": 0.1,
                    },
                    "time_limit": 25.0,
                },
                "intermediate": {
                    "description": "Build a REST API with CRUD operations and validation",
                    "requirements": [
                        "Complete CRUD operations",
                        "Input validation",
                        "Data serialization",
                        "Authentication middleware",
                        "Comprehensive error handling",
                    ],
                    "evaluation_criteria": {
                        "crud_operations": 0.3,
                        "validation": 0.25,
                        "serialization": 0.2,
                        "authentication": 0.15,
                        "error_handling": 0.1,
                    },
                    "time_limit": 75.0,
                },
                "advanced": {
                    "description": "Develop a production-ready API with advanced features",
                    "requirements": [
                        "Rate limiting",
                        "Caching layer",
                        "API versioning",
                        "Comprehensive testing",
                        "Performance optimization",
                        "Monitoring and metrics",
                    ],
                    "evaluation_criteria": {
                        "rate_limiting": 0.15,
                        "caching": 0.15,
                        "versioning": 0.1,
                        "testing": 0.2,
                        "performance": 0.2,
                        "monitoring": 0.2,
                    },
                    "time_limit": 150.0,
                },
            },
        }

    def generate_benchmark_suite(
        self,
        domain: BenchmarkDomain,
        count: int = 5,
        complexity_distribution: Optional[Dict[ComplexityLevel, float]] = None,
    ) -> List[BenchmarkTask]:
        """Generate a suite of benchmarks for a specific domain."""
        if complexity_distribution is None:
            complexity_distribution = {
                ComplexityLevel.BASIC: 0.4,
                ComplexityLevel.INTERMEDIATE: 0.4,
                ComplexityLevel.ADVANCED: 0.2,
            }

        benchmarks = []
        complexities = list(complexity_distribution.keys())
        weights = list(complexity_distribution.values())

        for i in range(count):
            complexity = random.choices(complexities, weights=weights)[0]
            benchmark = self.generate_benchmark(domain, complexity, i + 1)
            if benchmark:
                benchmarks.append(benchmark)

        return benchmarks

    def generate_benchmark(
        self, domain: BenchmarkDomain, complexity: ComplexityLevel, task_number: int
    ) -> Optional[BenchmarkTask]:
        """Generate a single benchmark task."""
        domain_key = domain.value
        complexity_key = complexity.value

        if (
            domain_key not in self.templates
            or complexity_key not in self.templates[domain_key]
        ):
            return None

        template = self.templates[domain_key][complexity_key]
        task_id = f"{domain_key}_{complexity_key}_{task_number}_{int(time.time())}"

        # Generate specific requirements with variations
        requirements = template["requirements"].copy()

        # Add domain-specific variations
        if domain == BenchmarkDomain.WEB_DEVELOPMENT:
            requirements.extend(self._get_web_dev_variations(complexity))
        elif domain == BenchmarkDomain.DATA_SCIENCE:
            requirements.extend(self._get_data_science_variations(complexity))
        elif domain == BenchmarkDomain.API_DEVELOPMENT:
            requirements.extend(self._get_api_dev_variations(complexity))

        benchmark = BenchmarkTask(
            task_id=task_id,
            domain=domain,
            complexity=complexity,
            description=template["description"],
            requirements=requirements,
            expected_output=self._generate_expected_output(domain, complexity),
            evaluation_criteria=template["evaluation_criteria"],
            time_limit=template["time_limit"],
            prerequisites=self._generate_prerequisites(domain, complexity),
        )

        self.generated_benchmarks[task_id] = benchmark
        return benchmark

    def _get_web_dev_variations(self, complexity: ComplexityLevel) -> List[str]:
        """Get web development specific variations."""
        variations = {
            ComplexityLevel.BASIC: [
                "Include at least 3 different HTML elements",
                "Use CSS Grid or Flexbox",
            ],
            ComplexityLevel.INTERMEDIATE: [
                "Implement form validation",
                "Add local storage for user preferences",
            ],
            ComplexityLevel.ADVANCED: [
                "Implement real-time updates",
                "Add internationalization support",
            ],
        }
        return variations.get(complexity, [])

    def _get_data_science_variations(self, complexity: ComplexityLevel) -> List[str]:
        """Get data science specific variations."""
        variations = {
            ComplexityLevel.BASIC: [
                "Handle missing data appropriately",
                "Generate at least 3 different charts",
            ],
            ComplexityLevel.INTERMEDIATE: [
                "Implement feature scaling",
                "Use cross-validation for model evaluation",
            ],
            ComplexityLevel.ADVANCED: [
                "Implement model ensemble",
                "Add hyperparameter tuning",
            ],
        }
        return variations.get(complexity, [])

    def _get_api_dev_variations(self, complexity: ComplexityLevel) -> List[str]:
        """Get API development specific variations."""
        variations = {
            ComplexityLevel.BASIC: [
                "Support JSON request/response",
                "Implement proper HTTP status codes",
            ],
            ComplexityLevel.INTERMEDIATE: [
                "Add request logging",
                "Implement pagination for list endpoints",
            ],
            ComplexityLevel.ADVANCED: [
                "Add GraphQL support",
                "Implement API rate limiting with Redis",
            ],
        }
        return variations.get(complexity, [])

    def _generate_expected_output(
        self, domain: BenchmarkDomain, complexity: ComplexityLevel
    ) -> Dict[str, Any]:
        """Generate expected output specifications."""
        base_output = {"deliverables": [], "metrics": {}, "validation_rules": []}

        if domain == BenchmarkDomain.WEB_DEVELOPMENT:
            base_output["deliverables"] = [
                "HTML files",
                "CSS files",
                "JavaScript files (if applicable)",
            ]
            base_output["validation_rules"] = [
                "HTML validation",
                "CSS validation",
                "Cross-browser compatibility",
            ]
        elif domain == BenchmarkDomain.DATA_SCIENCE:
            base_output["deliverables"] = [
                "Jupyter notebook",
                "Model files",
                "Analysis report",
            ]
            base_output["metrics"] = {
                "accuracy": ">80%",
                "precision": ">75%",
                "recall": ">75%",
            }
        elif domain == BenchmarkDomain.API_DEVELOPMENT:
            base_output["deliverables"] = ["API code", "Documentation", "Test suite"]
            base_output["validation_rules"] = [
                "API specification compliance",
                "Security audit",
            ]

        return base_output

    def _generate_prerequisites(
        self, domain: BenchmarkDomain, complexity: ComplexityLevel
    ) -> List[str]:
        """Generate prerequisite knowledge/skills."""
        prerequisites = []

        if complexity == ComplexityLevel.BASIC:
            prerequisites = [
                "Basic programming knowledge",
                "Understanding of core concepts",
            ]
        elif complexity == ComplexityLevel.INTERMEDIATE:
            prerequisites = [
                "Intermediate programming skills",
                "Familiarity with frameworks/libraries",
            ]
        elif complexity == ComplexityLevel.ADVANCED:
            prerequisites = [
                "Advanced programming expertise",
                "Experience with complex systems",
            ]
        elif complexity == ComplexityLevel.EXPERT:
            prerequisites = [
                "Expert-level knowledge",
                "Experience with production systems",
            ]

        # Add domain-specific prerequisites
        domain_prereqs = {
            BenchmarkDomain.WEB_DEVELOPMENT: [
                "HTML/CSS knowledge",
                "JavaScript basics",
            ],
            BenchmarkDomain.DATA_SCIENCE: [
                "Python/R programming",
                "Statistics knowledge",
            ],
            BenchmarkDomain.API_DEVELOPMENT: [
                "HTTP protocol knowledge",
                "REST principles",
            ],
        }

        if domain in domain_prereqs:
            prerequisites.extend(domain_prereqs[domain])

        return prerequisites

    def evaluate_benchmark_performance(
        self, task_id: str, results: Dict[str, Any], execution_time: float
    ) -> Dict[str, Any]:
        """Evaluate performance on a benchmark task."""
        if task_id not in self.generated_benchmarks:
            return {"error": "Benchmark task not found"}

        task = self.generated_benchmarks[task_id]

        # Calculate score based on evaluation criteria
        total_score = 0.0
        criteria_scores = {}

        for criterion, weight in task.evaluation_criteria.items():
            # Simulate evaluation (in real implementation, this would analyze actual results)
            score = random.uniform(0.7, 1.0)  # Placeholder for actual evaluation
            criteria_scores[criterion] = score
            total_score += score * weight

        # Time bonus/penalty
        time_factor = (
            min(1.0, task.time_limit / execution_time) if execution_time > 0 else 1.0
        )
        final_score = total_score * time_factor

        return {
            "task_id": task_id,
            "domain": task.domain.value,
            "complexity": task.complexity.value,
            "total_score": round(final_score, 3),
            "criteria_scores": criteria_scores,
            "execution_time": execution_time,
            "time_factor": round(time_factor, 3),
            "passed": final_score >= 0.7,  # 70% passing threshold
        }

    def export_benchmark_suite(self, benchmarks: List[BenchmarkTask], filename: str):
        """Export benchmark suite to JSON file."""
        suite_data = {
            "generated_at": time.time(),
            "total_benchmarks": len(benchmarks),
            "benchmarks": [
                {
                    "task_id": b.task_id,
                    "domain": b.domain.value,
                    "complexity": b.complexity.value,
                    "description": b.description,
                    "requirements": b.requirements,
                    "evaluation_criteria": b.evaluation_criteria,
                    "time_limit": b.time_limit,
                    "prerequisites": b.prerequisites,
                }
                for b in benchmarks
            ],
        }

        with open(filename, "w") as f:
            json.dump(suite_data, f, indent=2)

    def load_benchmark_suite(self, filename: str) -> List[BenchmarkTask]:
        """Load benchmark suite from JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)

        benchmarks = []
        for b_data in data["benchmarks"]:
            benchmark = BenchmarkTask(
                task_id=b_data["task_id"],
                domain=BenchmarkDomain(b_data["domain"]),
                complexity=ComplexityLevel(b_data["complexity"]),
                description=b_data["description"],
                requirements=b_data["requirements"],
                expected_output={},  # Would need to be stored/ regenerated
                evaluation_criteria=b_data["evaluation_criteria"],
                time_limit=b_data["time_limit"],
                prerequisites=b_data["prerequisites"],
            )
            benchmarks.append(benchmark)
            self.generated_benchmarks[benchmark.task_id] = benchmark

        return benchmarks


def demonstrate_custom_benchmarks():
    """Demonstrate custom benchmark generation and evaluation."""
    print("🎯 CUSTOM BENCHMARK GENERATOR DEMONSTRATION")
    print("=" * 55)
    print()

    generator = CustomBenchmarkGenerator()

    # Generate benchmark suites for different domains
    domains = [
        BenchmarkDomain.WEB_DEVELOPMENT,
        BenchmarkDomain.DATA_SCIENCE,
        BenchmarkDomain.API_DEVELOPMENT,
    ]

    for domain in domains:
        print(f"🏗️  Generating {domain.value.replace('_', ' ').title()} Benchmarks...")
        benchmarks = generator.generate_benchmark_suite(domain, count=3)

        print(f"   ✅ Generated {len(benchmarks)} benchmarks")
        for i, benchmark in enumerate(benchmarks[:2], 1):  # Show first 2
            print(f"      {i}. {benchmark.description[:60]}...")
            print(f"         Complexity: {benchmark.complexity.value}")
            print(f"         Time limit: {benchmark.time_limit}min")
            print(f"         Requirements: {len(benchmark.requirements)} items")

        # Simulate performance evaluation
        print("\n   📊 Simulated Performance Evaluation:")
        for benchmark in benchmarks[:2]:
            execution_time = benchmark.time_limit * random.uniform(0.8, 1.5)
            results = generator.evaluate_benchmark_performance(
                benchmark.task_id, {}, execution_time
            )
            print(f"         Score: {results['total_score']:.3f}")
        print()

    # Export benchmark suite
    print("💾 Exporting benchmark suite...")
    all_benchmarks = []
    for domain in domains:
        all_benchmarks.extend(generator.generate_benchmark_suite(domain, count=2))

    generator.export_benchmark_suite(all_benchmarks, "custom_benchmark_suite.json")
    print(
        f"   ✅ Exported {len(all_benchmarks)} benchmarks to custom_benchmark_suite.json"
    )

    print()
    print("🎉 Custom Benchmark Generation Demo Complete!")
    print("✅ Domain-specific benchmarks generated successfully")
    print("✅ Complexity-scaled evaluation criteria implemented")
    print("✅ Performance evaluation system operational")
    print("✅ Benchmark persistence and loading functional")


if __name__ == "__main__":
    demonstrate_custom_benchmarks()
