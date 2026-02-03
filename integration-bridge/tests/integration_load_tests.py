"""
Comprehensive integration and load testing for opencode ↔ LRS-Agents integration bridge.
"""

import asyncio
import time
import json
import statistics
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import pytest
import aiohttp
import structlog
from datetime import datetime, timedelta
import psutil
import psutil
import random

logger = structlog.get_logger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""

    concurrent_users: int = 100
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    target_rps: float = 50.0  # Requests per second
    scenarios: List[str] = field(
        default_factory=lambda: ["agent_creation", "tool_execution"]
    )
    endpoints: List[str] = field(
        default_factory=lambda: [
            "/agents",
            "/tools/execute",
            "/agents/{agent_id}",
            "/health",
        ]
    )


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    active_connections: List[int] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[min(index, len(sorted_times) - 1)]

    @property
    def avg_cpu_usage(self) -> float:
        """Calculate average CPU usage."""
        if not self.cpu_usage:
            return 0.0
        return statistics.mean(self.cpu_usage)

    @property
    def avg_memory_usage(self) -> float:
        """Calculate average memory usage."""
        if not self.memory_usage:
            return 0.0
        return statistics.mean(self.memory_usage)


class LoadTestScenario:
    """Base class for load test scenarios."""

    def __init__(self, name: str, config: LoadTestConfig, base_url: str):
        self.name = name
        self.config = config
        self.base_url = base_url
        self.metrics = PerformanceMetrics()
        self.session: Optional[aiohttp.ClientSession] = None
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def setup(self):
        """Setup scenario before test execution."""
        self.session = aiohttp.ClientSession()
        self.metrics = PerformanceMetrics()

    async def teardown(self):
        """Cleanup after test execution."""
        if self.session:
            await self.session.close()

    async def execute_request(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a single request and collect metrics."""
        start_time = time.time()
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.virtual_memory().percent

        try:
            url = f"{self.base_url}{endpoint}"

            if method == "GET":
                async with self.session.get(url, headers=headers) as response:
                    content = await response.text()
                    status = response.status
            elif method == "POST":
                async with self.session.post(
                    url, json=payload, headers=headers
                ) as response:
                    content = await response.text()
                    status = response.status
            elif method == "PUT":
                async with self.session.put(
                    url, json=payload, headers=headers
                ) as response:
                    content = await response.text()
                    status = response.status
            else:
                raise ValueError(f"Unsupported method: {method}")

            response_time = time.time() - start_time
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().percent

            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.response_times.append(response_time)
            self.metrics.cpu_usage.append((cpu_before + cpu_after) / 2)
            self.metrics.memory_usage.append((memory_before + memory_after) / 2)

            if 200 <= status < 400:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
                error_type = self._classify_error(status)
                self.metrics.errors_by_type[error_type] = (
                    self.metrics.errors_by_type.get(error_type, 0) + 1
                )

            return {
                "status": status,
                "response_time": response_time,
                "content_length": len(content),
                "success": 200 <= status < 400,
                "scenario": self.name,
            }

        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.response_times.append(response_time)

            return {
                "status": "error",
                "response_time": response_time,
                "error": str(e),
                "success": False,
                "scenario": self.name,
            }

    def _classify_error(self, status_code: int) -> str:
        """Classify HTTP status codes into error types."""
        if 400 <= status_code < 500:
            return "client_error"
        elif status_code >= 500:
            return "server_error"
        else:
            return "unknown_error"

    async def run_load_test(self) -> PerformanceMetrics:
        """Execute the load test scenario."""
        logger.info(f"Starting load test scenario: {self.name}")
        await self.setup()

        try:
            await self._execute_load_pattern()
        finally:
            await self.teardown()

        logger.info(
            f"Load test scenario completed: {self.name}", **self._get_metrics_summary()
        )
        return self.metrics

    async def _execute_load_pattern(self):
        """Execute the specific load pattern for this scenario."""
        # Override in subclasses
        raise NotImplementedError


class AgentCreationScenario(LoadTestScenario):
    """Load test for agent creation endpoints."""

    async def _execute_load_pattern(self):
        """Execute agent creation load pattern."""
        duration = self.config.duration_seconds
        ramp_up = self.config.ramp_up_seconds
        target_rps = self.config.target_rps
        concurrent_users = self.config.concurrent_users

        # Calculate request spacing
        request_spacing = 1.0 / target_rps if target_rps > 0 else 0.1

        start_time = time.time()
        end_time = start_time + duration

        # Start with gradual ramp-up
        initial_tasks = []

        for i in range(concurrent_users):
            # Calculate delay for ramp-up
            delay = (i / concurrent_users) * ramp_up

            task = asyncio.create_task(self._delayed_agent_creation(delay))
            initial_tasks.append(task)

        # Wait for all tasks to complete
        if initial_tasks:
            await asyncio.gather(*initial_tasks, return_exceptions=True)

        # Continuous load phase
        while time.time() < end_time:
            tasks = []
            current_time = time.time()

            # Calculate how many requests to make in next second
            for _ in range(int(target_rps)):
                if current_time >= end_time:
                    break

                task = asyncio.create_task(self._delayed_agent_creation(0))
                tasks.append(task)

            # Execute batch
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Small delay to maintain target RPS
            await asyncio.sleep(request_spacing)

    async def _delayed_agent_creation(self, delay: float):
        """Create agent with specified delay."""
        if delay > 0:
            await asyncio.sleep(delay)

        agent_data = {
            "agent_id": f"load_test_agent_{random.randint(10000, 99999)}",
            "agent_type": random.choice(["lrs", "opencode", "hybrid"]),
            "config": {
                "goal": f"Load test agent {random.randint(1, 1000)}",
                "preferences": {"max_retries": 3},
            },
            "tools": ["search", "file_operation"],
        }

        return await self.execute_request(
            "/agents",
            method="POST",
            payload=agent_data,
            headers={"Content-Type": "application/json"},
        )


class ToolExecutionScenario(LoadTestScenario):
    """Load test for tool execution endpoints."""

    async def _execute_load_pattern(self):
        """Execute tool execution load pattern."""
        duration = self.config.duration_seconds
        end_time = time.time() + duration

        # Create some agents first
        agents_created = []
        for i in range(min(50, self.config.concurrent_users)):
            agent_id = f"tool_test_agent_{i}"
            await self.execute_request(
                "/agents",
                method="POST",
                payload={
                    "agent_id": agent_id,
                    "agent_type": "hybrid",
                    "config": {"goal": "Tool test agent"},
                    "tools": ["search", "file_operation"],
                },
            )
            agents_created.append(agent_id)

        await asyncio.sleep(2)  # Let agents be created

        # Execute tools in parallel
        while time.time() < end_time:
            tasks = []

            for i in range(min(self.config.concurrent_users, len(agents_created))):
                agent_id = agents_created[i]
                tool_data = {
                    "tool_name": "search",
                    "parameters": {
                        "query": f"Load test query {random.randint(1, 1000)}",
                        "max_results": random.randint(5, 20),
                    },
                    "agent_id": agent_id,
                    "timeout": 30.0,
                }

                task = asyncio.create_task(
                    self.execute_request(
                        "/tools/execute",
                        method="POST",
                        payload=tool_data,
                        headers={"Content-Type": "application/json"},
                    )
                )
                tasks.append(task)

            # Execute batch
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Delay between batches
            await asyncio.sleep(0.1)


class MixedScenario(LoadTestScenario):
    """Mixed load test with multiple endpoint types."""

    async def _execute_load_pattern(self):
        """Execute mixed load pattern."""
        duration = self.config.duration_seconds
        end_time = time.time() + duration

        agents_created = []

        while time.time() < end_time:
            tasks = []

            # Mix of different operations
            operations = [
                ("GET", "/health", None),
                ("POST", "/agents", self._generate_agent_data()),
                ("POST", "/tools/execute", self._generate_tool_data()),
                ("GET", "/metrics", None),
            ]

            for i in range(min(self.config.concurrent_users, len(operations))):
                operation = operations[i % len(operations)]

                if operation[0] == "GET":
                    task = asyncio.create_task(
                        self.execute_request(operation[1], method=operation[0])
                    )
                else:
                    task = asyncio.create_task(
                        self.execute_request(
                            operation[1],
                            method=operation[0],
                            payload=operation[2],
                            headers={"Content-Type": "application/json"},
                        )
                    )
                tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(0.2)

    def _generate_agent_data(self) -> Dict[str, Any]:
        """Generate random agent data."""
        return {
            "agent_id": f"mixed_agent_{random.randint(10000, 99999)}",
            "agent_type": random.choice(["lrs", "opencode", "hybrid"]),
            "config": {"goal": "Mixed load test agent"},
            "tools": ["search", "file_operation"],
        }

    def _generate_tool_data(self) -> Dict[str, Any]:
        """Generate random tool execution data."""
        return {
            "tool_name": random.choice(["search", "file_operation"]),
            "parameters": {
                "query": f"Mixed test {random.randint(1, 1000)}",
                "random_param": random.randint(1, 100),
            },
            "timeout": 30.0,
        }


class LoadTestRunner:
    """Orchestrates multiple load test scenarios."""

    def __init__(self, config: LoadTestConfig, base_url: str):
        self.config = config
        self.base_url = base_url
        self.scenarios = {
            "agent_creation": AgentCreationScenario("agent_creation", config, base_url),
            "tool_execution": ToolExecutionScenario("tool_execution", config, base_url),
            "mixed": MixedScenario("mixed", config, base_url),
        }
        self.results: Dict[str, PerformanceMetrics] = {}

    async def run_all_scenarios(self) -> Dict[str, PerformanceMetrics]:
        """Run all configured load test scenarios."""
        logger.info("Starting comprehensive load testing")

        results = {}

        for scenario_name, scenario in self.scenarios.items():
            if scenario_name in self.config.scenarios:
                logger.info(f"Running scenario: {scenario_name}")
                results[scenario_name] = await scenario.run_load_test()

                # Cool down between scenarios
                await asyncio.sleep(5)

        self.results = results

        # Generate comprehensive report
        self._generate_report(results)

        logger.info("All load test scenarios completed")
        return results

    def _generate_report(self, results: Dict[str, PerformanceMetrics]):
        """Generate comprehensive load test report."""
        report = {
            "test_summary": {
                "timestamp": datetime.utcnow().isoformat(),
                "config": {
                    "concurrent_users": self.config.concurrent_users,
                    "duration_seconds": self.config.duration_seconds,
                    "target_rps": self.config.target_rps,
                    "scenarios_run": list(results.keys()),
                },
            },
            "scenario_results": {},
        }

        for scenario_name, metrics in results.items():
            report["scenario_results"][scenario_name] = {
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "p95_response_time": metrics.p95_response_time,
                    "avg_cpu_usage": metrics.avg_cpu_usage,
                    "avg_memory_usage": metrics.avg_memory_usage,
                    "errors_by_type": metrics.errors_by_type,
                },
                "performance_assessment": self._assess_performance(metrics),
            }

        # Save report
        report_file = (
            f"load_test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Load test report saved to: {report_file}")

    def _assess_performance(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Assess performance against benchmarks."""
        assessment = {
            "overall_grade": "A",  # A, B, C, D, F
            "issues": [],
            "recommendations": [],
        }

        # Response time assessment
        if metrics.p95_response_time > 2.0:  # 2 seconds threshold
            assessment["issues"].append("High P95 response time")
            assessment["recommendations"].append(
                "Optimize database queries and caching"
            )
            if metrics.p95_response_time > 5.0:
                assessment["overall_grade"] = "D"
        elif metrics.p95_response_time > 1.0:
            assessment["overall_grade"] = "B"

        # Success rate assessment
        if metrics.success_rate < 95:
            assessment["issues"].append("Low success rate")
            assessment["recommendations"].append(
                "Improve error handling and resilience"
            )
            if metrics.success_rate < 90:
                assessment["overall_grade"] = "C"
        elif metrics.success_rate < 85:
            assessment["overall_grade"] = "D"

        # Resource usage assessment
        if metrics.avg_cpu_usage > 80:
            assessment["issues"].append("High CPU usage")
            assessment["recommendations"].append(
                "Scale horizontally or optimize algorithms"
            )

        if metrics.avg_memory_usage > 80:
            assessment["issues"].append("High memory usage")
            assessment["recommendations"].append(
                "Optimize memory usage and implement pooling"
            )

        return assessment


# Pytest fixtures for integration testing
@pytest.fixture
async def integration_test_client():
    """Create test client for integration tests."""
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def test_config():
    """Test configuration for integration tests."""
    return LoadTestConfig(
        concurrent_users=10,
        duration_seconds=10,
        target_rps=5.0,
        scenarios=["agent_creation", "tool_execution"],
    )


class TestIntegrationScenarios:
    """Integration test scenarios."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, integration_test_client):
        """Test complete agent lifecycle: creation → execution → deletion."""
        async with integration_test_client as session:
            base_url = "http://localhost:9000"

            # 1. Create agent
            agent_data = {
                "agent_id": "integration_test_agent",
                "agent_type": "hybrid",
                "config": {"goal": "Integration test"},
                "tools": ["search", "file_operation"],
            }

            async with session.post(f"{base_url}/agents", json=agent_data) as resp:
                assert resp.status == 200
                agent_response = await resp.json()

            agent_id = agent_response["agent_id"]

            # 2. Execute tool
            tool_data = {
                "tool_name": "search",
                "parameters": {"query": "integration test query"},
                "agent_id": agent_id,
            }

            async with session.post(
                f"{base_url}/tools/execute", json=tool_data
            ) as resp:
                assert resp.status == 200
                tool_response = await resp.json()
                assert tool_response["status"] in ["pending", "completed"]

            # 3. Get agent state
            async with session.get(f"{base_url}/agents/{agent_id}") as resp:
                assert resp.status == 200
                state_response = await resp.json()
                assert state_response["agent_id"] == agent_id

            # 4. Update agent
            update_data = {
                "config": {"goal": "Updated integration test goal"},
                "status": "active",
            }

            async with session.put(
                f"{base_url}/agents/{agent_id}", json=update_data
            ) as resp:
                assert resp.status == 200

            # 5. Delete agent
            async with session.delete(f"{base_url}/agents/{agent_id}") as resp:
                assert resp.status == 200

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self, integration_test_client):
        """Test concurrent operations on multiple agents."""
        async with integration_test_client as session:
            base_url = "http://localhost:9000"

            # Create multiple agents concurrently
            tasks = []
            for i in range(5):
                agent_data = {
                    "agent_id": f"concurrent_test_agent_{i}",
                    "agent_type": "lrs",
                    "config": {"goal": f"Concurrent test agent {i}"},
                    "tools": ["search"],
                }

                task = asyncio.create_task(
                    session.post(f"{base_url}/agents", json=agent_data)
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            for resp in responses:
                assert resp.status == 200

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_integration(self, integration_test_client):
        """Test WebSocket integration."""
        import websockets

        uri = "ws://localhost:9001"

        async with websockets.connect(uri) as websocket:
            # Subscribe to agent updates
            subscribe_message = {
                "type": "subscribe_agent",
                "data": {"agent_id": "test_agent_001"},
            }

            await websocket.send(json.dumps(subscribe_message))

            # Wait for subscription response
            response = await websocket.recv()
            response_data = json.loads(response)
            assert response_data["type"] == "subscription_response"
            assert response_data["data"]["subscribed"] is True

            # Send ping
            ping_message = {"type": "ping", "data": {}}
            await websocket.send(json.dumps(ping_message))

            # Wait for pong
            pong_response = await websocket.recv()
            pong_data = json.loads(pong_response)
            assert pong_data["type"] == "pong"

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_load_scenario_performance(self, test_config):
        """Test load scenario performance."""
        config = LoadTestConfig(
            concurrent_users=20,
            duration_seconds=30,
            target_rps=10.0,
            scenarios=["agent_creation"],
        )

        runner = LoadTestRunner(config, "http://localhost:9000")
        results = await runner.run_all_scenarios()

        # Verify performance meets expectations
        agent_creation_metrics = results["agent_creation"]

        assert agent_creation_metrics.success_rate > 95.0
        assert agent_creation_metrics.p95_response_time < 2.0
        assert agent_creation_metrics.avg_cpu_usage < 80.0
        assert agent_creation_metrics.avg_memory_usage < 80.0


# CLI interface for running load tests
import click


@click.command()
@click.option("--concurrent-users", default=100, help="Number of concurrent users")
@click.option("--duration", default=60, help="Test duration in seconds")
@click.option("--target-rps", default=50.0, help="Target requests per second")
@click.option(
    "--scenarios",
    default="agent_creation,tool_execution,mixed",
    help="Comma-separated list of scenarios",
)
@click.option(
    "--base-url", default="http://localhost:9000", help="Base URL for the service"
)
@click.option(
    "--output-format",
    default="json",
    type=click.Choice(["json", "yaml", "console"]),
    help="Output format",
)
def run_load_tests(
    concurrent_users, duration, target_rps, scenarios, base_url, output_format
):
    """Run load tests."""
    scenarios_list = scenarios.split(",")

    config = LoadTestConfig(
        concurrent_users=concurrent_users,
        duration_seconds=duration,
        target_rps=target_rps,
        scenarios=scenarios_list,
    )

    runner = LoadTestRunner(config, base_url)

    async def run_tests():
        results = await runner.run_all_scenarios()

        if output_format == "json":
            output_file = (
                f"load_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {output_file}")
        elif output_format == "console":
            for scenario_name, metrics in results.items():
                print(f"\n=== {scenario_name.upper()} ===")
                print(f"Total Requests: {metrics.total_requests}")
                print(f"Success Rate: {metrics.success_rate:.2f}%")
                print(f"Avg Response Time: {metrics.avg_response_time:.3f}s")
                print(f"P95 Response Time: {metrics.p95_response_time:.3f}s")
                print(f"Avg CPU Usage: {metrics.avg_cpu_usage:.1f}%")
                print(f"Avg Memory Usage: {metrics.avg_memory_usage:.1f}%")
        elif output_format == "yaml":
            import yaml

            output_file = (
                f"load_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.yaml"
            )
            with open(output_file, "w") as f:
                yaml.dump(results, f, default_flow_style=False)
            print(f"Results saved to {output_file}")

    asyncio.run(run_tests())


if __name__ == "__main__":
    run_load_tests()
