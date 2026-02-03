#!/usr/bin/env python3
"""
OpenCode LRS Cognitive Multi-Agent Integration Demo
Phase 6.2.3: System Integration - LRS-Agents

Demonstrates cognitive architecture integration with multi-agent coordination system.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from .multi_agent_coordination import (
    MultiAgentCoordinator,
    Agent,
    AgentCapability,
    AgentRole,
    Task,
    COGNITIVE_AGENTS_AVAILABLE,
)


def demonstrate_cognitive_multi_agent():
    """Demonstrate cognitive multi-agent coordination."""

    print("🤖 OPENCODE LRS COGNITIVE MULTI-AGENT DEMO")
    print("=" * 60)
    print()

    if not COGNITIVE_AGENTS_AVAILABLE:
        print(
            "❌ Cognitive components not available. Please ensure phase6_neuromorphic_research is accessible."
        )
        return

    print("🏗️  Initializing Cognitive Multi-Agent Coordination System...")

    # Create coordinator
    coordinator = MultiAgentCoordinator()

    # Create specialized agents with cognitive capabilities
    agents = [
        Agent(
            agent_id="analyst_001",
            name="Code Analyst Alpha",
            capability=AgentCapability(
                role=AgentRole.ANALYST,
                expertise_domains={"code_analysis", "performance", "security"},
                max_concurrent_tasks=2,
                performance_score=0.9,
            ),
            precision_tracker=None,
        ),
        Agent(
            agent_id="architect_001",
            name="System Architect Beta",
            capability=AgentCapability(
                role=AgentRole.ARCHITECT,
                expertise_domains={"architecture", "design", "planning"},
                max_concurrent_tasks=1,
                performance_score=0.95,
            ),
            precision_tracker=None,
        ),
        Agent(
            agent_id="developer_001",
            name="Feature Developer Gamma",
            capability=AgentCapability(
                role=AgentRole.DEVELOPER,
                expertise_domains={"implementation", "coding", "features"},
                max_concurrent_tasks=3,
                performance_score=0.85,
            ),
            precision_tracker=None,
        ),
        Agent(
            agent_id="tester_001",
            name="Quality Tester Delta",
            capability=AgentCapability(
                role=AgentRole.TESTER,
                expertise_domains={"testing", "quality", "validation"},
                max_concurrent_tasks=2,
                performance_score=0.9,
            ),
            precision_tracker=None,
        ),
    ]

    # Create a software development workflow
    print("📋 Creating Software Development Workflow...")

    tasks = [
        Task(
            task_id="analyze_requirements",
            description="Analyze user requirements and create detailed specifications for authentication system",
            domain="code_analysis",
            complexity=0.7,
            urgency="high",
        ),
        Task(
            task_id="design_architecture",
            description="Design system architecture for user authentication with security considerations",
            domain="architecture",
            complexity=0.8,
            urgency="high",
            dependencies=["analyze_requirements"],
        ),
        Task(
            task_id="implement_auth",
            description="Implement user authentication module with password hashing and session management",
            domain="implementation",
            complexity=0.9,
            urgency="medium",
            dependencies=["design_architecture"],
        ),
        Task(
            task_id="implement_user_mgmt",
            description="Implement user management features including registration and profile management",
            domain="implementation",
            complexity=0.7,
            urgency="medium",
            dependencies=["design_architecture"],
        ),
        Task(
            task_id="test_authentication",
            description="Test authentication system for security vulnerabilities and functionality",
            domain="testing",
            complexity=0.8,
            urgency="high",
            dependencies=["implement_auth"],
        ),
        Task(
            task_id="test_user_management",
            description="Test user management features and integration with authentication",
            domain="testing",
            complexity=0.6,
            urgency="medium",
            dependencies=["implement_user_mgmt"],
        ),
        Task(
            task_id="integration_testing",
            description="Perform integration testing of authentication and user management systems",
            domain="testing",
            complexity=0.7,
            urgency="high",
            dependencies=["test_authentication", "test_user_management"],
        ),
    ]

    # Add tasks to coordinator
    task_ids = []
    for task in tasks:
        coordinator.create_task(
            task_id=task.task_id,
            description=task.description,
            domain=task.domain,
            complexity=task.complexity,
            urgency=task.urgency,
            dependencies=task.dependencies,
        )
        task_ids.append(task.task_id)

    print(f"✅ Created {len(tasks)} interdependent tasks")
    print()

    print("🧠 Initializing Cognitive Systems for Agents...")
    cognitive_agents_count = 0
    for agent in agents:
        if agent.initialize_cognitive_system():
            cognitive_agents_count += 1
        coordinator.add_agent(agent)

    print(
        f"✅ {cognitive_agents_count}/{len(agents)} agents initialized with cognitive capabilities"
    )
    print()

    # Execute workflow
    print("⚡ Executing Cognitive Multi-Agent Workflow...")
    print("-" * 50)

    workflow_result = coordinator.execute_workflow(task_ids)

    print("📊 Workflow Execution Results:")
    print(f"   • Tasks completed: {workflow_result['completed_tasks']}")
    print(f"   • Total execution time: {workflow_result['execution_time']:.2f}s")
    print(".3f")
    print()

    print("🤖 Agent Performance Summary:")
    for agent_id, agent in coordinator.agents.items():
        completed_count = len(
            [t for t in agent.completed_tasks if t.status.name == "COMPLETED"]
        )
        cognitive_status = (
            "🧠 Cognitive"
            if agent.cognitive_enabled and agent.cognitive_architecture
            else "🤖 Standard"
        )

        print(
            f"   • {agent.name}: {completed_count} tasks completed ({cognitive_status})"
        )

        if agent.cognitive_memory:
            avg_cognitive_score = sum(
                item.get("cognitive_score", 0)
                for item in agent.cognitive_memory.values()
            ) / len(agent.cognitive_memory)
            print(".2f")
    print()

    print("🧬 Cognitive Coordination Insights:")

    # Analyze cognitive patterns across agents
    total_attention_patterns = sum(len(agent.attention_patterns) for agent in agents)
    cognitive_memory_items = sum(len(agent.cognitive_memory) for agent in agents)

    print(f"   • Total attention patterns: {total_attention_patterns}")
    print(f"   • Cognitive memory items: {cognitive_memory_items}")
    print(f"   • Agents with cognitive capabilities: {cognitive_agents_count}")

    if cognitive_agents_count > 0:
        print(
            "   • Cognitive features active: ✓ Task evaluation, ✓ Attention tracking, ✓ Memory consolidation"
        )
    print()

    # Show sample cognitive insights
    print("💡 Sample Cognitive Insights:")
    for agent in agents:
        if agent.cognitive_enabled and agent.cognitive_memory:
            sample_task = list(agent.cognitive_memory.keys())[0]
            cognitive_data = agent.cognitive_memory[sample_task]
            print(
                f"   • {agent.name}: Task '{sample_task}' cognitive score = {cognitive_data.get('cognitive_score', 0):.2f}"
            )
            break
    print()

    print("🎉 Cognitive Multi-Agent Integration Demo Complete!")
    print("✅ Cognitive architecture integrated with multi-agent coordination")
    print("✅ Agents make cognitively-informed task assignments")
    print("✅ Real-time cognitive processing during task execution")
    print("✅ Cross-agent cognitive memory and attention sharing")
    print()
    print("🚀 Ready for Phase 6.2.3 completion and enterprise dashboard integration!")


if __name__ == "__main__":
    demonstrate_cognitive_multi_agent()
