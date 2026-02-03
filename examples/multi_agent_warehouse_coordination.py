"""
Multi-Agent Warehouse Coordination Example

Demonstrates social intelligence in LRS-Agents:
- Social precision tracking between agents
- Communication to reduce uncertainty
- Turn-based coordination
- Shared world state
"""

import random
from typing import Dict, Any

from lrs.multi_agent import MultiAgentCoordinator, SocialPrecisionTracker
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.integration.langgraph import create_lrs_agent
from langchain_anthropic import ChatAnthropic


class InventoryTool(ToolLens):
    """Check warehouse inventory."""

    def __init__(self):
        super().__init__(
            name="check_inventory",
            input_schema={"type": "object", "properties": {"item": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"quantity": {"type": "integer"}}},
        )
        self.inventory = {
            "widget_a": 100,
            "widget_b": 50,
            "widget_c": 200,
        }

    def get(self, state):
        """Implement abstract method from ToolLens."""
        item = state.get("item", "widget_a")
        quantity = self.inventory.get(item, 0)

        return ExecutionResult(
            success=True,
            value={"quantity": quantity, "item": item},
            error=None,
            prediction_error=0.1,
        )

    def set(self, state, observation):
        """Update state with inventory check result."""
        return {**state, "last_inventory_check": observation, "inventory_checked": True}
        self.inventory = {
            "widget_a": 100,
            "widget_b": 50,
            "widget_c": 200,
        }

    def get(self, state):
        """Implement abstract method from ToolLens."""
        item = state.get("item", "widget_a")
        quantity = self.inventory.get(item, 0)

        return ExecutionResult(
            success=True,
            value={"quantity": quantity, "item": item},
            error=None,
            prediction_error=0.1,
        )


class PickTool(ToolLens):
    """Pick items from warehouse shelves."""

    def __init__(self):
        super().__init__(
            name="pick_item",
            input_schema={
                "type": "object",
                "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}},
            },
            output_schema={"type": "object", "properties": {"picked": {"type": "integer"}}},
        )
        # Sometimes picking fails (item misplaced, etc.)
        self.failure_rate = 0.2

    def get(self, state):
        if random.random() < self.failure_rate:
            return ExecutionResult(
                success=False, value=None, error="Item not found at location", prediction_error=0.8
            )

        item = state.get("item", "widget_a")
        quantity = state.get("quantity", 1)

        return ExecutionResult(
            success=True, value={"picked": quantity, "item": item}, error=None, prediction_error=0.1
        )

    def set(self, state, observation):
        """Update state with pick result."""
        return {
            **state,
            "last_pick": observation,
            "items_picked": state.get("items_picked", 0) + observation.get("picked", 0),
        }
        # Sometimes picking fails (item misplaced, etc.)
        self.failure_rate = 0.2

    def get(self, state):
        if random.random() < self.failure_rate:
            return ExecutionResult(
                success=False, value=None, error="Item not found at location", prediction_error=0.8
            )

        item = state.get("item", "widget_a")
        quantity = state.get("quantity", 1)

        return ExecutionResult(
            success=True, value={"picked": quantity, "item": item}, error=None, prediction_error=0.1
        )


class PackTool(ToolLens):
    """Pack items for shipping."""

    def __init__(self):
        super().__init__(
            name="pack_item",
            input_schema={
                "type": "object",
                "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}},
            },
            output_schema={"type": "object", "properties": {"packages": {"type": "integer"}}},
        )

    def get(self, state):
        item = state.get("item", "widget_a")
        quantity = state.get("quantity", 1)
        packages = (quantity + 9) // 10  # 10 items per package

        return ExecutionResult(
            success=True,
            value={"packages": packages, "item": item, "quantity": quantity},
            error=None,
            prediction_error=0.05,
        )

    def set(self, state, observation):
        """Update state with pack result."""
        return {
            **state,
            "last_pack": observation,
            "packages_created": state.get("packages_created", 0) + observation.get("packages", 0),
        }

    def get(self, state):
        item = state.get("item", "widget_a")
        quantity = state.get("quantity", 1)
        packages = (quantity + 9) // 10  # 10 items per package

        return ExecutionResult(
            success=True,
            value={"packages": packages, "item": item, "quantity": quantity},
            error=None,
            prediction_error=0.05,
        )


def create_specialized_agent(role: str, tools: list):
    """Create an agent with a specific role."""

    # Role-specific system prompt
    role_prompts = {
        "inventory": """
        You are the Inventory Manager. Your responsibilities:
        - Monitor warehouse inventory levels
        - Communicate stock status to other agents
        - Advise on item availability
        - Be precise about quantities
        """,
        "picker": """
        You are the Picker. Your responsibilities:
        - Retrieve items from warehouse shelves  
        - Handle picking failures gracefully
        - Communicate with packer about picked items
        - Work efficiently but carefully
        """,
        "packer": """
        You are the Packer. Your responsibilities:
        - Pack items for shipping
        - Optimize packaging (10 items per package)
        - Communicate packaging status
        - Ensure quality packing
        """,
    }

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", model_kwargs={"temperature": 0.7})

    agent = create_lrs_agent(
        llm=llm, tools=tools, preferences={"success": 3.0, "error": -2.0, "step_cost": -0.1}
    )

    return agent


def main():
    """Run multi-agent warehouse coordination example."""

    print("🏭 Multi-Agent Warehouse Coordination Demo")
    print("=" * 50)

    # Create specialized agents
    inventory_agent = create_specialized_agent("inventory", [InventoryTool()])

    picker_agent = create_specialized_agent("picker", [InventoryTool(), PickTool()])

    packer_agent = create_specialized_agent("packer", [PickTool(), PackTool()])

    # Set up coordinator
    coordinator = MultiAgentCoordinator()
    coordinator.register_agent("inventory", inventory_agent)
    coordinator.register_agent("picker", picker_agent)
    coordinator.register_agent("packer", packer_agent)

    # Run coordination task
    print("\n📦 Task: Process customer order for 35 widget_a")
    print("\nStarting coordination...\n")

    # Create a realistic order
    task = """
    Customer Order: Process order for 35 widget_a units
    
    Required workflow:
    1. Check inventory availability
    2. Pick items from warehouse  
    3. Pack items for shipping
    4. Report final status
    
    Constraints:
    - Must verify sufficient inventory before picking
    - Pick failures require inventory recheck
    - Pack efficiently (10 items per package)
    - Communicate status between agents
    """

    results = coordinator.run(
        task=task,
        max_rounds=15,
        turn_order=["inventory", "picker", "packer"],  # Logical workflow order
    )

    # Display results
    print("\n" + "=" * 50)
    print("📊 COORDINATION RESULTS")
    print("=" * 50)

    print(f"\n⏱️  Execution time: {results['execution_time']:.2f} seconds")
    print(f"🔄 Total rounds: {results['total_rounds']}")
    print(f"💬 Total messages: {results['total_messages']}")

    # Show social precision evolution
    print("\n🤝 SOCIAL PRECISION EVOLUTION")
    print("-" * 30)

    for agent_id, social_precisions in results["social_precisions"].items():
        print(f"\n{agent_id.upper()} trust levels:")
        for other_id, precision in social_precisions.items():
            trust_level = "HIGH" if precision > 0.6 else "MEDIUM" if precision > 0.4 else "LOW"
            print(f"  → {other_id}: {precision:.3f} ({trust_level})")

    # Show final world state
    print("\n🌍 FINAL WORLD STATE")
    print("-" * 30)

    final_state = results["final_state"]
    for agent_id, state in final_state.items():
        if agent_id == "coordinator":
            continue

        print(f"\n{agent_id.upper()}:")
        if state.get("last_action"):
            action = state["last_action"]
            print(f"  Last action: {action}")

        if state.get("precision"):
            precision = state["precision"]
            if isinstance(precision, dict):
                # Show all precision levels
                for level, value in precision.items():
                    print(f"  {level} precision: {value:.3f}")
            else:
                print(f"  Precision: {precision:.3f}")

        if state.get("completed"):
            print(f"  ✅ Task completed")

    # Performance insights
    print("\n💡 PERFORMANCE INSIGHTS")
    print("-" * 30)

    efficiency = results["total_messages"] / max(results["total_rounds"], 1)
    print(f"📊 Communication efficiency: {efficiency:.2f} messages/round")

    # Calculate overall trust
    total_trust = 0
    trust_count = 0
    for social_precisions in results["social_precisions"].values():
        for precision in social_precisions.values():
            total_trust += precision
            trust_count += 1

    avg_trust = total_trust / max(trust_count, 1)
    print(f"🤝 Average inter-agent trust: {avg_trust:.3f}")

    success_rate = sum(1 for state in final_state.values() if state.get("completed", False)) / 3
    print(f"✅ Task completion rate: {success_rate:.1%}")

    print(f"\n🎯 Key Social Intelligence Demonstration:")
    print(f"   • Agents learned from each other's behavior patterns")
    print(f"   • Communication was triggered when social uncertainty was high")
    print(f"   • Trust evolved based on observed reliability")
    print(f"   • Coordination emerged through social precision tracking")


if __name__ == "__main__":
    main()
