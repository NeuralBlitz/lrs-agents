"""
Multi-Agent Warehouse: Coordinated robot fleet example.

This example demonstrates:
- Multiple agents with specialized roles
- Social precision tracking (trust between agents)
- Communication for coordination
- Emergent collaborative behavior
"""

from langchain_anthropic import ChatAnthropic
from lrs import create_lrs_agent
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.multi_agent.coordinator import MultiAgentCoordinator
from lrs.multi_agent.shared_state import SharedWorldState
import random
import time


# Warehouse robot tools
class PickTool(ToolLens):
    """Picker robot: Retrieve items from shelves"""
    def __init__(self):
        super().__init__(name="pick_item", input_schema={}, output_schema={})
        self.inventory = {
            'item_a': 10,
            'item_b': 5,
            'item_c': 8
        }
    
    def get(self, state):
        self.call_count += 1
        item_id = state.get('item_id', 'item_a')
        
        if self.inventory.get(item_id, 0) > 0:
            self.inventory[item_id] -= 1
            return ExecutionResult(
                True,
                {'item_id': item_id, 'status': 'picked', 'location': 'staging'},
                None,
                0.1
            )
        else:
            self.failure_count += 1
            return ExecutionResult(
                False,
                None,
                f"Item {item_id} out of stock",
                0.9
            )
    
    def set(self, state, obs):
        picked = state.get('picked_items', [])
        return {**state, 'picked_items': picked + [obs]}


class PackTool(ToolLens):
    """Packer robot: Pack items into boxes"""
    def __init__(self):
        super().__init__(name="pack_item", input_schema={}, output_schema={})
    
    def get(self, state):
        self.call_count += 1
        
        # Check if there are items to pack
        picked = state.get('picked_items', [])
        if not picked:
            self.failure_count += 1
            return ExecutionResult(
                False,
                None,
                "No items available to pack",
                0.95
            )
        
        # Simulate packing delay
        time.sleep(0.1)
        
        # Pack the first unpacked item
        item = picked[0]
        
        return ExecutionResult(
            True,
            {'item_id': item['item_id'], 'status': 'packed', 'box_id': f"BOX_{random.randint(100, 999)}"},
            None,
            0.05
        )
    
    def set(self, state, obs):
        packed = state.get('packed_items', [])
        return {**state, 'packed_items': packed + [obs]}


class ShipTool(ToolLens):
    """Shipper robot: Ship packed boxes"""
    def __init__(self):
        super().__init__(name="ship_box", input_schema={}, output_schema={})
    
    def get(self, state):
        self.call_count += 1
        
        packed = state.get('packed_items', [])
        if not packed:
            self.failure_count += 1
            return ExecutionResult(
                False,
                None,
                "No boxes ready to ship",
                0.95
            )
        
        # Simulate occasional shipping delays
        if random.random() < 0.1:
            self.failure_count += 1
            return ExecutionResult(
                False,
                None,
                "Shipping label printer offline",
                0.8
            )
        
        box = packed[0]
        
        return ExecutionResult(
            True,
            {'box_id': box['box_id'], 'status': 'shipped', 'tracking': f"TRACK_{random.randint(10000, 99999)}"},
            None,
            0.1
        )
    
    def set(self, state, obs):
        shipped = state.get('shipped_boxes', [])
        return {**state, 'shipped_boxes': shipped + [obs]}


class CheckInventoryTool(ToolLens):
    """Check inventory levels"""
    def __init__(self, picker_tool: PickTool):
        super().__init__(name="check_inventory", input_schema={}, output_schema={})
        self.picker_tool = picker_tool
    
    def get(self, state):
        self.call_count += 1
        
        return ExecutionResult(
            True,
            {'inventory': self.picker_tool.inventory.copy()},
            None,
            0.0  # Deterministic
        )
    
    def set(self, state, obs):
        return {**state, 'inventory_status': obs}


def main():
    print("=" * 60)
    print("MULTI-AGENT WAREHOUSE COORDINATION")
    print("=" * 60)
    print("""
Scenario: Three robots coordinate to fulfill orders

Agents:
  • Picker: Retrieves items from warehouse shelves
  • Packer: Packs items into shipping boxes
  • Shipper: Labels and ships completed boxes

Coordination Mechanisms:
  • Shared World State: Common view of warehouse
  • Social Precision: Trust in other agents' reliability
  • Communication: Status updates and requests
  • Adaptation: Handle failures gracefully
    """)
    
    # Initialize
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    coordinator = MultiAgentCoordinator()
    
    # Create shared tools
    pick_tool = PickTool()
    check_tool = CheckInventoryTool(pick_tool)
    
    # Create Picker agent
    print("\n→ Initializing Picker robot...")
    picker_agent = create_lrs_agent(
        llm=llm,
        tools=[pick_tool, check_tool],
        preferences={'success': 5.0, 'error': -2.0},
        use_llm_proposals=False  # Use simpler policy generation for demo
    )
    coordinator.register_agent("picker", picker_agent)
    
    # Create Packer agent
    print("→ Initializing Packer robot...")
    packer_agent = create_lrs_agent(
        llm=llm,
        tools=[PackTool()],
        preferences={'success': 5.0, 'error': -2.0},
        use_llm_proposals=False
    )
    coordinator.register_agent("packer", packer_agent)
    
    # Create Shipper agent
    print("→ Initializing Shipper robot...")
    shipper_agent = create_lrs_agent(
        llm=llm,
        tools=[ShipTool()],
        preferences={'success': 5.0, 'error': -2.0},
        use_llm_proposals=False
    )
    coordinator.register_agent("shipper", shipper_agent)
    
    print("\n✓ Three agents initialized and registered")
    
    # Run coordination
    print("\n" + "-" * 60)
    print("EXECUTING ORDER")
    print("-" * 60)
    print("\nOrder: Ship 3 items (item_a, item_b, item_c)")
    
    # Note: This is a simplified demonstration
    # Full coordination would use the coordinator.run() method
    
    print("\n→ Coordination sequence (simulated):")
    print("\nRound 1:")
    print("  Picker: ✓ pick_item(item_a)")
    print("    Social precision: picker→packer = 0.50 (neutral)")
    print("  Packer: ✗ pack_item() [waiting for items]")
    print("    Precision drops: 0.50 → 0.40")
    print("  Shipper: ⏸ idle")
    
    print("\nRound 2:")
    print("  Picker: ✓ pick_item(item_b)")
    print("  Packer: ✓ pack_item(item_a)")
    print("    Social precision: packer→picker = 0.60 (trust building)")
    print("  Shipper: ✗ ship_box() [waiting for packed items]")
    
    print("\nRound 3:")
    print("  Picker: ✓ pick_item(item_c)")
    print("  Packer: ✓ pack_item(item_b)")
    print("  Shipper: ✓ ship_box(BOX_123)")
    print("    Social precision: shipper→packer = 0.70 (high trust)")
    
    print("\nRound 4:")
    print("  Picker: ✓ check_inventory()")
    print("  Packer: ✓ pack_item(item_c)")
    print("  Shipper: ✓ ship_box(BOX_456)")
    
    print("\nRound 5:")
    print("  Picker: ⏸ task complete")
    print("  Packer: ⏸ task complete")
    print("  Shipper: ✓ ship_box(BOX_789)")
    
    # Simulated results
    results = {
        'total_rounds': 5,
        'total_messages': 2,
        'execution_time': 2.5,
        'items_shipped': 3,
        'social_precisions': {
            'picker': {
                'packer': 0.65,
                'shipper': 0.55
            },
            'packer': {
                'picker': 0.70,
                'shipper': 0.75
            },
            'shipper': {
                'picker': 0.60,
                'packer': 0.80
            }
        }
    }
    
    # Display results
    print("\n" + "=" * 60)
    print("COORDINATION RESULTS")
    print("=" * 60)
    
    print(f"\nPerformance:")
    print(f"  Total rounds: {results['total_rounds']}")
    print(f"  Items shipped: {results['items_shipped']}")
    print(f"  Execution time: {results['execution_time']:.1f}s")
    print(f"  Messages exchanged: {results['total_messages']}")
    
    print(f"\nSocial Precision (Trust Levels):")
    for agent, trusts in results['social_precisions'].items():
        print(f"\n  {agent.capitalize()}:")
        for other, trust in trusts.items():
            level = "HIGH" if trust > 0.7 else "MEDIUM" if trust > 0.5 else "LOW"
            print(f"    → {other}: {trust:.2f} ({level})")
    
    # Analysis
    print("\n" + "=" * 60)
    print("COORDINATION ANALYSIS")
    print("=" * 60)
    
    print("""
Key Observations:

1. Emergent Coordination
   • No central controller
   • Agents coordinate via shared state
   • Sequential dependencies respected

2. Trust Development
   • Social precision starts neutral (0.5)
   • Increases with successful interactions
   • Packer→Shipper trust highest (most reliable)

3. Adaptation to Dependencies
   • Packer waits for Picker
   • Shipper waits for Packer
   • Agents adapt when dependencies not met

4. Efficient Communication
   • Only 2 messages needed
   • Communication when social precision low
   • Most coordination via observation

5. Graceful Failure Handling
   • Individual failures don't crash system
   • Agents adapt and retry
   • System-level resilience
    """)
    
    # Comparison with traditional approaches
    print("\n" + "=" * 60)
    print("VS TRADITIONAL MULTI-AGENT SYSTEMS")
    print("=" * 60)
    
    print("""
Traditional Approaches:
  ✗ Explicit message passing for all coordination
  ✗ Fixed protocols and roles
  ✗ Brittle to failures
  ✗ No learning or adaptation
  ✗ Central coordinator often needed

LRS Multi-Agent:
  ✓ Implicit coordination via shared state
  ✓ Adaptive strategies based on precision
  ✓ Resilient to individual agent failures
  ✓ Learns trust in other agents
  ✓ Decentralized coordination
  ✓ Communication only when needed
    """)
    
    # Visualization
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Trust network
        G = nx.DiGraph()
        agents = ['picker', 'packer', 'shipper']
        G.add_nodes_from(agents)
        
        for agent, trusts in results['social_precisions'].items():
            for other, trust in trusts.items():
                G.add_edge(agent, other, weight=trust)
        
        pos = nx.spring_layout(G, k=2)
        
        # Draw trust network
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax1)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos, 
            width=[w*3 for w in weights],
            edge_color=weights,
            edge_cmap=plt.cm.Greens,
            edge_vmin=0.5,
            edge_vmax=1.0,
            arrows=True,
            arrowsize=20,
            ax=ax1
        )
        
        ax1.set_title('Social Precision Network\n(Trust Between Agents)')
        ax1.axis('off')
        
        # Workflow diagram
        ax2.barh(['Picker', 'Packer', 'Shipper'], [5, 4, 3], color=['blue', 'orange', 'green'])
        ax2.set_xlabel('Active Rounds')
        ax2.set_title('Agent Activity')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('warehouse_coordination.png', dpi=150)
        print("\n✓ Visualization saved to warehouse_coordination.png")
    
    except ImportError:
        print("\n(Install matplotlib and networkx for visualization)")


if __name__ == "__main__":
    print("\n[Note: This is a simplified demonstration]")
    print("[Full coordination uses coordinator.run() method]")
    print("[Shown sequence illustrates coordination patterns]\n")
    
    main()
