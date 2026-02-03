#!/usr/bin/env python3
"""
OpenCode LRS Command Integration

This module provides LRS-enhanced commands for OpenCode CLI.
Run this to add LRS capabilities to your OpenCode installation.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Import our LRS integration
    from lrs_agents.lrs.opencode.lrs_opencode_integration import opencode_lrs_command
except ImportError as e:
    print(f"Error importing LRS integration: {e}")
    sys.exit(1)


def integrate_with_opencode():
    """Integrate LRS commands with OpenCode CLI."""

    print("🔗 Integrating LRS-Agents with OpenCode CLI...")
    print("=" * 50)

    # Check if OpenCode is available
    try:
        result = subprocess.run(
            ["opencode", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("❌ OpenCode not found. Please install OpenCode first.")
            return False
        print("✅ OpenCode CLI detected")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ OpenCode CLI not found in PATH")
        return False

    # Test LRS functionality
    try:
        from lrs_agents.lrs.opencode.lrs_opencode_integration import ActiveInferenceAnalyzer

        analyzer = ActiveInferenceAnalyzer()
        print("✅ LRS Active Inference Analyzer loaded")

        from lrs_agents.lrs.opencode.lrs_opencode_integration import HierarchicalPlanner

        planner = HierarchicalPlanner()
        print("✅ LRS Hierarchical Planner loaded")

        from lrs_agents.lrs.opencode.lrs_opencode_integration import PolicyEvaluator

        evaluator = PolicyEvaluator()
        print("✅ LRS Policy Evaluator loaded")

    except Exception as e:
        print(f"❌ Error loading LRS components: {e}")
        return False

    print("\n🎯 LRS Commands Available:")
    print("   opencode lrs analyze <path>     - Active Inference codebase analysis")
    print("   opencode lrs refactor <file>    - Precision-guided refactoring analysis")
    print("   opencode lrs plan <description> - Hierarchical development planning")
    print("   opencode lrs evaluate <task> <strategies...> - Strategy evaluation")
    print("   opencode lrs stats              - LRS system statistics")
    print("\n💡 Example Usage:")
    print("   opencode lrs analyze .")
    print("   opencode lrs plan 'Build a REST API for user management'")
    print(
        "   opencode lrs evaluate 'Implement authentication' 'JWT tokens' 'OAuth2' 'Session-based'"
    )

    return True


def demo_lrs_integration():
    """Run a comprehensive demo of LRS integration."""

    print("\n🎪 LRS Integration Demo")
    print("=" * 25)

    # Demo 1: Codebase Analysis
    print("\n1️⃣  Active Inference Codebase Analysis:")
    print("-" * 40)
    try:
        from lrs_agents.lrs.opencode.lrs_opencode_integration import ActiveInferenceAnalyzer

        analyzer = ActiveInferenceAnalyzer()
        result = analyzer.analyze_codebase(".")

        print(f"   📊 Files analyzed: {result['total_files']}")
        print(f"   📝 Total lines: {result['total_lines']}")
        print(f"   🧠 Average complexity: {result['avg_complexity']:.2f}")
        print(f"   🎯 Free Energy G: {result['free_energy']:.3f}")
        print(
            f"   💡 Recommendation: {result['recommendations'][0] if result['recommendations'] else 'Analysis complete'}"
        )
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Demo 2: Hierarchical Planning
    print("\n2️⃣  Hierarchical Development Planning:")
    print("-" * 42)
    try:
        from lrs_agents.lrs.opencode.lrs_opencode_integration import HierarchicalPlanner

        planner = HierarchicalPlanner()
        plan = planner.create_development_plan(
            "Create a web dashboard for data visualization"
        )

        print(f"   🎯 Abstract goals: {len(plan['abstract_goals'])}")
        print(f"   📝 Planning tasks: {len(plan['planning_tasks'])}")
        print(f"   ⚙️  Execution steps: {len(plan['execution_steps'])}")
        print(f"   📊 Plan quality: {plan['plan_quality']}")
        print(f"   ⚠️  Risk level: {plan['risk_assessment']['overall_risk_level']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Demo 3: Strategy Evaluation
    print("\n3️⃣  Development Strategy Evaluation:")
    print("-" * 40)
    try:
        from lrs_agents.lrs.opencode.lrs_opencode_integration import PolicyEvaluator

        evaluator = PolicyEvaluator()
        strategies = [
            "Agile with 2-week sprints",
            "TDD with comprehensive tests",
            "MVP with rapid prototyping",
        ]
        evaluation = evaluator.evaluate_strategies(
            "Implement real-time chat feature", strategies
        )

        print(f"   📊 Strategies evaluated: {len(evaluation['evaluations'])}")
        recommended = evaluation["recommended_strategy"]
        print(f"   🏆 Recommended: {recommended['strategy']}")
        print(f"   📈 Confidence: {evaluation['selection_confidence']:.1%}")
        print(f"   ⚡ Free Energy G: {recommended['free_energy']:.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n🎉 LRS Integration Demo Complete!")
    print("💡 Use 'opencode lrs <command>' for interactive usage")


def create_opencode_plugin():
    """Create an OpenCode plugin file for LRS integration."""

    plugin_content = '''#!/usr/bin/env python3
"""
OpenCode LRS Plugin

Integrates LRS-Agents capabilities into OpenCode CLI.
"""

import sys
from pathlib import Path

# Add plugin directory to path
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

def register_commands():
    """Register LRS commands with OpenCode."""

    try:
        from lrs_agents.lrs.opencode.lrs_opencode_integration import opencode_lrs_command

        return {
            'lrs': opencode_lrs_command
        }
    except ImportError as e:
        print(f"Warning: LRS integration not available: {e}")
        return {}

# Export for OpenCode plugin system
commands = register_commands()
'''

    plugin_path = current_dir / "opencode_lrs_plugin.py"
    try:
        with open(plugin_path, "w") as f:
            f.write(plugin_content)
        print(f"✅ OpenCode plugin created: {plugin_path}")
        print("💡 To use: Place this file in your OpenCode plugins directory")
    except Exception as e:
        print(f"❌ Failed to create plugin: {e}")


if __name__ == "__main__":
    print("🧠 OpenCode LRS Integration Setup")
    print("=" * 35)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "demo":
            demo_lrs_integration()
        elif command == "plugin":
            create_opencode_plugin()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python setup_lrs_integration.py [demo|plugin]")
    else:
        # Default: full integration setup
        success = integrate_with_opencode()

        if success:
            print("\n🚀 Integration Complete!")
            print("💡 Try: opencode lrs analyze .")
            print("💡 Try: python setup_lrs_integration.py demo")
            print("💡 Try: python setup_lrs_integration.py plugin")
        else:
            print("\n❌ Integration Failed")
            sys.exit(1)
