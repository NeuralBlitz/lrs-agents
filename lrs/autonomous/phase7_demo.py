# Phase 7: Autonomous Code Generation Demo
# ========================================
# Comprehensive demonstration of AI-powered application creation

"""
PHASE 7 DEMO: AUTONOMOUS CODE GENERATION
========================================

This demo showcases the revolutionary autonomous code generation capabilities
of the OpenCode ↔ LRS-Agents Cognitive AI Hub.

Features Demonstrated:
- Natural language application creation
- Multi-language code generation (Python, JavaScript)
- Quality assurance and validation
- Complete application packages with all files
- Real-time performance metrics

Examples Include:
1. REST API for task management
2. Web dashboard with charts
3. CLI tool for data processing
4. Authentication system
5. Full-stack e-commerce platform
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the autonomous code generation system
from lrs_agents.lrs.autonomous.phase7_autonomous_code_generation import AutonomousCodeGenerator


class AutonomousCodeGenerationDemo:
    """Comprehensive demo of autonomous code generation capabilities."""

    def __init__(self):
        self.generator = AutonomousCodeGenerator()
        self.demo_results = []

    async def run_full_demo(self):
        """Run the complete autonomous code generation demonstration."""
        print("🚀 PHASE 7: AUTONOMOUS CODE GENERATION DEMO")
        print("🧠 OpenCode ↔ LRS-Agents Cognitive AI Hub")
        print("=" * 60)

        # Demo scenarios
        scenarios = [
            {
                "title": "Task Management REST API",
                "description": "Create a REST API for managing tasks with categories, priorities, and due dates using Python FastAPI",
                "category": "API Development",
            },
            {
                "title": "Sales Dashboard Web App",
                "description": "Build a web dashboard for visualizing sales data with interactive charts and filtering using JavaScript",
                "category": "Web Application",
            },
            {
                "title": "Data Processing CLI Tool",
                "description": "Make a command-line tool for processing CSV files, applying transformations, and generating reports using Python",
                "category": "CLI Utility",
            },
            {
                "title": "User Authentication System",
                "description": "Develop a secure user authentication system with JWT tokens, registration, and login using Node.js",
                "category": "Security System",
            },
            {
                "title": "E-commerce Product Catalog",
                "description": "Create a full-stack e-commerce product catalog with shopping cart and checkout using React and Express",
                "category": "Full-Stack Application",
            },
        ]

        print(f"🎯 Running {len(scenarios)} autonomous code generation scenarios...")
        print()

        for i, scenario in enumerate(scenarios, 1):
            print(f"📝 Scenario {i}: {scenario['title']}")
            print(f"📂 Category: {scenario['category']}")
            print(f"💬 Description: {scenario['description']}")
            print("-" * 40)

            start_time = asyncio.get_event_loop().time()

            try:
                # Generate application
                result = await self.generator.generate_from_description(
                    scenario["description"]
                )

                generation_time = asyncio.get_event_loop().time() - start_time

                # Store results
                demo_result = {
                    "scenario": scenario,
                    "result": result,
                    "generation_time": generation_time,
                    "timestamp": datetime.now().isoformat(),
                }
                self.demo_results.append(demo_result)

                # Display results
                await self.display_generation_results(scenario, result, generation_time)

            except Exception as e:
                print(f"❌ Generation failed: {str(e)}")
                print()

        # Final summary
        await self.display_demo_summary()

    async def display_generation_results(
        self, scenario: Dict[str, Any], result: Dict[str, Any], generation_time: float
    ):
        """Display the results of application generation."""
        req = result["requirements"]
        app = result["application"]
        quality = result["quality_report"]
        stats = result["generation_stats"]

        print("✅ Application Generated Successfully!")
        print(f"🏗️  Name: {req.name}")
        print(f"💻 Language: {req.language}")
        print(f"🛠️  Framework: {req.framework or 'None'}")
        print(f"📊 Complexity: {req.complexity}")
        print(f"📁 Files Generated: {stats['files_generated']}")
        print(f"⭐ Quality Score: {quality['overall_score']:.1f}%")
        print(f"⚡ Generation Time: {generation_time:.2f} seconds")
        print()

        # Features extracted
        print("🔧 Extracted Features:")
        for feature in req.features:
            print(f"  • {feature}")
        print()

        # Files generated
        print("📄 Generated Files:")
        for filename in app.files.keys():
            lines = len(app.files[filename].split("\n"))
            print(f"  • {filename} ({lines} lines)")
        print()

        # Quality assessment
        if quality["issues"]:
            print("⚠️  Quality Issues:")
            for issue in quality["issues"][:3]:  # Show first 3
                print(f"  • {issue}")
            if len(quality["issues"]) > 3:
                print(f"  • ... and {len(quality['issues']) - 3} more")
        else:
            print("✅ No quality issues found!")

        if quality["recommendations"]:
            print("💡 Recommendations:")
            for rec in quality["recommendations"][:3]:  # Show first 3
                print(f"  • {rec}")
        print()

        # Show sample code
        main_files = [
            f
            for f in app.files.keys()
            if f.endswith((".py", ".js")) and "test" not in f.lower()
        ]
        if main_files:
            main_file = main_files[0]
            print(f"📝 Sample Code from {main_file}:")
            code_lines = app.files[main_file].split("\n")[:15]  # First 15 lines
            for i, line in enumerate(code_lines, 1):
                print("2d")
            if len(app.files[main_file].split("\n")) > 15:
                print(
                    f"  ... ({len(app.files[main_file].split(chr(10))) - 15} more lines)"
                )
            print()

    async def display_demo_summary(self):
        """Display comprehensive demo summary."""
        print("🎉 PHASE 7 DEMO COMPLETE!")
        print("=" * 60)

        total_scenarios = len(self.demo_results)
        successful_generations = len([r for r in self.demo_results if "result" in r])

        print(f"📊 Demo Statistics:")
        print(f"  • Total Scenarios: {total_scenarios}")
        print(f"  • Successful Generations: {successful_generations}")
        print(
            f"  • Success Rate: {(successful_generations / total_scenarios) * 100:.1f}%"
        )

        if self.demo_results:
            avg_quality = (
                sum(
                    r["result"]["quality_report"]["overall_score"]
                    for r in self.demo_results
                    if "result" in r
                )
                / successful_generations
            )
            avg_time = (
                sum(r["generation_time"] for r in self.demo_results)
                / successful_generations
            )

            print(f"  • Average Quality Score: {avg_quality:.1f}%")
            print(f"  • Average Generation Time: {avg_time:.2f} seconds")

        print()
        print("🚀 Key Achievements Demonstrated:")
        print("  ✅ Natural Language Processing - Complex requirement understanding")
        print("  ✅ Multi-Language Code Generation - Python & JavaScript support")
        print("  ✅ Complete Application Packages - All files, configs, documentation")
        print("  ✅ Quality Assurance - Automated validation and testing")
        print("  ✅ Cognitive Architecture - AI-powered intelligence and reasoning")
        print("  ✅ Performance Optimization - Fast, efficient code generation")
        print()

        print("🧠 Cognitive AI Capabilities:")
        print("  • Understands complex application requirements")
        print("  • Generates production-ready code with best practices")
        print("  • Provides intelligent quality assessment")
        print("  • Creates complete deployment packages")
        print("  • Supports multiple programming paradigms")
        print()

        print("🎯 Next Steps:")
        print("  1. Try the web interface: python phase7_web_interface.py")
        print("  2. Explore generated applications in the 'generated_apps/' directory")
        print("  3. Customize the templates for your specific needs")
        print("  4. Integrate with your CI/CD pipelines")
        print()

        print("💡 Example Use Cases:")
        print("  • Rapid prototyping of new application ideas")
        print("  • Automated code generation for repetitive tasks")
        print("  • Learning tool for understanding different frameworks")
        print("  • Foundation for custom application builders")
        print("  • AI-assisted software development workflows")
        print()

        # Save demo results
        await self.save_demo_results()

    async def save_demo_results(self):
        """Save demo results to file."""
        results_file = "phase7_demo_results.json"

        results_data = {
            "demo_timestamp": datetime.now().isoformat(),
            "total_scenarios": len(self.demo_results),
            "successful_generations": len(
                [r for r in self.demo_results if "result" in r]
            ),
            "results": [],
        }

        for result in self.demo_results:
            result_data = {
                "scenario": result["scenario"],
                "generation_time": result["generation_time"],
                "timestamp": result["timestamp"],
            }

            if "result" in result:
                r = result["result"]
                result_data.update(
                    {
                        "success": True,
                        "application_name": r["application"].name,
                        "language": r["generation_stats"]["language"],
                        "files_generated": r["generation_stats"]["files_generated"],
                        "quality_score": r["quality_report"]["overall_score"],
                        "complexity": r["requirements"].complexity,
                    }
                )
            else:
                result_data["success"] = False

            results_data["results"].append(result_data)

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"💾 Demo results saved to {results_file}")

    async def run_quick_demo(self):
        """Run a quick demonstration with one example."""
        print("⚡ Quick Autonomous Code Generation Demo")
        print("=" * 40)

        description = "Create a simple REST API for managing a todo list with add, list, and complete functionality using Python"

        print(f"📝 Generating: {description}")
        print()

        start_time = asyncio.get_event_loop().time()
        result = await self.generator.generate_from_description(description)
        generation_time = asyncio.get_event_loop().time() - start_time

        req = result["requirements"]
        app = result["application"]

        print("✅ Generated Todo List API!")
        print(f"🏗️  Name: {req.name}")
        print(f"💻 Language: {req.language}")
        print(f"📁 Files: {len(app.files)}")
        print(f"⚡ Time: {generation_time:.2f}s")
        print()

        # Show main file content
        main_file = next((f for f in app.files.keys() if f.endswith(".py")), None)
        if main_file:
            print(f"📄 Main file ({main_file}):")
            lines = app.files[main_file].split("\n")[:20]
            for i, line in enumerate(lines, 1):
                print("2d")
            print()

        print("🎉 Demo complete! Run the full demo with: python phase7_demo.py --full")


async def main():
    """Main demo entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7: Autonomous Code Generation Demo"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full comprehensive demo"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick demo")
    parser.add_argument(
        "--description", type=str, help="Custom description to generate"
    )

    args = parser.parse_args()

    demo = AutonomousCodeGenerationDemo()

    if args.description:
        # Generate custom application
        print(f"🎯 Generating application from: {args.description}")
        result = await demo.generator.generate_from_description(args.description)
        req = result["requirements"]
        app = result["application"]
        print(f"✅ Generated {req.name} ({req.language}) with {len(app.files)} files!")

    elif args.full:
        # Run full demo
        await demo.run_full_demo()

    else:
        # Run quick demo by default
        await demo.run_quick_demo()


if __name__ == "__main__":
    asyncio.run(main())
