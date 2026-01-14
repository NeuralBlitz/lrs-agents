"""
Quickstart: Create your first LRS agent in 5 minutes.

This example shows:
1. Creating simple tools
2. Building an LRS agent
3. Running a task with automatic adaptation
"""

from langchain_anthropic import ChatAnthropic
from lrs import create_lrs_agent
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.monitoring.tracker import LRSStateTracker


# Step 1: Define custom tools
class WeatherAPITool(ToolLens):
    """Fetch weather data (simulated)"""
    
    def __init__(self):
        super().__init__(
            name="weather_api",
            input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
            output_schema={'type': 'object'}
        )
    
    def get(self, state):
        """Simulate API call"""
        self.call_count += 1
        city = state.get('city', 'Unknown')
        
        # Simulate occasional API failures
        import random
        if random.random() < 0.2:  # 20% failure rate
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error="API timeout",
                prediction_error=0.9
            )
        
        # Success
        return ExecutionResult(
            success=True,
            value={
                'city': city,
                'temperature': 72,
                'conditions': 'sunny'
            },
            error=None,
            prediction_error=0.1
        )
    
    def set(self, state, observation):
        """Update state with weather data"""
        return {**state, 'weather_data': observation}


class CacheTool(ToolLens):
    """Check cache for weather data (fast, reliable)"""
    
    def __init__(self):
        super().__init__(
            name="cache_lookup",
            input_schema={'type': 'object'},
            output_schema={'type': 'object'}
        )
        self.cache = {
            'San Francisco': {'temperature': 65, 'conditions': 'foggy'},
            'New York': {'temperature': 55, 'conditions': 'rainy'}
        }
    
    def get(self, state):
        """Check cache"""
        self.call_count += 1
        city = state.get('city', 'Unknown')
        
        if city in self.cache:
            return ExecutionResult(
                success=True,
                value=self.cache[city],
                error=None,
                prediction_error=0.0  # Cache is deterministic
            )
        else:
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error="Not in cache",
                prediction_error=0.5
            )
    
    def set(self, state, observation):
        return {**state, 'weather_data': observation}


# Step 2: Create agent
def main():
    print("=" * 60)
    print("LRS-AGENTS QUICKSTART")
    print("=" * 60)
    
    # Initialize LLM
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    
    # Create tools
    tools = [
        WeatherAPITool(),
        CacheTool()
    ]
    
    # Create tracker for monitoring
    tracker = LRSStateTracker()
    
    # Build LRS agent
    agent = create_lrs_agent(
        llm=llm,
        tools=tools,
        preferences={
            'success': 5.0,      # Reward for successful execution
            'error': -3.0,       # Penalty for errors
            'step_cost': -0.1    # Small cost per step
        },
        tracker=tracker,
        use_llm_proposals=True  # Use LLM for policy generation
    )
    
    print("\n✓ Agent created with 2 tools:")
    print("  - weather_api: Fetch from API (fast but unreliable)")
    print("  - cache_lookup: Check cache (slower but reliable)")
    
    # Step 3: Run task
    print("\n" + "-" * 60)
    print("EXECUTING TASK: Get weather for San Francisco")
    print("-" * 60)
    
    result = agent.invoke({
        'messages': [{
            'role': 'user',
            'content': 'Get the current weather for San Francisco'
        }],
        'belief_state': {
            'city': 'San Francisco',
            'goal': 'get_weather'
        },
        'max_iterations': 10
    })
    
    # Step 4: Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    tool_history = result.get('tool_history', [])
    print(f"\nTotal steps: {len(tool_history)}")
    print(f"Adaptations: {result.get('adaptation_count', 0)}")
    
    print("\nExecution trace:")
    for i, entry in enumerate(tool_history, 1):
        status = "✓" if entry['success'] else "✗"
        print(f"  {i}. {status} {entry['tool']} "
              f"(error: {entry['prediction_error']:.2f})")
    
    # Final precision
    precision = result.get('precision', {})
    print(f"\nFinal precision:")
    print(f"  Execution: {precision.get('execution', 0):.3f}")
    print(f"  Planning:  {precision.get('planning', 0):.3f}")
    print(f"  Abstract:  {precision.get('abstract', 0):.3f}")
    
    # Weather data
    weather = result.get('belief_state', {}).get('weather_data')
    if weather:
        print(f"\n✓ Weather retrieved: {weather['temperature']}°F, {weather['conditions']}")
    
    # Tracker summary
    summary = tracker.get_summary()
    print(f"\nTracker summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Adaptations: {summary['total_adaptations']}")
    print(f"  Avg precision: {summary['avg_precision']:.3f}")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. LRS agents automatically adapt when tools fail
2. Precision tracks confidence in the world model
3. Low precision → explore alternatives
4. High precision → exploit known strategies
5. No manual error handling needed!
    """)


if __name__ == "__main__":
    main()
