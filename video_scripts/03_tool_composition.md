# Video 3: "Building Complex Policies with Tool Composition" (7 minutes)

## Script

[OPENING - 0:00-0:20]
VISUAL: Code editor showing simple tool execution
VOICEOVER:
"In tutorial 2, we saw how agents track precision. But real tasks require 
multiple tools working together. This is where tool composition comes in—
the ability to chain tools into complex policies mathematically."

[PROBLEM STATEMENT - 0:20-1:00]
VISUAL: Diagram showing failed tool chain
VOICEOVER:
"Imagine you need to: fetch data from an API, parse the JSON, validate the 
schema, and write to a database. That's four tools. If any one fails, 
standard agents either crash or retry blindly. LRS agents compose tools 
as categorical morphisms—mathematical functions with automatic error 
propagation."

DIAGRAM:
API → Parse → Validate → Write ↓ ↓ ↓ ↓ Error Error Error Error ↓ ↓ ↓ ↓ Precision drops at failure point

[THEORY: LENSES - 1:00-2:00]
VISUAL: Mathematical diagram of lens composition
VOICEOVER:
"In LRS, tools are implemented as lenses from category theory. A lens has 
two operations: 'get'—which executes the tool, and 'set'—which updates 
the agent's belief state. The key insight: lenses compose automatically."

EQUATION ON SCREEN:
tool_a >> tool_b >> tool_c = compose(tool_a, compose(tool_b, tool_c))

With automatic error handling at each step.

ANIMATION:
- Show data flowing through composed lens
- Highlight belief state updates at each stage
- Show error propagating backward when tool fails

[CODE DEMO - 2:00-4:00]
VISUAL: Jupyter notebook with live coding
VOICEOVER:
"Let's build a real pipeline. We'll fetch weather data, parse the JSON, 
convert units, and generate a report. Watch how composition works."

CODE:
```python
from lrs.core.lens import ToolLens, ExecutionResult

class WeatherAPITool(ToolLens):
    def get(self, state):
        data = requests.get(f"api.weather.com/{state['city']}")
        return ExecutionResult(
            success=data.ok,
            value=data.json(),
            error=None if data.ok else "API failed",
            prediction_error=0.0 if data.ok else 0.9
        )
    
    def set(self, state, observation):
        return {**state, 'raw_data': observation}

class JSONParserTool(ToolLens):
    def get(self, state):
        try:
            parsed = json.loads(state['raw_data'])
            return ExecutionResult(True, parsed, None, 0.1)
        except:
            return ExecutionResult(False, None, "Parse error", 0.95)
    
    def set(self, state, observation):
        return {**state, 'parsed_data': observation}

class UnitConverterTool(ToolLens):
    def get(self, state):
        temp_f = state['parsed_data']['temperature']
        temp_c = (temp_f - 32) * 5/9
        return ExecutionResult(True, temp_c, None, 0.0)
    
    def set(self, state, observation):
        return {**state, 'temp_celsius': observation}

# THE MAGIC: Compose with >>
pipeline = WeatherAPITool() >> JSONParserTool() >> UnitConverterTool()

# Execute entire pipeline
result = pipeline.get({'city': 'San Francisco'})
VOICEOVER: “Notice the >> operator. This isn’t just syntactic sugar—it’s mathematical composition with automatic error propagation.”

[FAILURE HANDLING - 4:00-5:00] VISUAL: Side-by-side comparison of standard vs LRS handling VOICEOVER: “What happens when a tool in the middle fails? Let’s break the JSON parser.”

CODE:

# Inject failure
class BrokenParser(ToolLens):
    def get(self, state):
        return ExecutionResult(False, None, "Parser crashed", 0.95)

broken_pipeline = WeatherAPITool() >> BrokenParser() >> UnitConverterTool()
result = broken_pipeline.get({'city': 'London'})

print(f"Success: {result.success}")  # False
print(f"Error: {result.error}")      # "Parser crashed"
print(f"Prediction error: {result.prediction_error}")  # 0.95
VOICEOVER: “The pipeline short-circuits at the failure point. The error propagates backward, precision drops, and the agent can try an alternative pipeline. This is compositional resilience.”

[NATURAL TRANSFORMATIONS - 5:00-6:00] VISUAL: Diagram showing tool registry with alternatives VOICEOVER: “But composition gets even more powerful with natural transformations— automatic fallbacks. You register alternative tools that satisfy the same schema.”

CODE:

from lrs.core.registry import ToolRegistry

registry = ToolRegistry()

# Register with alternatives
registry.register(
    JSONParserTool(),
    alternatives=["xml_parser", "yaml_parser"]
)
registry.register(XMLParserTool())
registry.register(YAMLParserTool())

# When JSON parser fails, registry automatically suggests XML parser
VOICEOVER: “This is categorical polymorphism—tools become interchangeable based on their input-output types, not their names.”

[PRACTICAL EXAMPLE - 6:00-6:45] VISUAL: Real-world data pipeline (API → DB) VOICEOVER: “Here’s a production pipeline: fetch from REST API, if that fails try GraphQL, parse the response, validate against schema, if validation fails try alternative parser, write to PostgreSQL, if that fails write to backup CSV file.”

CODE (FAST PLAYBACK):

pipeline = (
    RESTAPITool() 
    >> (JSONParserTool() | AlternativeParser())
    >> SchemaValidator()
    >> (PostgresWriter() | CSVWriter())
)
VOICEOVER: “Four tools, each with fallbacks. The | operator creates parallel alternatives. If the left side fails, the right side executes automatically.”

[CLOSING - 6:45-7:00] VISUAL: Summary slide with composition operators VOICEOVER: “To recap: Tools are lenses. The >> operator composes them. The | operator creates fallbacks. Errors propagate mathematically. And precision controls which alternatives to explore. This is compositional agency.”

[END SCREEN]
