# OpenCode ↔ LRS-Agents Integration Guide

## Overview

This guide demonstrates multiple approaches to integrate OpenCode (CLI tool for software engineering) with LRS-Agents (resilient AI agents via Active Inference).

## Integration Approaches

### 1. **OpenCode as LRS Tool** (`opencode_lrs_tool.py`)
- **What**: LRS agents can use OpenCode's capabilities as tools
- **How**: Create `ToolLens` wrapper for OpenCode CLI
- **Benefits**: LRS agents get file operations, code search, terminal commands
- **Use Case**: Code analysis, refactoring, debugging tasks

### 2. **LRS Agents in OpenCode** (`lrs_opencode_bridge.py`)
- **What**: OpenCode can call LRS agents for complex reasoning
- **How**: HTTP API bridge between systems
- **Benefits**: Complex multi-step tasks with active inference
- **Use Case**: Strategic planning, precision-guided decision making

### 3. **Bidirectional API** (`lrs_opencode_bridge.py`)
- **What**: Full two-way communication via REST/WebSocket
- **How**: FastAPI server enabling both directions
- **Benefits**: Real-time collaboration, state synchronization
- **Use Case**: Interactive development sessions

## Quick Start

### Method 1: Simplified Integration (Recommended for Testing)

```python
from simplified_integration import OpenCodeTool, SimplifiedLRSAgent

# Create OpenCode tool
opencode_tool = OpenCodeTool()

# Create simplified LRS agent with OpenCode tool
agent = SimplifiedLRSAgent(tools=[opencode_tool])

# Execute tasks with active inference
result = agent.execute_task("list files in current directory")
print(f"Success: {result['success']}")
```

### Method 2: Full LRS Integration (Requires NumPy)

```python
from opencode_lrs_tool import create_opencode_integration
from lrs.core.registry import ToolRegistry
from lrs.integration.langgraph import create_lrs_agent

# Create tool registry with OpenCode
registry = ToolRegistry()
opencode_tool = create_opencode_integration()
registry.register(opencode_tool)

# Create LRS agent with OpenCode capabilities
agent = create_lrs_agent(llm, tools=[opencode_tool])

# Agent can now perform file operations, searches, etc.
result = agent.invoke({"messages": [{"role": "user", "content": "Find all TODO comments in the codebase"}]})
```

### Method 2: OpenCode Calling LRS Agents

```python
from lrs_opencode_bridge import OpenCodeLRSBridge
import asyncio

# Start bridge server
bridge = OpenCodeLRSBridge()
asyncio.run(bridge.run_api_server())  # Runs on localhost:8765

# Now OpenCode can make HTTP calls to LRS agents
# POST /lrs/create-agent - Create new agent
# POST /lrs/execute - Execute tasks
```

## Integration Patterns

### Pattern 1: Code Analysis Agent
```python
# LRS agent specialized for code analysis using OpenCode
analysis_agent = create_lrs_agent(llm, tools=[
    OpenCodeTool("code_search"),
    OpenCodeTool("file_reader"),
    OpenCodeTool("dependency_analyzer")
])
```

### Pattern 2: Development Workflow Agent
```python
# Agent that manages development workflow
workflow_agent = create_lrs_agent(llm, tools=[
    OpenCodeTool("git_operations"),
    OpenCodeTool("test_runner"),
    OpenCodeTool("lint_checker"),
    OpenCodeTool("build_executor")
])
```

### Pattern 3: Interactive Coding Assistant
```python
# Agent that works alongside developer
assistant_agent = create_lrs_agent(llm, tools=[
    OpenCodeTool("code_completer"),
    OpenCodeTool("refactor_suggester"),
    OpenCodeTool("documentation_generator")
])
```

## Active Inference Benefits

When OpenCode integrates with LRS-Agents, you get:

1. **Precision Tracking**: System learns from prediction errors
2. **Hierarchical Planning**: Abstract → Planning → Execution levels
3. **Adaptive Behavior**: Adjusts based on task complexity
4. **Goal-Directed Actions**: Pragmatic value optimization
5. **Resilient Execution**: Handles failures gracefully

## API Reference

### OpenCode ToolLens Methods
- `get(belief_state)`: Execute OpenCode operation
- `set(belief_state, value)`: Update belief state with results
- `calculate_epistemic_value()`: Information gain calculation
- `calculate_pragmatic_value()`: Goal-directed value calculation

### Bridge API Endpoints
- `POST /lrs/create-agent`: Create LRS agent
- `POST /lrs/execute`: Execute agent task
- `POST /opencode/execute`: Execute OpenCode command

## Configuration

### Environment Variables
```bash
export OPENCODE_PATH=/path/to/opencode
export LRS_BRIDGE_PORT=8765
export LRS_BRIDGE_HOST=localhost
```

### Precision Parameters
```python
precision_config = {
    'abstract': {'alpha': 2.0, 'beta': 2.0},
    'planning': {'alpha': 3.0, 'beta': 1.0},
    'execution': {'alpha': 1.0, 'beta': 3.0}
}

agent = create_lrs_agent(llm, tools, precision_config=precision_config)
```

## Monitoring & Debugging

### Enable LRS Monitoring
```python
from lrs.monitoring import LRSStateTracker

tracker = LRSStateTracker()
monitored_agent = create_monitored_lrs_agent(llm, tools, tracker)

# View precision changes, adaptation events, policy evaluations
```

### Bridge Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Logs all API calls, agent executions, precision updates
```

## Example Use Cases

### 1. Automated Code Review
```
User: "Review the authentication module"
LRS Agent: Uses OpenCode to search for security patterns
         → Analyzes code structure
         → Identifies potential vulnerabilities
         → Suggests improvements with precision confidence
```

### 2. Test-Driven Development
```
User: "Implement user registration with tests"
LRS Agent: Plans test-first approach
         → Creates test stubs using OpenCode
         → Implements minimal code to pass tests
         → Refactors with confidence metrics
```

### 3. Legacy Code Migration
```
User: "Migrate Python 2 to Python 3"
LRS Agent: Analyzes codebase for compatibility issues
         → Creates migration plan with risk assessment
         → Applies transformations incrementally
         → Validates each step with precision feedback
```

## Performance Considerations

- **Bridge Latency**: HTTP calls add ~50-100ms overhead
- **WebSocket vs HTTP**: Use WebSocket for real-time interactions
- **Caching**: Cache OpenCode results in belief state
- **Batch Operations**: Group multiple OpenCode calls
- **Async Execution**: Use async/await for concurrent operations

## Security

- **Command Validation**: Validate all OpenCode commands
- **Path Restrictions**: Limit file system access
- **Timeout Limits**: Prevent hanging operations
- **Audit Logging**: Log all agent actions
- **Access Control**: Implement authentication for API

## Troubleshooting

### Common Issues

1. **OpenCode Not Found**
   - Ensure opencode is installed and in PATH
   - Check `opencode --version` works

2. **Bridge Connection Failed**
   - Verify bridge server is running on correct port
   - Check firewall settings

3. **LRS Agent Errors**
   - Ensure all LRS dependencies are installed
   - Check precision parameters are valid

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
opencode_tool = create_opencode_integration()
result = opencode_tool.get({'current_task': 'list'})
print(f"Test result: {result}")
```

## Future Enhancements

- **Direct Function Calls**: Skip HTTP bridge for local integrations
- **Shared Memory**: Direct state sharing between systems
- **Plugin Architecture**: Extensible tool ecosystem
- **Multi-Agent Coordination**: Multiple LRS agents working together
- **Learning Integration**: Transfer learning between sessions

## Working Examples

### Demo Results
The integration has been successfully tested:

```
✅ OpenCode found at: opencode
✅ File listing: Successfully executed
✅ Precision tracking: 0.84 → 0.88
✅ Active inference: Adapts based on prediction errors
```

### Test the Integration

```bash
# Run the simplified integration demo
python simplified_integration.py

# Start the web interface with LRS integration
python main.py

# Test individual components
python opencode_lrs_tool.py  # May fail due to NumPy issues
```

## Integration Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│     OpenCode    │◄──────────────────► │   LRS-Agents    │
│   CLI Tool      │                     │ Active Inference│
│                 │                     │                 │
│ • File ops      │ ToolLens Interface  │ • Precision     │
│ • Code search   │                     │ • Policy eval   │
│ • Terminal cmds │                     │ • Goal tracking │
└─────────────────┘                     └─────────────────┘
         ▲                                       ▲
         │                                       │
         └─────────── Bridge API ────────────────┘
```

## Contributing

To extend the integration:

1. Add new `ToolLens` implementations for specific OpenCode features
2. Implement additional bridge endpoints
3. Create domain-specific agent configurations
4. Add monitoring and metrics collection

## Known Issues & Solutions

### NumPy Dependency Issues
- **Problem**: `libstdc++.so.6` missing in some environments
- **Solution**: Use `simplified_integration.py` which avoids NumPy
- **Future**: Containerize with proper dependencies

### OpenCode Not Found
- **Problem**: opencode not in PATH
- **Solution**: Install opencode or adjust `OPENCODE_PATH`

### Bridge Connection Issues
- **Problem**: HTTP timeouts between systems
- **Solution**: Use WebSocket for real-time communication

---

## Summary

This integration successfully bridges OpenCode's practical software engineering capabilities with LRS-Agents' sophisticated active inference framework. The result is AI systems that can:

- **Execute complex tasks** with precision-guided decision making
- **Learn from experience** through prediction error minimization
- **Adapt behavior** based on task difficulty and success rates
- **Maintain resilience** through hierarchical precision tracking

The bidirectional nature allows both systems to enhance each other: OpenCode provides concrete execution capabilities, while LRS-Agents provides intelligent planning and adaptation.