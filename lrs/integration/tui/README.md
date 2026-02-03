# LRS-Agents TUI Integration

Comprehensive bidirectional integration between opencode TUI and LRS-Agents, enabling real-time monitoring, control, and coordination of AI agents.

## Overview

This integration provides:

- **Real-time WebSocket communication** for live agent monitoring
- **REST API** for programmatic agent control
- **Bidirectional state synchronization** between TUI and LRS
- **Precision visualization** with confidence indicators
- **Multi-agent coordination** with TUI awareness
- **Extensible plugin architecture** for custom functionality
- **Comprehensive configuration management**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TUI Integration Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ TUI Bridge  │  │ State Mirror │  │ Precision Mapper│  │
│  │             │  │              │  │                 │  │
│  │ WebSocket   │  │ Bi-direction │  │ Confidence      │  │
│  │ REST API    │  │ Sync         │  │ Visualization   │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Tool Lens   │  │ Multi-Agent  │  │ Plugin Manager  │  │
│  │ Integration │  │ Coordinator  │  │                 │  │
│  │             │  │              │  │ Extensions      │  │
│  │ TUI Tool    │  │ Dashboard    │  │ Custom Widgets  │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    LRS-Agents Core                         │
│  Tool Registry │ Shared State │ Active Inference │ Agents   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Setup

```python
import asyncio
from lrs.integration.tui import TUIBridge
from lrs.integration.tui.config import TUIConfigManager

# Load configuration
config_manager = TUIConfigManager.from_file("config.yaml")
config = config_manager.get_config()

# Create TUI Bridge
bridge = TUIBridge(
    tool_registry=tool_registry,
    shared_state=shared_state,
    config=config
)

# Start server
asyncio.run(bridge.start())
```

### 2. WebSocket Connection

```javascript
// Connect to agent state stream
const ws = new WebSocket('ws://localhost:8000/ws/agents/agent_1/state');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Agent update:', data);
};

// Subscribe to precision updates
const precisionWs = new WebSocket('ws://localhost:8000/ws/precision');
```

### 3. REST API Usage

```python
import requests

# List agents
response = requests.get('http://localhost:8001/api/v1/agents')
agents = response.json()

# Create new agent
agent_data = {
    'agent_id': 'new_agent',
    'agent_type': 'lrs',
    'config': {'model': 'gpt-4'},
    'tools': ['search', 'analyze']
}
response = requests.post('http://localhost:8001/api/v1/agents', json=agent_data)

# Execute tool
tool_data = {
    'tool_name': 'search',
    'parameters': {'query': 'test query'}
}
response = requests.post(
    f'http://localhost:8001/api/v1/agents/{agent_id}/tools/execute',
    json=tool_data
)
```

## Key Components

### TUI Bridge

Core component providing:
- WebSocket and REST endpoints
- Event routing and broadcasting
- Background task management
- Health monitoring

```python
bridge = TUIBridge(
    tool_registry=tool_registry,
    shared_state=shared_state,
    coordinator=coordinator,
    config=config
)

# Start with custom host/port
await bridge.start(host="0.0.0.0", port=8080)
```

### State Mirror

Maintains bidirectional synchronization:

```python
# Sync precision to TUI
await bridge.state_mirror.sync_precision_to_tui('agent_1', {
    'value': 0.85,
    'alpha': 8.5,
    'beta': 1.5
})

# Get TUI state
tui_state = bridge.state_mirror.get_tui_state('agent_1')

# Handle conflict resolution
conflict_resolver = ConflictResolution(strategy='tui_wins')
state_mirror = TUIStateMirror(shared_state, bridge, conflict_resolver)
```

### Precision Mapper

Transforms mathematical precision to intuitive visualizations:

```python
mapper = TUIPrecisionMapper()

# Map precision to confidence
confidence = mapper.precision_to_confidence({'value': 0.8})
print(confidence.level)      # ConfidenceLevel.HIGH
print(confidence.score)      # 80.0
print(confidence.color)      # '#22c55e'

# Generate adaptation alerts
alert = mapper.adaptation_to_tui_alert({
    'agent_id': 'agent_1',
    'precision_before': 0.7,
    'precision_after': 0.3
})
print(alert.severity.value)  # 'error'
print(alert.suggested_actions)
```

### Multi-Agent Coordinator

Enhanced coordination with TUI awareness:

```python
coordinator = TUIMultiAgentCoordinator(bridge, shared_state, tool_registry)

# Register TUI-aware agent
config = TUIAgentConfig(
    agent_id='agent_1',
    agent_type='lrs',
    tui_panel='main',
    dashboard_metrics=['precision', 'status'],
    coordination_group='analysis_team'
)
coordinator.register_tui_agent(config)

# Coordinate agents via TUI
result = await coordinator.coordinate_via_tui(
    agent_ids=['agent_1', 'agent_2'],
    coordination_type='collaborative_task',
    data={'goal': 'analyze_dataset'},
    user_initiated=True
)
```

### Plugin System

Extensible architecture for custom functionality:

```python
# Create custom visualization plugin
class CustomChartPlugin(VisualizationPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom_chart",
            version="1.0.0",
            description="Custom chart visualization",
            author="Your Name",
            plugin_type=PluginType.VISUALIZATION
        )
    
    async def render(self, data, context):
        return {
            'type': 'custom_chart',
            'data': data,
            'config': self.config
        }

# Load plugin
plugin_manager = TUIPluginManager()
await plugin_manager.load_plugin("plugins.CustomChartPlugin")
```

## WebSocket Events

### Agent State Updates

```json
{
    "type": "state_update",
    "agent_id": "agent_1",
    "data": {
        "precision": {"value": 0.85},
        "status": "active",
        "current_task": "data_analysis"
    },
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Precision Changes

```json
{
    "type": "precision_update",
    "agent_id": "agent_1",
    "precision": {"value": 0.85, "confidence": "high"},
    "confidence": {
        "level": "high",
        "score": 85.0,
        "color": "#22c55e"
    },
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Tool Executions

```json
{
    "type": "tool_execution",
    "agent_id": "agent_1",
    "tool": "search_tool",
    "success": true,
    "prediction_error": 0.1,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Adaptation Events

```json
{
    "type": "adaptation",
    "agent_id": "agent_1",
    "alert": {
        "severity": "warning",
        "title": "Precision Drop",
        "message": "Agent confidence decreased from 0.7 to 0.3",
        "suggested_actions": ["monitor_closely", "check_environment"]
    },
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## REST API Endpoints

### Agent Management
- `GET /api/v1/agents` - List all agents
- `POST /api/v1/agents` - Create new agent
- `GET /api/v1/agents/{id}` - Get agent details
- `PUT /api/v1/agents/{id}/state` - Update agent state
- `DELETE /api/v1/agents/{id}` - Delete agent

### State and Precision
- `GET /api/v1/agents/{id}/precision` - Get agent precision
- `GET /api/v1/agents/{id}/tools` - Get available tools
- `GET /api/v1/agents/{id}/history` - Get execution history

### Tool Operations
- `POST /api/v1/agents/{id}/tools/execute` - Execute tool
- `GET /api/v1/tools` - List all available tools

### TUI Integration
- `POST /api/v1/tui/interaction` - Handle TUI interactions
- `GET /api/v1/tui/events` - Stream TUI events (SSE)

### System Information
- `GET /api/v1/system/info` - Get system information
- `GET /api/v1/system/health` - Health check

## Configuration

### Environment Variables

```bash
# WebSocket Configuration
LRS_TUI_WEBSOCKET_PORT=8000
LRS_TUI_WEBSOCKET_HOST=0.0.0.0

# REST API Configuration
LRS_TUI_REST_PORT=8001
LRS_TUI_ENABLE_CORS=true

# State Synchronization
LRS_TUI_SYNC_INTERVAL=1.0
LRS_TUI_CONFLICT_RESOLUTION=tui_wins

# Monitoring
LRS_TUI_LOG_LEVEL=info
LRS_TUI_ENABLE_METRICS=true
```

### Configuration File

See `config.yaml` for complete configuration options.

## Security Considerations

### Authentication
- API key authentication for REST endpoints
- WebSocket connection validation
- Plugin permission system

### Data Validation
- Input schema validation for all endpoints
- Type checking and sanitization
- Rate limiting and DoS protection

### Access Control
- Configurable allowed origins for CORS
- Plugin permission management
- Agent access restrictions

## Performance

### Optimization Features
- Event buffering and batching
- Connection pooling
- Background task management
- Memory-efficient state tracking

### Scaling Considerations
- Horizontal scaling support
- Load balancing ready
- Database persistence options
- Monitoring and alerting

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failures**
   - Check firewall settings
   - Verify port availability
   - Review CORS configuration

2. **State Synchronization Delays**
   - Increase sync interval in config
   - Check network latency
   - Monitor system resources

3. **Plugin Loading Errors**
   - Verify plugin dependencies
   - Check permission settings
   - Review plugin logs

### Debug Mode

```python
# Enable debug logging
config_manager.update_config({
    'debug': True,
    'monitoring': {'log_level': 'debug'}
})

# Enable detailed metrics
config_manager.update_config({
    'monitoring': {
        'enable_metrics': True,
        'enable_tracing': True
    }
})
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev,test]"

# Run all tests
pytest tests/ -v --cov=lrs.integration.tui

# Run specific test suite
pytest tests/test_bridge.py -v
```

### Creating Plugins

See `plugins.py` for plugin development guidelines and examples.

## License

This integration is part of LRS-Agents and follows the same MIT license.

## Contributing

Please see the main LRS-Agents contributing guidelines for information on how to contribute to this integration.