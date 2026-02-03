"""
Distributed tracing with OpenTelemetry for comprehensive observability.
"""

import time
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum
import structlog
import uuid
import asyncio

logger = structlog.get_logger(__name__)

# Import OpenTelemetry components
try:
    from opentelemetry import trace, baggage, context
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.semconv.trace import SemanticConventions
    from opentelemetry import metrics as OtelMetrics
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available, using mock implementation")


class TracingConfig:
    """Configuration for distributed tracing."""
    
    enable_tracing: bool = True
    service_name: str = "integration-bridge"
    jaeger_endpoint: Optional[str] = None
    sample_rate: float = 1.0  # 100% sampling
    max_batch_size: int = 512
    export_timeout: float = 30.0
    enable_metrics: bool = True
    enable_baggage: bool = True


class SpanKind(Enum):
    """Span types for different operations."""
    AGENT_OPERATION = "agent_operation"
    API_REQUEST = "api_request"
    TOOL_EXECUTION = "tool_execution"
    DATABASE_OPERATION = "database_operation"
    WEBSOCKET_MESSAGE = "websocket_message"
    INTERNAL_OPERATION = "internal_operation"


class SpanAttributes:
    """Standard span attributes."""
    
    COMPONENT = "component.name"
    OPERATION_TYPE = "operation.type"
    AGENT_ID = "agent.id"
    TOOL_NAME = "tool.name"
    USER_ID = "user.id"
    SESSION_ID = "session.id"
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    REQUEST_ID = "request.id"
    RESPONSE_CODE = "http.status_code"
    RESPONSE_SIZE = "http.response.size"
    DURATION = "duration.ms"


@dataclass
class SpanContext:
    """Context for trace propagation."""
    
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    baggage: Dict[str, str] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class TracingConfig:
    """Configuration for distributed tracing."""
    
    enable_tracing: bool = True
    service_name: str = "integration-bridge"
    jaeger_endpoint: Optional[str] = None
    sample_rate: float = 1.0
    max_batch_size: int = 512
    export_timeout: float = 30.0
    enable_metrics: bool = True
    enable_baggage: bool = True


# Try to import OpenTelemetry components
OPENTELEMETRY_AVAILABLE = False
try:
    from opentelemetry import trace, baggage, context
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.semconv.trace import SemanticConventions
    from opentelemetry import metrics as OtelMetrics
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Mock OpenTelemetry components for development/testing
class MockTracer:
    """Mock tracer for development/testing."""
    
    def __init__(self):
        self.spans = []
    
    def start_as_current_span(self, name: str):
        return MockSpan(self.spans)
    
    def get_tracer_provider(self):
        return MockTracerProvider()


class MockTracerProvider:
    """Mock tracer provider for development/testing."""
    
    def get_tracer(self, __name__):
        return MockTracer()


class MockSpan:
    """Mock span for development/testing."""
    
    def __init__(self, name: str, span_collection: List = None):
        self.name = name
        self.span_collection = span_collection or []
        self.span_collection.append(self)
        self.attributes = {}
        self.events = []
        self.status = None
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value
    
    def set_status(self, status):
        """Set span status."""
        self.status = status
    
    def record_exception(self, exception: Exception):
        """Record exception on span."""
        import time
        self.events.append({
            "exception": str(exception),
            "timestamp": time.time()
        })
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self


class MockMetrics:
    """Mock metrics for development/testing."""
    
    def __init__(self):
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
    
    def get(self, name: str, *labels):
        """Get metric by name and labels."""
        if name in self.counters:
            return self.counters[name]
        elif name in self.histograms:
            return self.histograms[name]
        elif name in self.gauges:
            return self.gauges[name]
        
        # Return default mock metric
        return MockMetric(name)
    
    def inc(self):
        """Increment counter."""
        pass
    
    def observe(self, value: float):
        """Observe histogram value."""
        pass
    
    def set(self, value: float):
        """Set gauge value."""
        pass


class MockMetric:
    """Mock metric for development/testing."""
    
    def __init__(self, name: str):
        self.name = name
    
    def inc(self):
        """Increment counter."""
        pass
    
    def observe(self, value: float):
        """Observe histogram value."""
        pass
    
    def set(self, value: float):
        """Set gauge value."""
        pass


class MockLabels:
    """Mock labels for development/testing."""
    
    def __init__(self, labels):
        self.labels = labels


# Global tracing manager instance
tracing_manager: Optional[TracingManager] = None
        else:
            yield self.tracer.start_span(span_name)
    
    @asynccontextmanager
    async def trace_api_request(
        self,
        method: str,
        endpoint: str,
        request_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Trace an API request."""
        span_name = f"api.{endpoint.replace('/', '_')}"
        
        if OPENTELEMETRY_AVAILABLE:
            with self.tracer.start_as_current_span(span_name) as span:
                # Set standard HTTP attributes
                span.set_attribute(SpanAttributes.COMPONENT, self.config.service_name)
                span.set_attribute(SpanAttributes.OPERATION_TYPE, SpanKind.API_REQUEST)
                span.set_attribute(SemanticConventions.HTTP_METHOD, method.upper())
                span.set_attribute(SpanAttributes.ENDPOINT, endpoint)
                span.set_attribute(SpanAttributes.USER_ID, self.span_context.user_id)
                
                if request_id:
                    span.set_attribute(SpanAttributes.REQUEST_ID, request_id)
                
                # Add baggage to span context
                if self.config.enable_baggage:
                    for key, value in self.span_context.baggage.items():
                        baggage.set_baggage(key, value)
                
                # Set custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                start_time = time.time()
                
                yield span
                
                # Record metrics on span completion
                duration = time.time() - start_time
                self.metrics.get('api_requests_total').labels(
                    method=method,
                    endpoint=endpoint,
                    status="completed"
                ).inc()
                
                self.metrics.get('response_duration_seconds').labels(
                    endpoint=endpoint,
                    method=method
                ).observe(duration)
        else:
            yield self.tracer.start_span(span_name)
    
    @asynccontextmanager
    async def trace_tool_execution(
        self,
        tool_name: str,
        agent_id: str,
        operation_type: str = "execute",
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ):
        """Trace a tool execution."""
        span_name = f"tool.{tool_name}"
        
        if OPENTELEMETRY_AVAILABLE:
            with self.tracer.start_as_current_span(span_name) as span:
                # Set standard attributes
                span.set_attribute(SpanAttributes.COMPONENT, self.config.service_name)
                span.set_attribute(SpanAttributes.OPERATION_TYPE, SpanKind.TOOL_EXECUTION)
                span.set_attribute(SpanAttributes.TOOL_NAME, tool_name)
                span.set_attribute(SpanAttributes.AGENT_ID, agent_id)
                span.set_attribute(SpanAttributes.OPERATION_TYPE, operation_type)
                
                if parameters:
                    span.set_attribute("tool.parameters", json.dumps(parameters))
                
                if timeout:
                    span.set_attribute("tool.timeout", str(timeout))
                
                # Add baggage
                if self.config.enable_baggage:
                    for key, value in self.span_context.baggage.items():
                        baggage.set_baggage(key, value)
                
                start_time = time.time()
                
                try:
                    yield span
                except Exception as e:
                    # Record error on span
                    span.set_status(trace.Status.ERROR)
                    span.set_attribute(SpanAttributes.ERROR_TYPE, "tool_execution_error")
                    span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
                    span.record_exception(e)
                finally:
                    # Record metrics
                    duration = time.time() - start_time
                    status = "success" if not span.status.is_set else "error"
                    
                    self.metrics.get('tool_executions_total').labels(
                        tool_name=tool_name,
                        agent_type=agent_id,
                        status=status
                    ).inc()
                    
                    self.metrics.get('agent_operations_total').labels(
                        operation_type="tool_execution",
                        agent_type=agent_id,
                        status=status
                    ).inc()
        else:
            yield self.tracer.start_span(span_name)
    
    @asynccontextmanager
    async def trace_database_operation(
        self,
        operation: str,
        query_type: str = "select",
        table: Optional[str] = None,
        query: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Trace a database operation."""
        span_name = f"database.{operation}"
        
        if OPENTELEMETRY_AVAILABLE:
            with self.tracer.start_as_current_span(span_name) as span:
                # Set standard attributes
                span.set_attribute(SpanAttributes.COMPONENT, self.config.service_name)
                span.set_attribute(SpanAttributes.OPERATION_TYPE, SpanKind.DATABASE_OPERATION)
                span.set_attribute("database.operation", operation)
                
                if table:
                    span.set_attribute("database.table", table)
                
                if query:
                    span.set_attribute("database.query", query[:500])  # Limit for size
                
                span.set_attribute("database.query_type", query_type)
                
                # Add baggage
                if self.config.enable_baggage:
                    for key, value in self.span_context.baggage.items():
                        baggage.set_baggage(key, value)
                
                # Set custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                start_time = time.time()
                
                try:
                    yield span
                except Exception as e:
                    # Record error on span
                    span.set_status(trace.Status.ERROR)
                    span.set_attribute(SpanAttributes.ERROR_TYPE, "database_error")
                    span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
                    span.record_exception(e)
                finally:
                    # Record metrics
                    duration = time.time() - start_time
                    status = "success" if not span.status.is_set else "error"
                    
                    # You could have database-specific metrics here
                    # self.db_metrics.get('database_operations_total').labels(
                    #     operation=operation, table=table, status=status
                    # ).inc()
                    # )
                    
                    pass
        else:
            yield self.tracer.start_span(span_name)
    
    def set_baggage(self, key: str, value: str):
        """Set baggage for current context."""
        self.span_context.baggage[key] = value
        
    def get_baggage(self) -> Dict[str, str]:
        """Get current baggage."""
        return self.span_context.baggage.copy()
    
    def set_user_context(self, user_id: str, session_id: str = None):
        """Set user context for tracing."""
        self.span_context.user_id = user_id
        self.span_context.session_id = session_id
    
    def clear_user_context(self):
        """Clear user context."""
        self.span_context.user_id = None
        self.span_context.session_id = None
    
    def force_flush(self):
        """Force flush of spans and metrics."""
        if OPENTELEMETRY_AVAILABLE:
            logger.info("Flushing OpenTelemetry spans and metrics")
            # OpenTelemetry handles automatic flushing, but we can force it if needed
            pass


class MockTracer:
    """Mock tracer for development/testing."""
    
    def __init__(self):
        self.spans = []
    
    def start_as_current_span(self, name: str):
        return MockSpan(self, name, self.spans)
    
    def get_tracer_provider(self):
        return self


class MockSpan:
    """Mock span for development/testing."""
    
    def __init__(self, name: str, span_collection: List):
        self.name = name
        self.span_collection = span_collection
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.status = None
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value
    
    def set_status(self, status):
        """Set span status."""
        self.status = status
    
    def record_exception(self, exception: Exception):
        """Record exception on span."""
        import time
        self.events.append({
            "exception": str(exception),
            "timestamp": time.time()
        })
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self


class MockMetrics:
    """Mock metrics for development/testing."""
    
    def __init__(self):
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
    
    def get(self, name: str, *labels):
        """Get metric by name and labels."""
        # Return mock metric
        if name in self.counters:
            return self.counters[name]
        elif name in self.histograms:
            return self.histograms[name]
        elif name in self.gauges:
            return self.gauges[name]
        
        # Return default mock metric
        return MockMetric(name)
    
    def labels(self, *labels):
        """Create labels for metrics."""
        return MockLabels(labels)


class MockMetric:
    """Mock metric for development/testing."""
    
    def __init__(self, name: str):
        self.name = name
    
    def inc(self):
        """Increment counter."""
        pass
    
    def observe(self, value: float):
        """Observe histogram value."""
        pass
    
    def set(self, value: float):
        """Set gauge value."""
        pass


class MockLabels:
    """Mock labels for metrics."""
    
    def __init__(self, labels):
        self.labels = labels


# Global tracing manager instance
tracing_manager: Optional[TracingManager] = None


async def get_tracing_manager(config: TracingConfig) -> TracingManager:
    """Get or create global tracing manager."""
    global tracing_manager
    
    if tracing_manager is None:
        tracing_manager = TracingManager(config)
        logger.info("Global tracing manager initialized")
    
    return tracing_manager


# Decorators for easy tracing
def trace_agent_operation(
    operation_name: str,
    agent_id_param: str = "agent_id",
    operation_type: str = "execute",
    attributes_param: str = "attributes"
):
    """Decorator to trace agent operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract tracing parameters from kwargs
            agent_id = kwargs.get(agent_id_param, "unknown")
            attributes = kwargs.get(attributes_param, {})
            
            tracing = await get_tracing_manager(
                TracingConfig(enable_tracing=True, service_name="integration-bridge")
            )
            
            async with tracing.trace_agent_operation(
                operation_name=operation_name,
                agent_id=agent_id,
                operation_type=operation_type,
                attributes=attributes
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add result to span if successful
                    if hasattr(result, 'status'):
                        span.set_attribute("operation.result.status", result.status)
                    if hasattr(result, 'execution_id'):
                        span.set_attribute("operation.execution_id", result.execution_id)
                    
                    return result
                    
                except Exception as e:
                    span.set_status(trace.Status.ERROR)
                    span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_api_request(
    method_param: str = "method",
    endpoint_param: str = "endpoint",
    request_id_param: str = "request_id",
    attributes_param: str = "attributes"
):
    """Decorator to trace API requests."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            method = kwargs.get(method_param, "GET")
            endpoint = kwargs.get(endpoint_param, "unknown")
            request_id = kwargs.get(request_id_param)
            attributes = kwargs.get(attributes_param, {})
            
            tracing = await get_tracing_manager(
                TracingConfig(enable_tracing=True, service_name="integration-bridge")
            )
            
            async with tracing.trace_api_request(
                method=method,
                endpoint=endpoint,
                request_id=request_id,
                attributes=attributes
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add response info to span
                    if hasattr(result, 'status_code'):
                        span.set_attribute(SpanAttributes.RESPONSE_CODE, str(result.status_code))
                    if hasattr(result, 'response_size'):
                        span.set_attribute(SpanAttributes.RESPONSE_SIZE, str(len(str(result))))
                    
                    return result
                    
                except Exception as e:
                    span.set_status(trace.Status.ERROR)
                    span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_tool_execution(
    tool_name_param: str = "tool_name",
    agent_id_param: str = "agent_id",
    timeout_param: str = "timeout",
    parameters_param: str = "parameters",
    attributes_param: str = "attributes"
):
    """Decorator to trace tool executions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tool_name = kwargs.get(tool_name_param, "unknown")
            agent_id = kwargs.get(agent_id_param, "unknown")
            timeout = kwargs.get(timeout_param)
            parameters = kwargs.get(parameters_param, {})
            attributes = kwargs.get(attributes_param, {})
            
            tracing = await get_tracing_manager(
                TracingConfig(enable_tracing=True, service_name="integration-bridge")
            )
            
            async with tracing.trace_tool_execution(
                tool_name=tool_name,
                agent_id=agent_id,
                timeout=timeout,
                parameters=parameters,
                attributes=attributes
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add result to span
                    if hasattr(result, 'status'):
                        span.set_attribute("execution.result.status", result.status)
                    if hasattr(result, 'execution_time'):
                        span.set_attribute("execution.time", str(result.execution_time))
                    if hasattr(result, 'error'):
                        span.set_attribute("execution.error", result.error)
                    
                    return result
                    
                except Exception as e:
                    span.set_status(trace.Status.ERROR)
                    span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator