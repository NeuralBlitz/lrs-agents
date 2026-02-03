"""
Circuit breaker implementation for external service resilience.
"""

import asyncio
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: type = Exception  # Exception that counts as failure
    success_threshold: int = 2  # Successes to close circuit from half-open
    monitor_timeout: float = 30.0  # Timeout for individual calls


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, service_name: str, state: CircuitState):
        self.service_name = service_name
        self.state = state
        super().__init__(f"Circuit breaker for {service_name} is {state.value}")


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0
        self.call_count = 0
        self.total_failures = 0
        self.total_successes = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.call_count += 1

        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker moving to half-open", name=self.name)
            else:
                raise CircuitBreakerError(self.name, self.state)

        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.monitor_timeout
                )
            else:
                # For sync functions, run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: asyncio.wait_for(
                        func(*args, **kwargs), timeout=self.config.monitor_timeout
                    ),
                )

            # Success - record metrics
            self._on_success()
            return result

        except asyncio.TimeoutError as e:
            # Timeout - treat as failure
            self._on_failure()
            raise
        except Exception as e:
            # Check if this exception counts as failure
            if isinstance(e, self.config.expected_exception) or isinstance(
                e, asyncio.TimeoutError
            ):
                self._on_failure()
            else:
                self._on_success()  # Other exceptions might be expected
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return False

        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        else:
            # If we were closed but had failures, reset
            if self.failure_count > 0:
                self._close_circuit()

    def _on_failure(self):
        """Handle failed call."""
        self.total_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit."""
        self.state = CircuitState.OPEN
        self.failure_count = 0
        self.success_count = 0
        logger.warning(
            "Circuit breaker opened",
            name=self.name,
            total_failures=self.total_failures,
            failure_threshold=self.config.failure_threshold,
        )

    def _close_circuit(self):
        """Close the circuit."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(
            "Circuit breaker closed",
            name=self.name,
            total_successes=self.total_successes,
        )

    def force_open(self):
        """Manually open the circuit (for testing)."""
        self._open_circuit()

    def force_close(self):
        """Manually close the circuit (for testing)."""
        self._close_circuit()

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        total_calls = self.total_successes + self.total_failures
        success_rate = (self.total_successes / total_calls) if total_calls > 0 else 0.0

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "monitor_timeout": self.config.monitor_timeout,
            },
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def register(self, name: str, circuit_breaker: CircuitBreaker):
        """Register a circuit breaker."""
        self.circuit_breakers[name] = circuit_breaker
        logger.info("Registered circuit breaker", name=name)

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats() for name, breaker in self.circuit_breakers.items()
        }

    async def call_with_circuit_breaker(
        self, service_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Call a function with circuit breaker protection."""
        breaker = self.get(service_name)
        if breaker is None:
            # Create default circuit breaker if not exists
            config = CircuitBreakerConfig()
            breaker = CircuitBreaker(service_name, config)
            self.register(service_name, breaker)

        return await breaker.call(func, *args, **kwargs)

    def force_open_all(self):
        """Force open all circuit breakers (for testing)."""
        for breaker in self.circuit_breakers.values():
            breaker.force_open()

    def force_close_all(self):
        """Force close all circuit breakers (for testing)."""
        for breaker in self.circuit_breakers.values():
            breaker.force_close()


# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()


# Decorator for easy use
def with_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 2,
    monitor_timeout: float = 30.0,
):
    """Decorator to add circuit breaker protection to functions."""

    def decorator(func):
        # Create or get circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            monitor_timeout=monitor_timeout,
        )

        breaker = circuit_breaker_registry.get(service_name)
        if breaker is None:
            breaker = CircuitBreaker(service_name, config)
            circuit_breaker_registry.register(service_name, breaker)

        if asyncio.iscoroutinefunction(func):

            async def wrapper(*args, **kwargs):
                return await breaker.call(func, *args, **kwargs)

            return wrapper
        else:

            async def wrapper(*args, **kwargs):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: breaker.call(func, *args, **kwargs)
                )

            return wrapper

    return decorator


# Example usage for specific services
@with_circuit_breaker("opencode_api", failure_threshold=3, recovery_timeout=30.0)
async def call_opencode_api(endpoint: str, data: dict):
    """Protected opencode API call."""
    # Implementation would go here
    pass


@with_circuit_breaker("lrs_api", failure_threshold=5, recovery_timeout=60.0)
async def call_lrs_api(tool_name: str, parameters: dict):
    """Protected LRS API call."""
    # Implementation would go here
    pass
