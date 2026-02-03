"""
Enhanced timeout handling and async task cancellation utilities.
"""

import asyncio
import time
from typing import Optional, Any, Callable, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import structlog
from concurrent.futures import TimeoutError as ConcurrentTimeoutError

logger = structlog.get_logger(__name__)


class TimeoutStrategy(Enum):
    """Timeout handling strategies."""

    FAIL_FAST = "fail_fast"
    FALLBACK = "fallback"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""

    default_timeout: float = 30.0
    max_retries: int = 3
    base_backoff: float = 1.0
    max_backoff: float = 60.0
    jitter: bool = True
    strategy: TimeoutStrategy = TimeoutStrategy.FAIL_FAST
    fallback_func: Optional[Callable] = None
    on_timeout: Optional[Callable] = None


class TimeoutError(Exception):
    """Custom timeout error with additional context."""

    def __init__(
        self,
        message: str,
        timeout: float,
        operation: str,
        context: Dict[str, Any] = None,
    ):
        self.timeout = timeout
        self.operation = operation
        self.context = context or {}
        super().__init__(message)


class CancellationManager:
    """Manages graceful cancellation of async tasks."""

    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.cancellation_handlers: Dict[str, List[Callable]] = {}

    async def execute_with_cancellation(
        self,
        operation_name: str,
        coro,
        timeout: Optional[float] = None,
        cleanup_handlers: Optional[List[Callable]] = None,
    ) -> Any:
        """Execute coroutine with cancellation support."""

        # Create task
        task = asyncio.create_task(coro)
        self.active_tasks[operation_name] = task

        # Register cleanup handlers
        if cleanup_handlers:
            self.cancellation_handlers[operation_name] = cleanup_handlers

        try:
            if timeout:
                return await asyncio.wait_for(task, timeout=timeout)
            else:
                return await task

        except asyncio.TimeoutError:
            # Handle timeout
            await self._handle_timeout(operation_name, task, timeout)
            raise TimeoutError(
                f"Operation {operation_name} timed out after {timeout}s",
                timeout=timeout,
                operation=operation_name,
            )

        except asyncio.CancelledError:
            # Handle cancellation
            await self._handle_cancellation(operation_name, task)
            raise

        finally:
            # Cleanup
            del self.active_tasks[operation_name]
            if operation_name in self.cancellation_handlers:
                del self.cancellation_handlers[operation_name]

    async def _handle_timeout(
        self, operation_name: str, task: asyncio.Task, timeout: Optional[float]
    ):
        """Handle timeout for a specific operation."""
        logger.warning("Operation timed out", operation=operation_name, timeout=timeout)

        # Cancel the task if it's still running
        if not task.done():
            task.cancel()

            # Wait for cancellation to complete
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Execute cleanup handlers
        await self._execute_cleanup_handlers(operation_name)

    async def _handle_cancellation(self, operation_name: str, task: asyncio.Task):
        """Handle cancellation for a specific operation."""
        logger.info("Operation cancelled", operation=operation_name)

        # Execute cleanup handlers
        await self._execute_cleanup_handlers(operation_name)

    async def _execute_cleanup_handlers(self, operation_name: str):
        """Execute cleanup handlers for an operation."""
        handlers = self.cancellation_handlers.get(operation_name, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    # Run sync handler in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler)
            except Exception as e:
                logger.error(
                    "Cleanup handler failed", operation=operation_name, error=str(e)
                )

    def cancel_operation(
        self, operation_name: str, reason: str = "Manual cancellation"
    ):
        """Manually cancel an operation."""
        if operation_name not in self.active_tasks:
            logger.warning(
                "Attempted to cancel non-existent operation", operation=operation_name
            )
            return False

        task = self.active_tasks[operation_name]
        if task.done():
            logger.info("Operation already completed", operation=operation_name)
            return True

        logger.info("Cancelling operation", operation=operation_name, reason=reason)

        task.cancel()
        return True

    def get_active_operations(self) -> List[str]:
        """Get list of active operations."""
        return list(self.active_tasks.keys())

    async def cancel_all_operations(self, reason: str = "System shutdown"):
        """Cancel all active operations."""
        operations = list(self.active_tasks.keys())

        logger.info("Cancelling all operations", count=len(operations), reason=reason)

        for operation_name in operations:
            self.cancel_operation(operation_name, reason)


class TimeoutManager:
    """Advanced timeout management with multiple strategies."""

    def __init__(self, config: TimeoutConfig):
        self.config = config

    async def execute_with_timeout(
        self,
        operation_name: str,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """Execute function with advanced timeout handling."""

        timeout = timeout or self.config.default_timeout
        start_time = time.time()
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            attempt_start = time.time()

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout
                    )
                else:
                    # For sync functions
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: asyncio.wait_for(
                            func(*args, **kwargs), timeout=timeout
                        ),
                    )

                # Success - return result
                elapsed = time.time() - attempt_start
                logger.info(
                    "Operation completed successfully",
                    operation=operation_name,
                    attempt=attempt + 1,
                    elapsed=elapsed,
                )

                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                elapsed = time.time() - attempt_start

                logger.warning(
                    "Operation attempt timed out",
                    operation=operation_name,
                    attempt=attempt + 1,
                    timeout=timeout,
                    elapsed=elapsed,
                )

                # Apply timeout strategy
                if self.config.strategy == TimeoutStrategy.FAIL_FAST:
                    raise TimeoutError(
                        f"Operation {operation_name} failed after timeout",
                        timeout=timeout,
                        operation=operation_name,
                        context={
                            "attempt": attempt + 1,
                            "total_time": time.time() - start_time,
                        },
                    )

                elif self.config.strategy == TimeoutStrategy.FALLBACK:
                    if self.config.fallback_func and attempt < self.config.max_retries:
                        try:
                            fallback_result = await self._execute_fallback(
                                operation_name, *args, **kwargs
                            )
                            elapsed = time.time() - start_time
                            logger.info(
                                "Operation completed via fallback",
                                operation=operation_name,
                                elapsed=elapsed,
                            )
                            return fallback_result
                        except Exception as fallback_error:
                            logger.error(
                                "Fallback operation failed",
                                operation=operation_name,
                                error=str(fallback_error),
                            )

                # Continue to next attempt or raise final error
                if attempt >= self.config.max_retries:
                    raise TimeoutError(
                        f"Operation {operation_name} failed after {self.config.max_retries} attempts",
                        timeout=timeout,
                        operation=operation_name,
                        context={
                            "attempts": attempt + 1,
                            "total_time": time.time() - start_time,
                        },
                    )

            except Exception as e:
                last_exception = e
                elapsed = time.time() - attempt_start

                logger.error(
                    "Operation attempt failed",
                    operation=operation_name,
                    attempt=attempt + 1,
                    error=str(e),
                    elapsed=elapsed,
                )

                # Continue to next attempt for non-timeout errors
                if attempt >= self.config.max_retries:
                    raise TimeoutError(
                        f"Operation {operation_name} failed after {self.config.max_retries} attempts",
                        timeout=timeout,
                        operation=operation_name,
                        context={
                            "attempts": attempt + 1,
                            "total_time": time.time() - start_time,
                            "last_error": str(e),
                        },
                    )

            # Apply backoff before next attempt
            if attempt < self.config.max_retries:
                backoff = self._calculate_backoff(attempt)
                logger.info(
                    "Waiting before retry",
                    operation=operation_name,
                    attempt=attempt + 1,
                    backoff=backoff,
                )
                await asyncio.sleep(backoff)

    async def _execute_fallback(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute fallback function."""
        if self.config.fallback_func:
            if asyncio.iscoroutinefunction(self.config.fallback_func):
                return await self.config.fallback_func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self.config.fallback_func(*args, **kwargs)
                )
        raise TimeoutError(f"No fallback function configured for {operation_name}")

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_backoff = self.config.base_backoff
        max_backoff = self.config.max_backoff

        # Exponential backoff
        exponential_backoff = min(base_backoff * (2**attempt), max_backoff)

        # Add jitter if enabled
        if self.config.jitter:
            import random

            jitter_factor = 0.1  # 10% jitter
            jitter = exponential_backoff * jitter_factor * random.uniform(-1, 1)
            return max(0, exponential_backoff + jitter)

        return exponential_backoff


class TaskManager:
    """High-level task management with timeout and cancellation."""

    def __init__(self):
        self.cancellation_manager = CancellationManager()
        self.timeout_managers: Dict[str, TimeoutManager] = {}

    def get_timeout_manager(
        self, operation_type: str, config: Optional[TimeoutConfig] = None
    ) -> TimeoutManager:
        """Get or create timeout manager for operation type."""
        if operation_type not in self.timeout_managers:
            manager_config = config or TimeoutConfig()
            self.timeout_managers[operation_type] = TimeoutManager(manager_config)

        return self.timeout_managers[operation_type]

    async def execute_operation(
        self,
        operation_name: str,
        operation_type: str,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        cleanup_handlers: Optional[List[Callable]] = None,
        **kwargs,
    ) -> Any:
        """Execute operation with comprehensive timeout and cancellation support."""

        timeout_manager = self.get_timeout_manager(operation_type)

        return await self.cancellation_manager.execute_with_cancellation(
            operation_name=operation_name,
            coro=timeout_manager.execute_with_timeout(
                operation_name=operation_name,
                func=func,
                *args,
                timeout=timeout,
                **kwargs,
            ),
            timeout=timeout,
            cleanup_handlers=cleanup_handlers,
        )

    def cancel_operation(
        self, operation_name: str, reason: str = "Manual cancellation"
    ):
        """Cancel a specific operation."""
        return self.cancellation_manager.cancel_operation(operation_name, reason)

    def get_active_operations(self) -> List[str]:
        """Get all active operations."""
        return self.cancellation_manager.get_active_operations()

    async def shutdown_all_operations(self, reason: str = "System shutdown"):
        """Shutdown all operations gracefully."""
        await self.cancellation_manager.cancel_all_operations(reason)


# Global task manager instance
task_manager = TaskManager()


# Decorator for easy timeout handling
def with_timeout(
    operation_type: str = "default",
    timeout: Optional[float] = None,
    max_retries: int = 3,
    strategy: TimeoutStrategy = TimeoutStrategy.FAIL_FAST,
    fallback_func: Optional[Callable] = None,
):
    """Decorator to add timeout protection to functions."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate operation name if not provided
            operation_name = getattr(func, "__name__", "unknown_operation")

            return await task_manager.execute_operation(
                operation_name=operation_name,
                operation_type=operation_type,
                func=func,
                *args,
                timeout=timeout,
                cleanup_handlers=None,
                **kwargs,
            )

        return wrapper

    return decorator


# Example usage
@with_timeout(operation_type="database_query", timeout=10.0, max_retries=2)
async def query_database(sql: str, params: dict):
    """Database query with timeout protection."""
    # Database query implementation
    pass


@with_timeout(
    operation_type="external_api",
    timeout=30.0,
    strategy=TimeoutStrategy.FALLBACK,
    fallback_func=lambda: {"fallback": True, "data": []},
)
async def call_external_api(endpoint: str, data: dict):
    """External API call with fallback."""
    # API call implementation
    pass
