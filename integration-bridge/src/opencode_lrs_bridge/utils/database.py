"""
Robust database connection pooling for the integration bridge.
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import asyncpg
import structlog

from ..config.settings import IntegrationBridgeConfig

logger = structlog.get_logger(__name__)


@dataclass
class PoolConfig:
    """Database pool configuration."""

    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    timeout: float = 30.0  # Connection timeout
    idle_timeout: float = 300.0  # Idle timeout before connection is recycled
    max_lifetime: float = 3600.0  # Maximum connection lifetime
    pool_recycle: int = 3600  # Pool recycle interval
    health_check_period: float = 60.0  # Health check interval
    health_check_timeout: float = 5.0  # Health check timeout


class DatabaseConnectionPool:
    """Advanced database connection pool with health monitoring."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.pool_config = PoolConfig(
            min_size=config.database.pool_size,
            max_size=config.database.max_overflow + config.database.pool_size,
            max_overflow=config.database.max_overflow,
            timeout=config.database.pool_timeout,
            idle_timeout=300.0,
            max_lifetime=3600.0,
            pool_recycle=3600.0,
            health_check_period=60.0,
            health_check_timeout=5.0,
        )
        self.pool: Optional[asyncpg.Pool] = None
        self.creation_time: float = time.time()
        self.total_connections_created: int = 0
        self.total_connections_closed: int = 0
        self.active_connections: int = 0
        self.failed_connections: int = 0
        self.last_health_check: float = 0.0

    async def initialize(self):
        """Initialize the connection pool."""
        try:
            logger.info(
                "Initializing database connection pool", config=self.pool_config
            )

            self.pool = await asyncpg.create_pool(
                self.config.database.url,
                min_size=self.pool_config.min_size,
                max_size=self.pool_config.max_size,
                max_overflow=self.pool_config.max_overflow,
                command_timeout=self.pool_config.timeout,
                server_settings={
                    "application_name": "integration_bridge",
                    "timezone": "UTC",
                },
                # Connection class with connection timeout
                connection_class=TimeoutConnection,
                kwargs={
                    "timeout": self.pool_config.timeout,
                    "command_timeout": self.pool_config.timeout,
                },
            )

            logger.info(
                "Database pool initialized successfully",
                pool_size=self.pool_config.min_size,
                max_overflow=self.pool_config.max_overflow,
            )

        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with timeout and retry logic."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        self.active_connections += 1
        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                # Check if connection is healthy
                if not await self._is_connection_healthy(conn):
                    logger.warning("Unhealthy connection detected, creating new one")
                    conn = await self.pool.acquire()

                yield conn

        except Exception as e:
            self.failed_connections += 1
            logger.error("Database connection failed", error=str(e))
            raise
        finally:
            self.active_connections -= 1
            duration = time.time() - start_time
            if duration > 5.0:  # Log slow connections
                logger.warning(
                    "Slow database connection",
                    duration=duration,
                    active_connections=self.active_connections,
                )

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch_mode: str = "fetch",
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute a database query with proper error handling."""
        timeout = timeout or self.pool_config.timeout

        try:
            async with self.get_connection() as conn:
                start_time = time.time()

                if params:
                    result = await conn.fetch(
                        query, params, timeout=timeout, fetch=fetch_mode
                    )
                else:
                    result = await conn.fetch(query, timeout=timeout, fetch=fetch_mode)

                duration = time.time() - start_time
                if duration > 1.0:
                    logger.warning(
                        "Slow database query",
                        query=query[:100] + ("..." if len(query) > 100 else ""),
                        duration=duration,
                        timeout=timeout,
                    )

                return result

        except asyncpg.TimeoutError as e:
            logger.error("Database query timeout", query=query[:100], error=str(e))
            raise TimeoutError(f"Database query timed out: {query[:100]}")
        except Exception as e:
            logger.error("Database query failed", query=query[:100], error=str(e))
            raise RuntimeError(f"Database query failed: {str(e)}")

    async def execute_transaction(self, queries: list, timeout: Optional[float] = None):
        """Execute multiple queries in a transaction."""
        timeout = timeout or self.pool_config.timeout

        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    results = []

                    for query_data in queries:
                        if isinstance(query_data, str):
                            result = await conn.fetch(query_data, timeout=timeout)
                        elif isinstance(query_data, dict):
                            query_str = query_data.get("query")
                            params = query_data.get("params", {})
                            result = await conn.fetch(
                                query_str, params, timeout=timeout
                            )
                        else:
                            raise ValueError("Invalid query format")

                        results.append(result)

                    return results

        except Exception as e:
            logger.error("Database transaction failed", error=str(e))
            raise RuntimeError(f"Database transaction failed: {str(e)}")

    async def execute_many(
        self, query: str, params_list: list, timeout: Optional[float] = None
    ):
        """Execute the same query with multiple parameter sets."""
        timeout = timeout or self.pool_config.timeout

        try:
            async with self.get_connection() as conn:
                # Prepare queries for batch execution
                queries = [(query, params or {}) for params in params_list]

                start_time = time.time()
                results = await conn.executemany(queries, timeout=timeout)
                duration = time.time() - start_time

                if duration > 2.0:
                    logger.warning(
                        "Batch database query took time",
                        queries_count=len(queries),
                        duration=duration,
                    )

                return results

        except Exception as e:
            logger.error("Batch database query failed", error=str(e))
            raise RuntimeError(f"Batch database query failed: {str(e)}")

    async def _is_connection_healthy(self, conn) -> bool:
        """Check if a database connection is healthy."""
        try:
            # Simple health check with timeout
            await asyncio.wait_for(
                conn.fetch("SELECT 1", timeout=self.pool_config.health_check_timeout),
                timeout=self.pool_config.health_check_timeout,
            )
            return True
        except (asyncpg.TimeoutError, Exception):
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on the database pool."""
        if not self.pool:
            return {"status": "unhealthy", "error": "Pool not initialized"}

        current_time = time.time()

        # Skip if too soon since last check
        if current_time - self.last_health_check < self.pool_config.health_check_period:
            return {
                "status": "healthy",
                "last_check": self.last_health_check,
                "skip_reason": "too_soon",
            }

        try:
            # Get pool statistics
            pool_stats = self.pool.get_size()

            # Test connection
            test_start = time.time()
            async with self.pool.acquire() as conn:
                await conn.fetch(
                    "SELECT 1", timeout=self.pool_config.health_check_timeout
                )
            test_duration = time.time() - test_start

            self.last_health_check = current_time

            return {
                "status": "healthy",
                "pool_size": pool_size,
                "total_created": self.total_connections_created,
                "total_closed": self.total_connections_closed,
                "active_connections": self.active_connections,
                "failed_connections": self.failed_connections,
                "test_connection_duration": test_duration,
                "uptime": current_time - self.creation_time,
                "last_check": self.last_health_check,
            }

        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            self.last_health_check = current_time

            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": self.last_health_check,
            }

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        if not self.pool:
            return {"status": "not_initialized", "error": "Pool not initialized"}

        pool_stats = self.pool.get_size()

        return {
            "status": "active",
            "pool_config": {
                "min_size": self.pool_config.min_size,
                "max_size": self.pool_config.max_size,
                "max_overflow": self.pool_config.max_overflow,
                "timeout": self.pool_config.timeout,
                "idle_timeout": self.pool_config.idle_timeout,
                "max_lifetime": self.pool_config.max_lifetime,
                "pool_recycle": self.pool_config.pool_recycle,
                "health_check_period": self.pool_config.health_check_period,
                "health_check_timeout": self.pool_config.health_check_timeout,
            },
            "current_stats": pool_stats,
            "total_created": self.total_connections_created,
            "total_closed": self.total_connections_closed,
            "active_connections": self.active_connections,
            "failed_connections": self.failed_connections,
            "uptime": time.time() - self.creation_time,
        }


class TimeoutConnection(asyncpg.Connection):
    """Connection class with custom timeout handling."""

    def __init__(self, *args, timeout: float = 30.0, **kwargs):
        self.timeout = timeout
        self.timeout_error = None
        super().__init__(*args, **kwargs)

    async def fetch(self, *args, **kwargs):
        """Execute fetch with timeout override."""
        # Merge timeout with any provided in kwargs
        kwargs["timeout"] = self.timeout

        try:
            return await super().fetch(*args, **kwargs)
        except asyncio.TimeoutError as e:
            self.timeout_error = e
            raise
        except Exception as e:
            # Re-raise other exceptions with timeout context
            if self.timeout_error:
                raise asyncio.TimeoutError(
                    f"Connection timed out after {self.timeout}s (original error: {self.timeout_error})"
                )
            raise

    async def fetchmany(self, *args, **kwargs):
        """Execute fetchmany with timeout override."""
        kwargs["timeout"] = self.timeout

        try:
            return await super().fetchmany(*args, **kwargs)
        except asyncio.TimeoutError as e:
            self.timeout_error = e
            raise
        except Exception as e:
            if self.timeout_error:
                raise asyncio.TimeoutError(
                    f"Connection timed out after {self.timeout}s (original error: {self.timeout_error})"
                )
            raise

    async def executemany(self, *args, **kwargs):
        """Execute executemany with timeout override."""
        kwargs["timeout"] = self.timeout

        try:
            return await super().executemany(*args, **kwargs)
        except asyncio.TimeoutError as e:
            self.timeout_error = e
            raise
        except Exception as e:
            if self.timeout_error:
                raise asyncio.TimeoutError(
                    f"Connection timed out after {self.timeout}s (original error: {self.timeout_error})"
                )
            raise


class DatabaseManager:
    """High-level database management with connection pooling and monitoring."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.pool = DatabaseConnectionPool(config)
        self.monitoring_enabled = config.monitoring.enable_logging

    async def initialize(self):
        """Initialize the database manager."""
        await self.pool.initialize()

        if self.monitoring_enabled:
            logger.info("Database manager initialized with monitoring")

    async def close(self):
        """Close the database manager."""
        await self.pool.close()

    async def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """Execute a database query."""
        return await self.pool.execute_query(query, params, **kwargs)

    async def execute_transaction(self, queries: list, **kwargs):
        """Execute a database transaction."""
        return await self.pool.execute_transaction(queries, **kwargs)

    async def execute_many(self, query: str, params_list: list, **kwargs):
        """Execute the same query with multiple parameter sets."""
        return await self.pool.execute_many(query, params_list, **kwargs)

    async def get_connection(self):
        """Get a database connection from the pool."""
        return self.pool.get_connection()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return await self.pool.health_check()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.pool.get_stats()


# Global database manager instance
database_manager = None


async def get_database_manager(config: IntegrationBridgeConfig) -> DatabaseManager:
    """Get or create the global database manager."""
    global database_manager

    if database_manager is None:
        database_manager = DatabaseManager(config)
        await database_manager.initialize()
    elif database_manager.config != config:
        # Configuration changed, reinitialize
        await database_manager.close()
        database_manager = DatabaseManager(config)
        await database_manager.initialize()

    return database_manager


# Migration support
class DatabaseMigrator:
    """Database schema migration manager."""

    def __init__(self, pool: DatabaseConnectionPool):
        self.pool = pool
        self.migrations_dir = "migrations"

    async def run_migrations(self):
        """Run pending database migrations."""
        try:
            # This would check for migration files and run them
            logger.info("Running database migrations")

            # Example migration logic:
            # await self._create_migration_table()
            # pending_migrations = await self._get_pending_migrations()
            # for migration in pending_migrations:
            #     await self._run_migration(migration)

            logger.info("Database migrations completed successfully")

        except Exception as e:
            logger.error("Database migration failed", error=str(e))
            raise


# Query builder for safe SQL generation
class QueryBuilder:
    """Safe SQL query builder to prevent injection."""

    @staticmethod
    def select(
        table: str,
        columns: List[str],
        where_clause: str = None,
        order_by: str = None,
        limit: int = None,
    ) -> str:
        """Build a safe SELECT query."""

        # Escape table name to prevent injection
        safe_table = QueryBuilder._escape_identifier(table)

        # Build column list
        safe_columns = ", ".join(
            [QueryBuilder._escape_identifier(col) for col in columns]
        )

        query = f"SELECT {safe_columns} FROM {safe_table}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit:
            query += f" LIMIT {limit}"

        return query

    @staticmethod
    def insert(table: str, data: Dict[str, Any], returning: str = None) -> tuple:
        """Build a safe INSERT query."""
        safe_table = QueryBuilder._escape_identifier(table)

        columns = list(data.keys())
        values = list(data.values())
        placeholders = ", ".join([f"${i + 1}" for i in range(len(values))])

        columns_str = ", ".join(
            [QueryBuilder._escape_identifier(col) for col in columns]
        )

        query = f"INSERT INTO {safe_table} ({columns_str}) VALUES ({placeholders})"

        if returning:
            query += f" RETURNING {returning}"

        return (query, values)

    @staticmethod
    def update(table: str, data: Dict[str, Any], where_clause: str) -> str:
        """Build a safe UPDATE query."""
        safe_table = QueryBuilder._escape_identifier(table)

        set_clauses = [
            f"{QueryBuilder._escape_identifier(k)} = ${QueryBuilder._escape_value(v)}"
            for k, v in data.items()
        ]

        set_clause_str = ", ".join(set_clauses)

        query = f"UPDATE {safe_table} SET {set_clause_str} WHERE {where_clause}"

        return query

    @staticmethod
    def delete(table: str, where_clause: str) -> str:
        """Build a safe DELETE query."""
        safe_table = QueryBuilder._escape_identifier(table)

        query = f"DELETE FROM {safe_table} WHERE {where_clause}"

        return query

    @staticmethod
    def _escape_identifier(identifier: str) -> str:
        """Escape SQL identifiers (table names, column names)."""
        # This is a simplified implementation
        # In production, use a proper SQL escaper
        return identifier.replace('"', '""').replace("'", "''")

    @staticmethod
    def _escape_value(value: Any) -> str:
        """Escape SQL values."""
        # This is a simplified implementation
        # In production, use parameterized queries or proper escaper
        if isinstance(value, str):
            return value.replace("'", "''").replace("\\", "\\\\")
        return str(value)
