"""
Main application entry point for opencode ↔ LRS-Agents integration bridge.
"""

import asyncio
import uvicorn
from typing import Optional
import structlog

from .config.settings import IntegrationBridgeConfig
from .api.endpoints import create_app
from .websocket.manager import WebSocketBridge
from .tools.integration import ToolExecutionManager
from .utils.sync import StateSynchronizer
from .auth.security import SecurityMiddleware
from .utils.analytics import AnalyticsManager


async def create_bridge_application(config: IntegrationBridgeConfig):
    """Create and initialize the bridge application."""
    # Initialize components
    websocket_bridge = WebSocketBridge(config)
    tool_manager = ToolExecutionManager(config)
    state_synchronizer = StateSynchronizer(config)
    analytics_manager = AnalyticsManager(config)
    security_middleware = SecurityMiddleware(config)

    # Create FastAPI app
    app = create_app(config)

    # Store components in app state for dependency injection
    app.state.websocket_bridge = websocket_bridge
    app.state.tool_manager = tool_manager
    app.state.state_synchronizer = state_synchronizer
    app.state.analytics_manager = analytics_manager
    app.state.security_middleware = security_middleware

    # Start background tasks
    asyncio.create_task(websocket_bridge.start_background_connections())
    asyncio.create_task(analytics_manager.start_metrics_collection())

    return app


def main():
    """Main entry point for the integration bridge."""
    import click

    @click.command()
    @click.option(
        "--config", "-c", type=click.Path(exists=True), help="Configuration file path"
    )
    @click.option("--host", default="0.0.0.0", help="Host to bind to")
    @click.option("--port", default=9000, type=int, help="Port to bind to")
    @click.option("--workers", default=4, type=int, help="Number of worker processes")
    @click.option("--reload", is_flag=True, help="Enable auto-reload for development")
    @click.option("--log-level", default="INFO", help="Log level")
    def run(
        config: Optional[str],
        host: str,
        port: int,
        workers: int,
        reload: bool,
        log_level: str,
    ):
        """Start the opencode ↔ LRS-Agents Integration Bridge."""

        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        logger = structlog.get_logger(__name__)
        logger.info(
            "Starting Integration Bridge", host=host, port=port, workers=workers
        )

        # Load configuration
        if config:
            app_config = IntegrationBridgeConfig.from_file(config)
        else:
            app_config = IntegrationBridgeConfig()

        # Override with command line arguments
        app_config.api.host = host
        app_config.api.port = port
        app_config.api.workers = workers
        app_config.monitoring.log_level = log_level

        # Run the application
        uvicorn.run(
            "opencode_lrs_bridge.main:create_bridge_application",
            factory=True,
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level=log_level.lower(),
            access_log=True,
            app_dir=".",
        )

    run()


if __name__ == "__main__":
    main()
