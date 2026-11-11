"""
Application factory and command-line entry point for the integration bridge.
"""

import asyncio
import os
import ssl
from typing import Optional

import click
import structlog
import uvicorn
from fastapi import FastAPI

from .api.endpoints import create_app
from .auth.security import SecurityMiddleware
from .config.settings import IntegrationBridgeConfig
from .utils.analytics import AnalyticsManager
from .utils.sync import StateSynchronizer
from .websocket.manager import WebSocketBridge


def create_bridge_application(
    config: Optional[IntegrationBridgeConfig] = None,
) -> FastAPI:
    """Create the FastAPI application and attach bridge services.

    Uvicorn's factory contract requires a synchronous callable that returns
    an ASGI application. Long-running work starts in FastAPI startup, after
    the event loop exists.
    """
    if config is None:
        env_file = os.getenv("INTEGRATION_BRIDGE_CONFIG") or ".env"
        config = IntegrationBridgeConfig(_env_file=env_file)

    if config.environment.lower() == "production" and not config.security.enable_auth:
        raise ValueError("Production Integration Bridge requires authentication")

    websocket_bridge = WebSocketBridge(config)
    analytics_manager = AnalyticsManager(config)
    state_synchronizer = StateSynchronizer(config)
    security_middleware = SecurityMiddleware(config)

    app = create_app(config)
    app.state.websocket_bridge = websocket_bridge
    app.state.analytics_manager = analytics_manager
    app.state.state_synchronizer = state_synchronizer
    app.state.security_middleware = security_middleware

    @app.on_event("startup")
    async def start_background_services() -> None:
        """Start optional background integrations after the event loop exists."""
        await websocket_bridge.start_background_connections()
        app.state.metrics_task = await analytics_manager.start_metrics_collection()

    @app.on_event("shutdown")
    async def stop_background_services() -> None:
        """Stop analytics collection and await its task cancellation."""
        analytics_manager.stop_metrics_collection()
        metrics_task = getattr(app.state, "metrics_task", None)
        if metrics_task is not None and not metrics_task.done():
            metrics_task.cancel()
            await asyncio.gather(metrics_task, return_exceptions=True)

    return app


def main() -> None:
    """Run the integration bridge from the installed console command."""

    @click.command()
    @click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False),
                  help="Environment file containing bridge configuration")
    @click.option("--host", default="0.0.0.0", show_default=True)
    @click.option("--port", default=9000, type=click.IntRange(1, 65535), show_default=True)
    @click.option("--workers", default=1, type=click.IntRange(1), show_default=True)
    @click.option("--reload", is_flag=True, help="Enable development auto-reload")
    @click.option("--log-level", default="INFO", show_default=True)
    @click.option("--ssl-keyfile", type=click.Path(exists=True, dir_okay=False))
    @click.option("--ssl-certfile", type=click.Path(exists=True, dir_okay=False))
    @click.option("--ssl-ca-certs", type=click.Path(exists=True, dir_okay=False))
    def run(
        config: Optional[str],
        host: str,
        port: int,
        workers: int,
        reload: bool,
        log_level: str,
        ssl_keyfile: Optional[str],
        ssl_certfile: Optional[str],
        ssl_ca_certs: Optional[str],
    ) -> None:
        """Start the opencode ↔ LRS-Agents Integration Bridge."""
        if bool(ssl_keyfile) != bool(ssl_certfile):
            raise click.UsageError("--ssl-keyfile and --ssl-certfile must be supplied together")
        if ssl_ca_certs and not (ssl_keyfile and ssl_certfile):
            raise click.UsageError("--ssl-ca-certs requires both --ssl-keyfile and --ssl-certfile")

        if config:
            os.environ["INTEGRATION_BRIDGE_CONFIG"] = config

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
        logger.info("Starting Integration Bridge", host=host, port=port,
                    workers=workers, reload=reload)

        uvicorn.run(
            "opencode_lrs_bridge.main:create_bridge_application",
            factory=True,
            host=host,
            port=port,
            workers=1 if reload else workers,
            reload=reload,
            log_level=log_level.lower(),
            access_log=True,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_ca_certs=ssl_ca_certs,
            ssl_cert_reqs=ssl.CERT_REQUIRED if ssl_ca_certs else ssl.CERT_NONE,
        )

    run()


if __name__ == "__main__":
    main()
