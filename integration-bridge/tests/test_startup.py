"""Startup contract tests for the Integration Bridge application factory."""

from fastapi.testclient import TestClient

from opencode_lrs_bridge.main import create_bridge_application
from opencode_lrs_bridge.utils.analytics import AnalyticsManager
from opencode_lrs_bridge.websocket.manager import WebSocketBridge


async def _no_background_connections(self):
    return None


async def _no_metrics_collection(self):
    return None


def _no_metrics_stop(self):
    return None


def test_application_factory_exposes_health_endpoint(
    test_config, monkeypatch
):
    """The Uvicorn factory must synchronously return a working ASGI app."""
    monkeypatch.setattr(
        WebSocketBridge,
        "start_background_connections",
        _no_background_connections,
    )
    monkeypatch.setattr(
        AnalyticsManager,
        "start_metrics_collection",
        _no_metrics_collection,
    )
    monkeypatch.setattr(
        AnalyticsManager,
        "stop_metrics_collection",
        _no_metrics_stop,
    )

    app = create_bridge_application(test_config)

    assert app.state.websocket_bridge is not None
    assert app.state.analytics_manager is not None
    assert app.state.state_synchronizer is not None
    assert app.state.security_middleware is not None

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
