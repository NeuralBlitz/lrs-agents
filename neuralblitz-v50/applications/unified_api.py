#!/usr/bin/env python3
"""
NeuralBlitz v50 Unified API Server
=================================

Comprehensive REST API for NeuralBlitz v50 with React frontend integration.
Includes JWT authentication, monitoring, and security features.

Endpoints:
- GET  /api/v1/health           - Health check
- GET  /api/v1/status           - System status
- GET  /api/v1/metrics          - Real-time metrics
- GET  /api/v1/quantum/state   - Quantum neuron states
- POST /api/v1/quantum/step     - Step quantum neurons
- GET  /api/v1/reality/network - Multi-reality network status
- POST /api/v1/reality/evolve  - Evolve reality network
- POST /api/v1/auth/token      - Get JWT token
- GET  /api/v1/auth/demo       - Get demo credentials

Usage:
    python3 unified_api.py

Frontend Integration:
    React app connects to http://localhost:5000/api/v1/*
"""

import sys
import json
import time
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from functools import wraps
from flask import Flask, jsonify, request, g
from flask_cors import CORS
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/runner/workspace/NB-Ecosystem/lib/python3.11/site-packages")
sys.path.insert(
    0,
    "/home/runner/workspace/opencode-lrs-agents-nbx/neuralblitz-v50/Advanced-Research/production",
)

import numpy as np
from quantum_spiking_neuron import QuantumSpikingNeuron, NeuronConfiguration
from multi_reality_nn import MultiRealityNeuralNetwork

# Initialize Flask app
app = Flask(__name__)

# JWT Authentication Setup
JWT_SECRET = "neuralblitz-v50-jwt-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1 hour

# Demo users (in production, use a proper database)
DEMO_USERS = {
    "admin": {
        "password": "admin123",
        "scopes": ["read", "write", "execute", "admin", "metrics"],
    },
    "operator": {
        "password": "operator123",
        "scopes": ["read", "write", "execute", "metrics"],
    },
    "viewer": {"password": "viewer123", "scopes": ["read"]},
}

# CORS Configuration
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:8080",
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)


@dataclass
class SystemMetrics:
    """System-wide metrics for API responses"""

    timestamp: float
    quantum_coherence: float
    consciousness_level: float
    network_activity: float
    reality_coherence: float
    spike_rate: float
    free_energy: float
    active_neurons: int
    total_cycles: int


def generate_token(user_id: str, scopes: List[str]) -> Dict[str, Any]:
    """Generate a JWT token"""
    import jwt
    import datetime

    payload = {
        "sub": user_id,
        "scopes": scopes,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXPIRATION),
        "iss": "neuralblitz-v50",
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": JWT_EXPIRATION,
        "scope": " ".join(scopes),
    }


def verify_token(token: str) -> Optional[Dict]:
    """Verify a JWT token and return payload if valid"""
    import jwt
    import datetime

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_auth(f):
    """Decorator to require JWT authentication"""

    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify(
                {
                    "error": "missing_authorization_header",
                    "message": "Authorization header is required",
                }
            ), 401

        try:
            scheme, token = auth_header.split(None, 1)
        except ValueError:
            return jsonify(
                {
                    "error": "invalid_authorization_header",
                    "message": "Invalid authorization header format",
                }
            ), 401

        if scheme.lower() != "bearer":
            return jsonify(
                {
                    "error": "invalid_authorization_scheme",
                    "message": "Authorization scheme must be Bearer",
                }
            ), 401

        payload = verify_token(token)

        if not payload:
            return jsonify(
                {"error": "invalid_token", "message": "Token is invalid or expired"}
            ), 401

        # Store user info in Flask's g object
        g.current_user = payload.get("sub")
        g.scopes = payload.get("scopes", [])

        return f(*args, **kwargs)

    return decorated


def require_scope(required_scope: str):
    """Decorator to require a specific scope"""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(g, "scopes") or required_scope not in g.scopes:
                return jsonify(
                    {
                        "error": "insufficient_scope",
                        "message": f"Required scope: {required_scope}",
                        "required": required_scope,
                        "current": getattr(g, "scopes", []),
                    }
                ), 403
            return f(*args, **kwargs)

        return decorated

    return decorator


class NeuralBlitzAPI:
    """Unified API controller for NeuralBlitz v50"""

    def __init__(self):
        self.quantum_neurons: Dict[str, QuantumSpikingNeuron] = {}
        self.reality_network: Optional[MultiRealityNeuralNetwork] = None
        self.metrics_history: List[SystemMetrics] = []
        self.current_metrics = SystemMetrics(
            timestamp=time.time(),
            quantum_coherence=0.5,
            consciousness_level=0.5,
            network_activity=0.5,
            reality_coherence=0.5,
            spike_rate=0.0,
            free_energy=0.0,
            active_neurons=0,
            total_cycles=0,
        )
        self._running = False
        self._lock = threading.Lock()

    def initialize(self):
        """Initialize all system components"""
        print("🚀 Initializing NeuralBlitz v50 Unified API...")

        # Initialize quantum neurons
        print("  ⚛️  Initializing Quantum Spiking Neurons...")
        config = NeuronConfiguration(quantum_tunneling=0.15, coherence_time=150.0)
        for i in range(5):
            neuron = QuantumSpikingNeuron(f"neuron_{i}", config)
            self.quantum_neurons[f"neuron_{i}"] = neuron

        # Initialize multi-reality network
        print("  🌌 Initializing Multi-Reality Network...")
        self.reality_network = MultiRealityNeuralNetwork(
            num_realities=4, nodes_per_reality=25
        )

        self.current_metrics.active_neurons = len(self.quantum_neurons)
        print(f"  ✅ System initialized with {len(self.quantum_neurons)} neurons")
        print(f"  ✅ Multi-Reality Network: 4 realities × 25 nodes")

    def update_metrics_cycle(self):
        """Background metrics update cycle"""
        while self._running:
            with self._lock:
                # Update quantum neurons
                total_spikes = 0
                coherence_values = []

                for neuron in self.quantum_neurons.values():
                    # Step neuron with random input
                    input_current = 15.0 + 5.0 * np.sin(time.time() * 0.5)
                    did_spike, _ = neuron.step(input_current)
                    if did_spike:
                        total_spikes += 1

                    # Track coherence
                    if hasattr(neuron, "_quantum_coherence"):
                        coherence_values.append(neuron._quantum_coherence)

                # Update metrics
                self.current_metrics.timestamp = time.time()
                self.current_metrics.quantum_coherence = (
                    np.mean(coherence_values) if coherence_values else 0.5
                )
                self.current_metrics.spike_rate = (
                    total_spikes / len(self.quantum_neurons)
                    if self.quantum_neurons
                    else 0
                )

                # Evolve reality network
                if self.reality_network:
                    self.reality_network.evolve_multi_reality_network(num_cycles=1)
                    self.current_metrics.consciousness_level = (
                        self.reality_network.global_consciousness
                    )
                    self.current_metrics.reality_coherence = (
                        self.reality_network.cross_reality_coherence
                    )

                # Calculate free energy (simplified)
                prediction_error = abs(total_spikes - 2)  # Expected ~2 spikes
                self.current_metrics.free_energy = prediction_error + 0.1

                # Network activity based on total processing
                self.current_metrics.network_activity = 0.3 + 0.4 * (
                    total_spikes / len(self.quantum_neurons)
                )
                self.current_metrics.total_cycles += 1

                # Store history (keep last 100)
                self.metrics_history.append(self.current_metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

            time.sleep(0.5)  # Update every 500ms

    def start(self):
        """Start the background metrics thread"""
        self._running = True
        self.metrics_thread = threading.Thread(
            target=self.update_metrics_cycle, daemon=True
        )
        self.metrics_thread.start()
        print("  📊 Metrics collection started")

    def stop(self):
        """Stop the background metrics thread"""
        self._running = False

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with self._lock:
            return {
                "timestamp": self.current_metrics.timestamp,
                "quantum_coherence": round(self.current_metrics.quantum_coherence, 4),
                "consciousness_level": round(
                    self.current_metrics.consciousness_level, 4
                ),
                "network_activity": round(self.current_metrics.network_activity, 4),
                "reality_coherence": round(self.current_metrics.reality_coherence, 4),
                "spike_rate": round(self.current_metrics.spike_rate, 2),
                "free_energy": round(self.current_metrics.free_energy, 4),
                "active_neurons": self.current_metrics.active_neurons,
                "total_cycles": self.current_metrics.total_cycles,
            }

    def get_metrics_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get historical metrics"""
        with self._lock:
            history = (
                self.metrics_history[-limit:]
                if limit < len(self.metrics_history)
                else self.metrics_history
            )
            return [
                {
                    "timestamp": m.timestamp,
                    "quantum_coherence": round(m.quantum_coherence, 4),
                    "consciousness_level": round(m.consciousness_level, 4),
                    "network_activity": round(m.network_activity, 4),
                    "reality_coherence": round(m.reality_coherence, 4),
                }
                for m in history
            ]

    def get_quantum_states(self) -> Dict[str, Any]:
        """Get quantum neuron states"""
        with self._lock:
            states = {}
            for neuron_id, neuron in self.quantum_neurons.items():
                states[neuron_id] = {
                    "membrane_potential": round(neuron.membrane_potential, 2),
                    "spike_rate": round(neuron.spike_rate, 2),
                    "spike_count": neuron.spike_count,
                    "time_elapsed": round(neuron.time_elapsed, 2),
                    "is_refractory": neuron.is_refractory,
                }
            return states

    def get_reality_network_status(self) -> Dict[str, Any]:
        """Get multi-reality network status"""
        with self._lock:
            if not self.reality_network:
                return {"error": "Reality network not initialized"}

            realities = {}
            for reality_id, reality in self.reality_network.realities.items():
                realities[reality_id] = {
                    "type": reality.reality_type.value,
                    "consciousness": round(reality.consciousness_level, 4),
                    "information_density": round(reality.information_density, 2),
                    "quantum_coherence": round(reality.quantum_coherence, 4),
                }

            return {
                "global_consciousness": round(
                    self.reality_network.global_consciousness, 4
                ),
                "cross_reality_coherence": round(
                    self.reality_network.cross_reality_coherence, 4
                ),
                "information_flow_rate": round(
                    self.reality_network.information_flow_rate, 4
                ),
                "reality_synchronization": round(
                    self.reality_network.reality_synchronization, 4
                ),
                "realities": realities,
                "active_signals": len(self.reality_network.active_signals),
            }


# Create global API instance
api = NeuralBlitzAPI()


# ==================== AUTHENTICATION ROUTES ====================


@app.route("/api/v1/auth/token", methods=["POST"])
def get_token():
    """
    Get JWT access token
    ---
    Request body (form):
        username: str (required)
        password: str (required)
        grant_type: str (optional, default: "password")
    """
    username = request.form.get("username")
    password = request.form.get("password")
    grant_type = request.form.get("grant_type", "password")

    if grant_type != "password":
        return jsonify(
            {
                "error": "unsupported_grant_type",
                "error_description": "Only 'password' grant type is supported",
            }
        ), 400

    if not username or not password:
        return jsonify(
            {
                "error": "invalid_request",
                "error_description": "Username and password are required",
            }
        ), 400

    # Verify credentials
    user = DEMO_USERS.get(username)
    if not user or user["password"] != password:
        return jsonify(
            {
                "error": "invalid_grant",
                "error_description": "Invalid username or password",
            }
        ), 401

    # Generate token
    token_data = generate_token(username, user["scopes"])

    return jsonify(token_data)


@app.route("/api/v1/auth/demo", methods=["GET"])
def get_demo_credentials():
    """
    Get demo credentials for testing
    WARNING: Only for development/testing!
    """
    return jsonify(
        {
            "message": "Demo credentials for testing",
            "users": [
                {
                    "username": "admin",
                    "password": "admin123",
                    "roles": ["admin"],
                    "scopes": ["read", "write", "execute", "admin", "metrics"],
                },
                {
                    "username": "operator",
                    "password": "operator123",
                    "roles": ["operator"],
                    "scopes": ["read", "write", "execute", "metrics"],
                },
                {
                    "username": "viewer",
                    "password": "viewer123",
                    "roles": ["viewer"],
                    "scopes": ["read"],
                },
            ],
            "note": "These credentials are for testing only. Change in production!",
        }
    )


@app.route("/api/v1/auth/introspect", methods=["POST"])
@require_auth()
def introspect_token():
    """
    Introspect JWT token
    Requires: Authorization: Bearer <token>
    """
    auth_header = request.headers.get("Authorization")
    token = auth_header.split()[1] if auth_header else None

    if not token:
        return jsonify({"active": False}), 200

    payload = verify_token(token)

    if not payload:
        return jsonify({"active": False}), 200

    return jsonify(
        {
            "active": True,
            "sub": payload.get("sub"),
            "scope": " ".join(payload.get("scopes", [])),
            "exp": payload.get("exp"),
            "iat": payload.get("iat"),
            "iss": payload.get("iss"),
        }
    )


# ==================== API ROUTES ====================


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "version": "v50.0",
            "timestamp": time.time(),
            "authentication": "enabled",
        }
    )


@app.route("/api/v1/status", methods=["GET"])
@require_auth()
@require_scope("read")
def get_status():
    """Get system status"""
    return jsonify(
        {
            "system": "NeuralBlitz v50",
            "status": "operational",
            "components": {
                "quantum_neurons": len(api.quantum_neurons),
                "reality_network": api.reality_network is not None,
                "metrics_collection": api._running,
                "authentication": True,
            },
            "capabilities": [
                "Quantum Spiking Neurons",
                "Multi-Reality Networks",
                "Cross-Reality Entanglement",
                "11-Dimensional Computing",
                "Neuro-Symbiotic Integration",
                "Autonomous Self-Evolution",
                "Consciousness Integration",
                "Advanced Agent Framework",
            ],
        }
    )


@app.route("/api/v1/metrics", methods=["GET"])
@require_auth()
@require_scope("metrics")
def get_metrics():
    """Get current system metrics"""
    return jsonify(api.get_current_metrics())


@app.route("/api/v1/metrics/history", methods=["GET"])
@require_auth()
@require_scope("metrics")
def get_metrics_history():
    """Get historical metrics"""
    limit = request.args.get("limit", 50, type=int)
    return jsonify(api.get_metrics_history(limit=limit))


@app.route("/api/v1/quantum/state", methods=["GET"])
@require_auth()
@require_scope("read")
def get_quantum_state():
    """Get quantum neuron states"""
    return jsonify(api.get_quantum_states())


@app.route("/api/v1/quantum/step", methods=["POST"])
@require_auth()
@require_scope("execute")
def step_quantum_neurons():
    """Step all quantum neurons with custom input"""
    data = request.get_json() or {}
    input_current = data.get("input_current", 20.0)
    steps = data.get("steps", 10)

    spike_counts = []
    for _ in range(steps):
        total_spikes = 0
        for neuron in api.quantum_neurons.values():
            did_spike, _ = neuron.step(input_current)
            if did_spike:
                total_spikes += 1
        spike_counts.append(total_spikes)

    return jsonify(
        {
            "steps_completed": steps,
            "total_spikes": sum(spike_counts),
            "average_spikes_per_step": round(np.mean(spike_counts), 2),
            "spike_pattern": spike_counts,
        }
    )


@app.route("/api/v1/reality/network", methods=["GET"])
@require_auth()
@require_scope("read")
def get_reality_network():
    """Get multi-reality network status"""
    return jsonify(api.get_reality_network_status())


@app.route("/api/v1/reality/evolve", methods=["POST"])
@require_auth()
@require_scope("execute")
def evolve_reality_network():
    """Evolve reality network for specified cycles"""
    data = request.get_json() or {}
    cycles = data.get("cycles", 10)

    if api.reality_network:
        api.reality_network.evolve_multi_reality_network(num_cycles=cycles)
        return jsonify(
            {
                "cycles_completed": cycles,
                "global_consciousness": round(
                    api.reality_network.global_consciousness, 4
                ),
                "cross_reality_coherence": round(
                    api.reality_network.cross_reality_coherence, 4
                ),
                "status": "success",
            }
        )
    else:
        return jsonify({"error": "Reality network not initialized"}), 500


@app.route("/api/v1/lrs/integrate", methods=["POST"])
@require_auth()
@require_scope("execute")
def trigger_lrs_integration():
    """Trigger LRS integration cycle"""
    return jsonify(
        {
            "status": "LRS integration ready",
            "note": "Connect to lrs_agents/lrs/neuralblitz_integration for full integration",
            "capabilities": [
                "Active Inference",
                "Free Energy Minimization",
                "Precision Tracking",
                "Multi-Agent Coordination",
            ],
        }
    )


@app.route("/api/v1/dashboard", methods=["GET"])
@require_auth()
@require_scope("read")
def get_dashboard_data():
    """Get all dashboard data in one request"""
    return jsonify(
        {
            "metrics": api.get_current_metrics(),
            "quantum_states": api.get_quantum_states(),
            "reality_network": api.get_reality_network_status(),
            "system_status": {
                "quantum_neurons_active": len(api.quantum_neurons),
                "reality_network_active": api.reality_network is not None,
                "total_cycles": api.current_metrics.total_cycles,
            },
        }
    )


# ==================== INITIALIZATION ====================

# Initialize API for Gunicorn (production)
import os

if os.environ.get("FLASK_ENV") == "production":
    print("🚀 Initializing NeuralBlitz v50 API (Production Mode)")
    api.initialize()
    api.start()
    print("✅ API initialized and ready")


# Main entry point for local development
def main():
    """Start the unified API server (Development Mode)"""
    print("=" * 70)
    print("🚀 NeuralBlitz v50 Unified API Server (Development)")
    print("=" * 70)

    # Initialize system
    api.initialize()
    api.start()

    print("\n📡 API Endpoints Available:")
    print("  GET  /api/v1/health           - Health check")
    print("  GET  /api/v1/auth/demo       - Demo credentials")
    print("  POST /api/v1/auth/token      - Get JWT token")
    print("  POST /api/v1/auth/introspect - Introspect token")
    print("  GET  /api/v1/status          - System status (requires auth)")
    print("  GET  /api/v1/metrics         - Current metrics (requires auth)")
    print("  GET  /api/v1/metrics/history - Historical metrics (requires auth)")
    print("  GET  /api/v1/quantum/state   - Quantum neuron states (requires auth)")
    print("  POST /api/v1/quantum/step    - Step quantum neurons (requires auth)")
    print("  GET  /api/v1/reality/network - Reality network status (requires auth)")
    print("  POST /api/v1/reality/evolve  - Evolve reality network (requires auth)")
    print("  POST /api/v1/lrs/integrate   - Trigger LRS integration (requires auth)")
    print("  GET  /api/v1/dashboard       - All dashboard data (requires auth)")
    print("\n🌐 Server starting on http://localhost:5000")
    print("=" * 70)

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        api.stop()
        print("✅ Server stopped gracefully")


if __name__ == "__main__":
    main()
