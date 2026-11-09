"""Security configuration regression tests for NeuralBlitz v50."""

import os
import subprocess
import sys
from pathlib import Path


def test_production_import_fails_without_jwt_secret():
    app_dir = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["FLASK_ENV"] = "production"
    environment.pop("JWT_SECRET", None)
    environment.pop("ENABLE_DEMO_AUTH", None)
    environment.pop("NEURALBLITZ_DEMO_USERS_JSON", None)
    environment["PYTHONPATH"] = str(app_dir)

    result = subprocess.run(
        [sys.executable, "-c", "import applications.unified_api"],
        cwd=app_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "JWT_SECRET must be configured in production" in result.stderr
