"""Security regressions for the local-only OpenCode bridge policy.

All tests use pure validation functions. No shell commands, HTTP calls, or
actual OpenCode binary are executed.
"""

import os
from pathlib import Path

import pytest

from lrs.opencode.bridge_security import (
    build_opencode_command,
    is_loopback_host,
    resolve_bridge_directory,
    resolve_bridge_file,
    valid_bridge_authorization,
    validate_bridge_url,
)


def test_bearer_token_must_be_configured_and_match_exactly():
    secret = "s" * 32
    assert valid_bridge_authorization("Bearer " + secret, secret)
    assert not valid_bridge_authorization("Bearer " + "x" * 32, secret)
    assert not valid_bridge_authorization("Basic " + secret, secret)
    assert not valid_bridge_authorization("Bearer " + secret, "short")
    assert not valid_bridge_authorization("", secret)
    assert not valid_bridge_authorization("Bearer " + secret, "")


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_server_accepts_loopback_hosts(host):
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.1", "example.org", ""])
def test_server_rejects_non_loopback_hosts(host):
    assert not is_loopback_host(host)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8765",
        "http://127.0.0.1:8765",
        "http://[::1]:8765",
        "https://trusted.example/api",
    ],
)
def test_client_allows_loopback_http_or_https(url):
    validate_bridge_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://bridge.example/api",
        "http://192.168.1.9:8765",
        "file:///tmp/socket",
        "https://user:password@bridge.example",
        "not-a-url",
    ],
)
def test_client_rejects_insecure_or_malformed_urls(url):
    with pytest.raises(ValueError):
        validate_bridge_url(url)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls", ["opencode", "ls"]),
        ("opencode ls", ["opencode", "ls"]),
        ('grep "two words"', ["opencode", "grep", "two words"]),
        ("read notes.txt", ["opencode", "read", "notes.txt"]),
        ("question what is Python", ["opencode", "question", "what", "is", "Python"]),
    ],
)
def test_command_parser_keeps_supported_operations_as_argument_lists(command, expected):
    assert build_opencode_command("opencode", command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "bash rm -rf files",
        "opencode bash echo test",
        "run script",
        "ls ../../",
        "grep -R /",
        "read --config",
        "read",
        "question",
        "--help",
        "opencode",
        "",
        'grep "unterminated',
        "x" * 4097,
    ],
)
def test_command_parser_rejects_unsafe_or_invalid_requests(command):
    with pytest.raises(ValueError):
        build_opencode_command("opencode", command)


def test_shell_execution_requires_explicit_opt_in():
    with pytest.raises(ValueError):
        build_opencode_command("opencode", "bash echo hello")
    assert build_opencode_command(
        "opencode", "bash echo hello", allow_shell=True
    ) == ["opencode", "bash", "echo", "hello"]


def test_working_directory_cannot_escape_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    assert resolve_bridge_directory(str(root), "nested") == str(nested)
    assert resolve_bridge_directory(str(root), ".") == str(root)
    with pytest.raises(ValueError, match="escapes"):
        resolve_bridge_directory(str(root), "..")
    with pytest.raises(ValueError, match="exist"):
        resolve_bridge_directory(str(root), "nonexistent")


def test_read_target_must_be_regular_file_inside_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    inside = root / "file.txt"
    inside.write_text("data")
    outside = tmp_path / "private.txt"
    outside.write_text("private")

    assert resolve_bridge_file(str(root), str(root), "file.txt") == str(inside)
    with pytest.raises(ValueError, match="escapes"):
        resolve_bridge_file(str(root), str(root), "../private.txt")
    with pytest.raises(ValueError, match="regular file"):
        resolve_bridge_file(str(root), str(root), "missing.txt")


def test_symlink_escape_is_denied(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("private")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not supported by this environment")
    with pytest.raises(ValueError, match="escapes"):
        resolve_bridge_file(str(root), str(root), "link.txt")


def test_fastapi_endpoints_require_bearer_guard_when_dependency_available():
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from lrs.opencode.lrs_opencode_bridge import OpenCodeLRSBridge

    bridge = object.__new__(OpenCodeLRSBridge)
    app = bridge._create_api_app()
    protected = {
        "/lrs/create-agent",
        "/lrs/execute",
        "/opencode/execute",
    }
    routes = {route.path: route for route in app.routes if route.path in protected}
    assert set(routes) == protected
    for route in routes.values():
        names = [item.dependency.__name__ for item in route.dependencies]
        assert "require_bridge_authorization" in names
