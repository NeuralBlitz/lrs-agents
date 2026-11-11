"""Fail-closed access and command policies for the local OpenCode HTTP bridge.

This is authorization and input validation, not a subprocess sandbox.
"""

import hmac
import os
import shlex
from pathlib import Path
from urllib.parse import urlsplit


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_READ_ONLY_SUBCOMMANDS = {"ls", "grep", "read", "question"}


def valid_bridge_authorization(authorization: str, secret: str) -> bool:
    """Require a 32+ character bearer secret; compare without timing leaks."""
    if not isinstance(secret, str) or len(secret) < 32:
        return False
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        return False
    presented = authorization[len("Bearer "):]
    return bool(presented) and hmac.compare_digest(presented, secret)


def is_loopback_host(host: str) -> bool:
    """Do not bind an unaudited command bridge to an external interface."""
    return host in _LOOPBACK_HOSTS


def validate_bridge_url(url: str) -> None:
    """Never send bridge credentials over cleartext to non-loopback hosts."""
    parsed = urlsplit(url)
    if parsed.username or parsed.password or not parsed.hostname:
        raise ValueError("Invalid bridge service URL")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and is_loopback_host(parsed.hostname):
        return
    raise ValueError("Bridge requests require HTTPS or loopback HTTP")


def resolve_bridge_directory(workspace: str, requested: str) -> str:
    """Keep the command's working directory inside the configured workspace."""
    base = Path(workspace).resolve()
    if not base.is_dir():
        raise ValueError("Bridge workspace does not exist")
    if not isinstance(requested, str) or not requested:
        raise ValueError("A working directory is required")
    target = (base / requested).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Working directory escapes configured workspace")
    if not target.is_dir():
        raise ValueError("Working directory does not exist")
    return str(target)


def build_opencode_command(
    executable: str,
    command: str,
    *,
    allow_shell: bool = False,
) -> list:
    """Parse arguments without shell expansion and restrict executable actions."""
    if not isinstance(executable, str) or not executable:
        raise ValueError("OpenCode executable is unavailable")
    if not isinstance(command, str) or not 0 < len(command) <= 4096:
        raise ValueError("Command must be a nonempty bounded string")
    try:
        args = shlex.split(command)
    except ValueError as error:
        raise ValueError("Invalid command quoting") from error
    if args and args[0] == "opencode":
        args = args[1:]
    if not args:
        raise ValueError("OpenCode subcommand is required")
    allowed = _READ_ONLY_SUBCOMMANDS | ({"bash"} if allow_shell else set())
    if args[0] not in allowed:
        raise ValueError("OpenCode subcommand is not permitted")
    if args[0] == "ls" and len(args) != 1:
        raise ValueError("ls does not accept bridge-supplied arguments")
    if args[0] in {"read", "grep"}:
        if len(args) != 2 or args[1].startswith("-"):
            raise ValueError("read/grep require exactly one positional argument")
    if args[0] == "question" and len(args) < 2:
        raise ValueError("question requires nonempty text")
    return [executable, *args]



def resolve_bridge_file(workspace: str, working_dir: str, filename: str) -> str:
    """Confine file-read arguments to the same permitted workspace."""
    base = Path(workspace).resolve()
    target = (Path(working_dir) / filename).resolve()
    if not target.is_relative_to(base):
        raise ValueError("File path escapes configured workspace")
    if not target.is_file():
        raise ValueError("File path is not an existing regular file")
    return str(target)


def configured_workspace() -> str:
    """Resolve the explicit workspace or the server's current directory."""
    return os.environ.get("LRS_BRIDGE_WORKSPACE", os.getcwd())
