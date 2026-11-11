# OpenCode HTTP bridge: local security contract

The bridge exposes agent and OpenCode actions over HTTP. It is **not** a
general-purpose remote execution service and does **not** sandbox OpenCode or
the underlying operating system. Keep it on a trusted machine, isolated from
untrusted networks, with the least privileges possible.

## Required configuration

Set a strong randomly generated bearer token before starting the bridge.
The bridge rejects all requests if `LRS_BRIDGE_TOKEN` is missing or shorter
than 32 characters. Do not put tokens in source files or commit them.

```sh
export LRS_BRIDGE_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export LRS_BRIDGE_WORKSPACE="$HOME/my-trusted-project"
```

The default workspace is the server process's current directory. Point it at
a dedicated project folder rather than a home directory or an entire drive.

The bridge only permits loopback listeners (`localhost`, `127.0.0.1`, `::1`).
HTTP clients can use plain HTTP only for loopback addresses; other hosts
require HTTPS and a separately reviewed proxy/authentication architecture.

## Allowed operations

By default, the bridge allows only the following OpenCode subcommands:

- `ls` (no arguments);
- `read PATH` (one existing regular file confined to the workspace);
- `grep PATTERN` (one positional search pattern);
- `question TEXT` (one or more words).

The bridge parses arguments using `shlex`, never via `shell=True`. This
does **not** guarantee that the OpenCode executable itself cannot perform
side effects, especially when using a model-backed `question` action.

Shell command execution (`bash ...`) is **disabled by default**. It is
available only with the explicit opt-in `LRS_BRIDGE_ALLOW_SHELL=1`. That mode
grants arbitrary command execution to anyone holding the bearer token and is
unsuitable for untrusted clients.

Working directories must resolve within `LRS_BRIDGE_WORKSPACE`. `read`
file paths are checked after symlink resolution and may not escape the
workspace. These checks do not form an OS sandbox and do not police external
resources that OpenCode may access.

Timeouts must be integers from 1 to 60 seconds. Command strings are limited
to 4096 characters. Successful calls can still generate large output; use
process-level resource limits when operating with untrusted workloads.

## Request example

```sh
curl --fail-with-body \
  -H "Authorization: Bearer ${LRS_BRIDGE_TOKEN}" \
  -H 'Content-Type: application/json' \
  --data '{"command":"ls","working_dir":".","timeout":10}' \
  http://127.0.0.1:8765/opencode/execute
```

The same authorization is required for `/lrs/create-agent` and
`/lrs/execute`. No external interface should be exposed solely because
a bearer token has been configured.

## Limits and outstanding security work

- A bearer token is not user-based identity, authorization scopes, audit
  logging, or key rotation.
- The OpenCode runtime itself may access network, filesystem, and other
  system resources. Use an isolated account/container and OS controls.
- The current implementation does not bound process output or implement
  request quotas or per-client rate limiting.
- Do not turn this bridge into a public production service or deploy it to
  Vercel without a separate architecture and security review.
