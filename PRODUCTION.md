# Production Deployment

The live playground executes model-generated Python. Do not expose it with the
`local` execution environment. Production deployments must use an isolated
environment and must require a bearer token on run endpoints.

## Required Secrets

Set these in the backend deployment:

```bash
ANTHROPIC_API_KEY=...
E2B_API_KEY=...
RLM_RUN_API_TOKEN=...
```

Set these in the visualizer deployment:

```bash
RLM_BACKEND_URL=https://your-backend.example.com
RLM_RUN_API_TOKEN=...
RLM_ALLOW_PUBLIC_PLAYGROUND=1
```

`RLM_RUN_API_TOKEN` is the server-to-server token between the Next.js route and
the Python backend. Keep the Python backend private by policy and token-protected
even if it has a public URL.

For a private demo, set `RLM_PLAYGROUND_TOKEN` instead of
`RLM_ALLOW_PUBLIC_PLAYGROUND=1`. Clients must send it as either:

```http
Authorization: Bearer <RLM_PLAYGROUND_TOKEN>
```

or:

```http
X-RLM-Run-Token: <RLM_PLAYGROUND_TOKEN>
```

## Production Defaults

`render.yaml` sets:

- `RLM_PRODUCTION=true`
- `RLM_EXEC_ENVIRONMENT=e2b`
- `RLM_MAX_ITERATIONS=8`
- `RLM_MAX_DEPTH=2`
- `RLM_RUN_TIMEOUT_SECONDS=300`
- `RLM_MAX_DOCUMENT_CHARS=500000`
- `RLM_MAX_PROMPT_CHARS=8000`

With `RLM_PRODUCTION=true`, `run_playground.py` refuses `local` and `docker`
execution unless `RLM_ALLOW_LOCAL_EXEC=1` is set. Use that override only for a
trusted private deployment.

The Next.js route and Python backend both include lightweight in-memory rate
limits. Use provider-level WAF/rate limiting or a shared store such as Redis for
high-traffic public deployments, because in-memory limits are per process and
per serverless instance.

## Logging

User runs no longer write traces to `visualizer/public/logs` by default. To store
logs, set `RLM_LOG_DIR` to a private path. Do not set `RLM_ENABLE_PUBLIC_LOGS=1`
for user-submitted documents.
