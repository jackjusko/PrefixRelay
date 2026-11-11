# PrefixRelay

An OpenAI-compatible proxy for Cursor that routes requests to OpenRouter, local
inference servers, and other chat-completions endpoints. PrefixRelay keeps
conversation prefixes stable for KV/prompt-cache reuse and summarizes old
history before it overruns a model's context window.

```text
Cursor ── HTTPS Quick Tunnel ──▶ PrefixRelay on your computer ──▶ model API
                                       │
                                       ├── route by Cursor model name
                                       ├── stabilize cached prompt prefixes
                                       └── compact long context when needed
```

## Quick start

You need Python 3.10 or newer. `cloudflared` is needed only if Cursor must reach
your computer through a temporary public HTTPS URL. PrefixRelay itself binds to
`127.0.0.1` and does not need a Cloudflare account for Quick Tunnels.

### 1. Install prerequisites

Install Python from [python.org](https://www.python.org/downloads/) or your
package manager. Install `cloudflared` from
[Cloudflare's installation guide](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).

Common package-manager commands:

```bash
# macOS
brew install cloudflared

# Windows PowerShell (winget)
winget install --id Cloudflare.cloudflared
```

On Linux, follow Cloudflare's package instructions for your distribution. Check
both tools are available with `python --version` and `cloudflared --version`.

### 2. Download and install PrefixRelay

```bash
git clone https://github.com/jackjusko/PrefixRelay.git
cd PrefixRelay
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
# macOS / Linux
source .venv/bin/activate
python -m pip install -r requirements.txt

# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `config.yaml` and a random key for Cursor:

```bash
python scripts/setup.py
```

Edit `config.yaml` to set model IDs, context sizes, and endpoint URLs. The
generated key is in `secrets/proxy.token`; use it as the OpenAI API key in
Cursor. Keep provider API keys separate from this proxy key.

### 3. Add your provider key

The example routes to OpenRouter. Set its API key in the environment before
starting PrefixRelay:

```bash
# macOS / Linux
export OPENROUTER_API_KEY="your-openrouter-key"

# Windows PowerShell
$env:OPENROUTER_API_KEY = "your-openrouter-key"
```

`api_key_env` names the environment variable for each backend. You can instead
set `api_key_file` to an ignored local file, for example
`secrets/openrouter.key`. Never put provider keys in a public config or commit
them to git.

### 4. Start the proxy

In one terminal, with the virtual environment active:

```bash
python scripts/run_proxy.py
```

The default address is `http://127.0.0.1:8081`. Check it from another terminal
using your generated proxy key:

```bash
curl -H "Authorization: Bearer $(cat secrets/proxy.token)" \
  http://127.0.0.1:8081/healthz
```

PowerShell equivalent:

```powershell
$key = Get-Content secrets/proxy.token -Raw
Invoke-RestMethod -Headers @{ Authorization = "Bearer $($key.Trim())" } `
  http://127.0.0.1:8081/healthz
```

### 5. Start a Cloudflare Quick Tunnel

In a second terminal, from the PrefixRelay folder and with the environment
active:

```bash
python scripts/quick_tunnel.py
```

The script starts `cloudflared tunnel --url http://127.0.0.1:8081`, prints the
temporary HTTPS URL, and writes it to the ignored `tunnel.url` file. Keep this
terminal running while using Cursor. Stopping it ends the tunnel; the next
Quick Tunnel will usually have a different URL. No named tunnel or Cloudflare
credentials are used.

**Quick Tunnel limits:** Cloudflare currently documents that Quick Tunnels do
not support Server-Sent Events. Cursor requests streamed responses, so its UI
may receive the answer only after generation ends or time out on long runs.
Quick Tunnels also have a 200 in-flight request limit and no uptime guarantee.
They are useful for a short development session; see
[Cloudflare's Quick Tunnel limitations](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations).

### 6. Point Cursor at PrefixRelay

In Cursor Settings → Models, enable the OpenAI-compatible API / base URL
override. Set:

- **Base URL:** the printed Quick Tunnel URL with `/v1` appended, for example
  `https://random-words.trycloudflare.com/v1`
- **API key:** the contents of `secrets/proxy.token`
- **Model:** a configured route name, such as `openrouter-default`

If Cursor asks you to add a custom model name, enter the route's `name` from
`config.yaml` exactly. For a built-in OpenAI-family model label, add a matching
entry under `aliases:` so that label routes to the intended backend.

If Cursor's version presents these fields with different labels, use its
OpenAI-compatible provider settings. The proxy's `/v1/models` endpoint returns
the configured route names. Restart Cursor if it caches the model list.

Cursor currently applies **Override OpenAI Base URL** globally to OpenAI-family
requests. Turn it off when you want Cursor's own managed OpenAI-family models
to use Cursor's service again; the override can route those requests to
PrefixRelay too. See [Cursor's current explanation](https://forum.cursor.com/t/custom-base-urls-for-each-custom-model/147219/13).

## Configure models and endpoints

The routing table in `config.yaml` maps the model name Cursor sends to an
upstream backend and model ID. You can add as many routes and backends as you
need. Any provider implementing OpenAI Chat Completions can be configured by
setting its base URL, API key source, model ID, and context length.

### OpenRouter

```yaml
backends:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    usage_style: usage_object
    stabilize: true
    compact_enabled: true

routes:
  - name: my-model
    backend: openrouter
    upstream_model: provider/model-id
    context_tokens: 128000
```

Use the context size supported by the actual model/provider route. This value
is per Cursor-visible model, so two routes sharing an OpenRouter backend can
have different limits.

### Local models

```yaml
backends:
  local:
    base_url: http://127.0.0.1:1234/v1
    stabilize: true
    compact_enabled: true

routes:
  - name: local-model
    backend: local
    upstream_model: model-name-served-by-your-server
    context_tokens: 32768
```

This works with local servers that expose OpenAI-compatible
`/v1/chat/completions`. Set `context_tokens` to the model's real context limit.
For llama.cpp servers, `context_tokens: 0` lets PrefixRelay try `/props` to
discover the configured context. If a context size is unknown or discovery
fails, compaction safely skips that request and reports the reason in stats.
Context discovery runs automatically for localhost and private-IP endpoints.
For a local server behind a custom DNS name, set `context_discovery: true` on
its backend. Public providers are never probed at `/props`; configure their
route's `context_tokens` explicitly.

You can use a different credential variable or a key file for each backend:

```yaml
api_key_env: MY_LOCAL_KEY
# Or:
api_key_file: secrets/local-model.key
```

Aliases let Cursor model labels route to an existing configured route:

```yaml
aliases:
  gpt-4o-mini: my-model
```

See [config.example.yaml](config.example.yaml) for a complete starting config.

## What PrefixRelay does

- **Prefix stabilization:** In default `stabilize` mode, canonicalizes each
  conversation so previous messages and image slots remain stable across
  Cursor turns. This helps compatible local KV caches and upstream prompt
  caches reuse the longest matching prefix.
- **Context-aware compaction:** Enabled by default at 80% of the configured or
  discovered context window. It trims large old tool outputs first, then
  summarizes older history while keeping recent messages and tool-call pairs.
  The summary request uses the route's upstream model and credentials, so it
  can incur provider usage. Set `compact_enabled: false` on a backend to turn
  it off, or adjust `compact_at`, `compact_tail_msgs`, and
  `compact_summary_max_tokens`.
- **OpenAI-compatible API:** Supports `/v1/chat/completions`, streaming, tool
  calls, `/v1/models`, and non-streaming chat completions.
- **Routing:** Route names select backend/model pairs, allowing one Cursor
  proxy URL to serve local and hosted models.

## Privacy and security

The Quick Tunnel gives a public HTTPS route to your local proxy. PrefixRelay
requires the generated bearer key on chat, model-list, health, and stats
requests. Keep the key private and stop the tunnel when you are finished.
Provider keys are sent only to the backend configured for that route.

Request/response body recording is off by default. The recorder still stores
request timing, route, status, and usage metadata under `captures/`; it does
not include prompt text or response text. If you set `record_bodies: true`,
request and forwarded bodies are written under `bodies/`; response bodies are
also saved only when `log_response_bodies: true`.

The cache stabilizer separately keeps canonical conversation state (including
image payloads) in the ignored `state/` directory so it can reconstruct stable
prefixes after a restart. This is local prompt data. Protect that directory,
and delete it when you want to clear cached conversation state. Never publish
`config.yaml`, `secrets/`, `state/`, `captures/`, `bodies/`, `tunnel.url`, or
`.env` files.

`/healthz`, `/stats`, and `/v1/models` require the proxy bearer key. Health
output reports whether an upstream key is configured, never the key itself.

## Troubleshooting

- **401 from Cursor:** copy the current value in `secrets/proxy.token` into
  Cursor's OpenAI API key field.
- **No traffic after a restart:** restart the Quick Tunnel and update Cursor's
  Base URL if the temporary hostname changed.
- **Unknown model or route:** make Cursor's model label match a `routes[].name`
  or configured alias, and check `default_model` in `config.yaml`.
- **Compaction says `no-n_ctx`:** set `context_tokens` on that route, or use a
  supported local `/props` endpoint. Compaction intentionally skips when the
  limit is unknown.
- **Provider 401/403:** check the backend's `api_key_env` or `api_key_file` and
  restart PrefixRelay after changing environment variables.
- **Local upstream unreachable:** verify the server URL and that it exposes
  `/v1/chat/completions`.
- **Cursor waits without showing tokens:** this is a Quick Tunnel SSE buffering
  limitation. The response may arrive only when generation completes.

## Development

Install dependencies, then run the offline test suite:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest
python scripts/test_routing.py
```

`scripts/test_routing.py` exercises routing and authentication against mocked
upstreams. The live smoke tool `scripts/probe_upstreams.py` can test configured
routes through a running proxy. It sends real requests and may incur provider
costs.

## License

PrefixRelay is licensed under the [Apache License 2.0](LICENSE).
