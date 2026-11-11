# PrefixRelay

**A cache-aware routing proxy for Cursor.** PrefixRelay protects conversation
prefixes from avoidable KV-cache invalidation, routes each model name to its
configured endpoint, and compacts history using that model's context (CTX)
length.

```text
Cursor ── model name + conversation ──▶ PrefixRelay
                                          ├─ stabilize the cached prefix
                                          ├─ route to a backend or replica pool
                                          └─ compact before the model's CTX limit
                                                     │
                                                     ├── OpenRouter
                                                     ├── local server
                                                     └── any compatible endpoint
```

## Why put PrefixRelay in front of an OpenAI-compatible endpoint?

OpenAI compatibility describes the API shape. It lets Cursor send chat
completion requests, but it does not by itself manage Cursor's changing
conversation history, select among your backends, or keep a prompt under each
model's context limit. PrefixRelay adds those controls in one local proxy.

| Capability | Plain compatible endpoint | PrefixRelay |
|---|---|---|
| KV-cache safety | Depends on the client and server receiving a stable prefix | Keeps a canonical conversation prefix and restores earlier image content when Cursor changes it |
| Model routing | Often configured for one endpoint/model at a time | Routes Cursor-visible model names and aliases to backends, upstream IDs, or a pool of replicas |
| Context limits | Usually the caller must fit the request to the model window | Uses each route's CTX limit to decide when to elide old tool output and summarize older history |
| Credentials | A key is usually configured for the endpoint | Keeps a separate key source for each backend and a separate key for Cursor-to-proxy access |
| Cursor setup | You supply a compatible URL | Includes setup steps and a Cloudflare Quick Tunnel helper for temporary HTTPS access |

### 1. Keep KV cache reuse from breaking

Cursor resends conversation history on each agent turn. A changed earlier
message, volatile system field, or re-encoded image can alter the prefix that
the model server sees and invalidate its cached work. In default `stabilize`
mode, PrefixRelay keeps a canonical history and sends the stable earlier
messages forward, appending new turns. It also matches images by their place in
the conversation and can restore the original image payload when Cursor
changes or drops it.

This gives a cache-capable upstream a stable prefix it can reuse. PrefixRelay
does not own the model's KV cache or guarantee a hit: reuse still depends on
the upstream server/provider and its cache policy. When usage is available,
PrefixRelay records `cached_tokens` so you can see what the upstream reports.

### 2. Route each model to the right endpoint

The model name Cursor sends selects a route. Each route points to a backend
(such as OpenRouter or a local server) and the model ID that backend expects.
Aliases let built-in or custom Cursor model names select the same route. One
Cursor base URL can therefore serve hosted and local models without putting
provider keys into Cursor.

Unknown model names use the configured `default_model` route. Choose that
default deliberately: it determines where unrecognized names go.

#### One Cursor model name, several endpoints

A backend can also represent a pool of replicas. Cursor still selects a single
model name, such as `minion`; PrefixRelay then chooses among that backend's
endpoints. For each request, it prefers a healthy, available replica whose
recorded message history has the longest exact prefix in common with the
incoming conversation. This sends continuing sessions back to the endpoint
most likely to have their prompt prefix cached. If there's no history match,
the least-recently-used replica gets the new session; that spreads fresh work
across the pool. If all replicas are busy, requests wait; if none pass the
health check, PrefixRelay returns `503`.

```yaml
backends:
  minion-pool:
    base_url: http://127.0.0.1:9010
    replicas:
      - {base_url: http://127.0.0.1:9010, model: minion-1}
      - {base_url: http://127.0.0.1:9011, model: minion-2}
      - {base_url: http://127.0.0.1:9012, model: minion-3}
    stabilize: true
    compact_enabled: true

routes:
  - name: minion
    backend: minion-pool
    context_tokens: 131072
```

Each replica should serve the same model capability and context size, expose
`/health` for health checks, and provide OpenAI-compatible
`/v1/chat/completions`. PrefixRelay sends the replica's configured `model` ID
upstream. This is intended for separate model servers with independent KV
caches; the pool uses conversation-prefix overlap to steer requests toward the
cache that is most likely to be warm.

### 3. Compact history using the model's CTX length

Compaction is on by default. Each route can declare its model's context length
with `context_tokens`; local llama.cpp-compatible servers can also report it
through `/props`. PrefixRelay starts compaction at 80% of the known window by
default, leaving room for the current request and the model's answer.

When a prompt reaches the threshold, PrefixRelay first elides large older tool
outputs. If that is not enough, it asks the route's upstream model to summarize
older history while keeping the recent message tail and tool-call pairs. That
summary call uses the same backend credentials and can incur provider usage.
If the context size is unknown, compaction skips the request rather than
guessing. If the compacted projection still appears too large, PrefixRelay
keeps the original request; a very large recent tail can still exceed a model's
window.

## Observed cache reuse

Recent captures from the internal `cursor-proxy` setup show the reuse rates
PrefixRelay is designed to preserve. For requests with upstream usage data from
**November 5–11, 2026**:

| Backend group | Requests | Prompt tokens | Cached tokens | Prompt tokens cached |
|---|---:|---:|---:|---:|
| OpenRouter | 827 | 83.9M | 81.8M | **97.5%** |
| Local models | 2,271 | 156.0M | 151.1M | **96.9%** |

These are weighted ratios (`sum(cached_tokens) / sum(prompt_tokens)`) from one
real workload, not a benchmark or a before/after comparison. Providers report
cache usage differently, and results vary by model, prompt, and cache policy.
They illustrate observed reuse with PrefixRelay in the path; they are not a
performance guarantee.

## Quick start

You need Python 3.10 or newer. `cloudflared` is only needed if Cursor must
reach the proxy through a temporary public HTTPS URL.

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

On Linux, follow Cloudflare's package instructions for your distribution.

### 2. Install PrefixRelay

```bash
git clone https://github.com/jackjusko/PrefixRelay.git
cd PrefixRelay
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Create a local config and a random key for Cursor:

```bash
python scripts/setup.py
```

Edit `config.yaml` to set model IDs, context lengths, and endpoint URLs. The
generated proxy key is in `secrets/proxy.token`; use it as Cursor's OpenAI API
key. Provider keys are configured separately.

### 3. Add a provider key and choose routes

The example config includes an OpenRouter route. Set its key in the environment
before starting PrefixRelay:

```bash
# macOS / Linux
export OPENROUTER_API_KEY="your-openrouter-key"

# Windows PowerShell
$env:OPENROUTER_API_KEY = "your-openrouter-key"
```

`api_key_env` names the environment variable for a backend. You can use
`api_key_file` instead to load a key from an ignored local file. Never commit
provider keys or your generated proxy key.

### 4. Start the proxy

In one terminal, with the environment active:

```bash
python scripts/run_proxy.py
```

By default, PrefixRelay listens only on `http://127.0.0.1:8081`.

### 5. Start a Cloudflare Quick Tunnel

In a second terminal:

```bash
python scripts/quick_tunnel.py
```

The helper starts `cloudflared tunnel --url http://127.0.0.1:8081`, prints the
temporary HTTPS URL, and saves it to the ignored `tunnel.url`. Keep it running
while using Cursor. Quick Tunnels need no Cloudflare account or named tunnel;
their URL changes when restarted.

**Quick Tunnel limitation:** Cloudflare currently documents that Quick Tunnels
do not support Server-Sent Events. Cursor streams responses, so text may only
appear when generation ends, and long runs may time out. Quick Tunnels also
have a 200 in-flight request limit and no uptime guarantee. See
[Cloudflare's Quick Tunnel limitations](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations).

### 6. Configure Cursor

In Cursor Settings → Models, enable the OpenAI API key and base URL override.
Set:

- **Base URL:** the tunnel URL with `/v1`, for example
  `https://random-words.trycloudflare.com/v1`
- **API key:** the contents of `secrets/proxy.token`
- **Model:** a route name from `config.yaml`, such as `openrouter-default`

If Cursor asks you to add a custom model, enter the route's `name` exactly. To
route a built-in OpenAI-family model label, add that label under `aliases:`.
The proxy's `/v1/models` endpoint lists its routes and aliases.

Cursor's **Override OpenAI Base URL** setting applies globally to OpenAI-family
requests. Turn it off when you want those models to use Cursor's own service.
See [Cursor's explanation](https://forum.cursor.com/t/custom-base-urls-for-each-custom-model/147219/13).

## Configure models and endpoints

The `routes` table maps a name Cursor sends to a backend and upstream model ID.
Add a backend for each endpoint and a route for each model you want to select.
Any service that implements OpenAI Chat Completions can be configured.

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

Set `context_tokens` to the model's actual context length. It belongs on the
route so different models sharing a provider can have different limits.

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

For a local llama.cpp server, set `context_tokens: 0` to discover the window
from `/props`. Discovery runs automatically for localhost and private-IP
endpoints. For a local server on a custom DNS name, set `context_discovery: true`
on its backend. Public providers are never probed at `/props`; configure their
route's `context_tokens` explicitly. If the limit is unknown, compaction skips
that request and reports `no-n_ctx` in stats.

Set a separate key source per backend if needed:

```yaml
api_key_env: MY_LOCAL_KEY
# Or:
api_key_file: secrets/local-model.key
```

See [config.example.yaml](config.example.yaml) for the full starting config.

## Privacy and security

The Quick Tunnel makes a public HTTPS route to the local proxy. PrefixRelay
requires the generated bearer key for chat, model-list, health, and stats
requests. Keep it private and stop the tunnel when finished. Provider keys are
sent only to the backend selected for that route.

Request and response body recording is off by default. PrefixRelay records
timing, route, status, and usage metadata under `captures/`, but not prompt or
response text. Setting `record_bodies: true` writes request and forwarded
bodies to `bodies/`; response bodies also require
`log_response_bodies: true`.

The KV stabilizer keeps canonical conversation state, including image payloads,
in the ignored `state/` directory so prefixes can survive a restart. This is
local prompt data. Protect that directory and delete it to clear saved session
state. Never publish `config.yaml`, `secrets/`, `state/`, `captures/`,
`bodies/`, `tunnel.url`, or `.env` files.

## Development

```bash
python -m pip install -r requirements.txt pytest
python -m pytest
python scripts/test_routing.py
```

The routing test uses mocked endpoints. `scripts/probe_upstreams.py` can send
live requests through a running proxy; those calls may incur provider costs.

## License

PrefixRelay is licensed under the [Apache License 2.0](LICENSE).
