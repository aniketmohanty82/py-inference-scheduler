# vllm-router Integration with py-inference-scheduler

## Compatibility Notice

**This integration is designed for [our vllm-router fork](https://github.com/aniketmohanty82/router/tree/external-policy)**
(branch `external-policy`, based on [vllm-router v0.1.15](https://github.com/vllm-project/router/releases/tag/v0.1.15)),
validated against vLLM 0.23.0 engines. Stock vllm-router has no external-policy hook — the fork is required.

## Architecture

vllm-router is the vLLM ecosystem's Rust router. Our fork adds an external-policy hook:

- At launch, `--external-policy-factory` imports a factory and calls it once to obtain a selection callable.
- On every request, the router calls `select(workers, request_text, headers)` on a Rust thread and routes
  to the returned worker index.
- `None` or any exception falls back to a built-in policy (`--external-fallback-policy`, default
  `round_robin`), so the scheduler can never fail a request.
- Engine metrics are scraped off the request path by the shared `MetricsPoller`. Router-side inflight
  counts arrive with each call in the worker dicts.
- The router keeps full ownership of serving (streaming, retries, circuit breakers, worker management).
  We only decide which worker serves each request.

Key components:
- [adapter.py](./adapter.py): bridges the callable contract to `Scheduler` — lazy worker registry,
  metrics polling, index mapping.
- [factory.py](./factory.py): the target of `--external-policy-factory`; builds the `Scheduler` and
  returns `adapter.select`.
- [`__main__.py`](./__main__.py): the `python -m integration.vllm_router` launcher.

---

## Prerequisites (Step 1)

Install the fork and this repo into the same environment.

**Build and install the fork.** The build needs a Rust toolchain plus a C compiler, `pkg-config`, and
OpenSSL headers (Debian/Ubuntu: `apt install build-essential pkg-config libssl-dev`), and the extension
builds against the Python it is installed with:

```bash
git clone -b external-policy https://github.com/aniketmohanty82/router.git
cd router
pip install .
```

> [!NOTE]
> The build compiles the Rust extension silently for several minutes — this is normal.

**Clone this repo and install the scheduler**:

```bash
git clone https://github.com/llm-d-incubation/py-inference-scheduler.git
cd py-inference-scheduler
pip install -e . --no-deps
pip install aiohttp prometheus-client pyyaml setproctitle
```

## Integration Configuration (Step 2)

The routing policy reuses slime's [`examples/scheduler.yaml`](../slime/examples/scheduler.yaml) (the scorers
are engine-agnostic). Edit that file directly to customize, or pass
`--external-scheduler-config /path/to/your.yaml` to the launcher in Step 3. See the
[Scheduler Customization Guide](../../docs/scheduler_customization.md).

## Running the Router (Step 3)

**Start the router** — run from the `py-inference-scheduler` repo root. `--external-scheduler-config` is ours - every other flag is
forwarded to vllm-router unchanged (worker discovery, PD flags, timeouts — see `vllm-router --help`):

```bash
python -m integration.vllm_router \
  --external-scheduler-config integration/slime/examples/scheduler.yaml \
  --external-metrics-interval-ms 100 \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --host 0.0.0.0 --port 30000
```

Workers may also register at runtime through the router's `POST /workers` API; the adapter tracks whatever
worker set the router offers per request.

The launcher adds two flags of its own; every other flag is stock vllm-router:

- `--external-scheduler-config` (required): path to the scheduler yaml from Step 2.
- `--external-metrics-interval-ms` (default 100): how often the adapter polls each worker's `/metrics`.

What the router expects of each worker (real vLLM servers provide all of this out of the box):

- `GET /health`: the router health-gates on this **before serving**, blocking up to 600s until all
  `--worker-urls` answer.
- `GET /metrics`: Prometheus text. The scorers read `vllm:num_requests_waiting`,
  `vllm:num_requests_running`, and `vllm:kv_cache_usage_perc`; missing gauges are treated as zero.
- The inference endpoints the router proxies: `/v1/completions`, `/v1/chat/completions`, `/generate`, ...

Note the router renames its process to `vllm::router` (setproctitle), so match that name with
`pkill`/`pgrep`, not `python`.

## Verifying Results (Step 4)

The router prints to **stdout** — the terminal where you started it in Step 3. On startup, watch for the
scheduler config loading, the fork assigning registered workers to us (`Assigning policy external`), and
the adapter picking workers up: `Seeded worker <url>` for the startup `--worker-urls` set,
`Tracking worker <url>` for workers registered later via `POST /workers`.

Routing decisions are visible with the debug toggle `RLS_DECISION_LOG=1` set in the router's environment
(each request's per-endpoint stats at decision time) or `--log-level debug` (per-scorer raw scores and the
selected endpoint for every request). Either way the extra lines are formatted inside the scheduling call
itself, adding routing latency on every request while enabled — leave them off for benchmarks.
