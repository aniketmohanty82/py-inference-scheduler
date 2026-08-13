---
name: vllm-router-integration
description: >-
  Guide to integrate py-inference-scheduler into vllm-router as the routing
  policy via the external-policy fork, and to troubleshoot broken builds,
  launches, routing, or metrics. Use when a user runs vllm-router for RL
  rollout traffic (any platform, any RL framework) and wants this repo's
  scorers deciding which worker serves each request.
---

# vllm-router Scheduler Integration Skill

This skill provides the architectural context, setup decisions, and diagnostic flows required to wire `py-inference-scheduler` into vllm-router through [our external-policy fork](https://github.com/aniketmohanty82/router/tree/external-policy) (branch `external-policy`, based on vllm-router v0.1.15).

Before integrating or debugging, always read the [vllm-router Integration README](../../../integration/vllm_router/README.md) — it is the canonical install / configure / run / verify sequence. This skill adds the setup-specific decisions and failure modes the README does not cover.

---

## 1. Initial Triage: Establish the Setup

This integration is always used for RL training traffic (bursty rollout dispatch). Before executing any steps, you **MUST** ask the user to clarify their setup if it is not already explicitly clear from the conversation history. Ask the user:

1. **RL framework**: Which framework drives the rollouts? If it is **slime, miles, or vime**, a dedicated integration already exists ([integration/slime](../../../integration/slime), [integration/vime](../../../integration/vime)) — confirm the user specifically wants the vllm-router path (e.g. they already operate vllm-router) before proceeding.
2. **Worker registration**: Are engine URLs **static** (known at router launch) or do engines **self-register at runtime** (`POST /workers`)?
3. **Request identity**: Does the framework tag each request with a per-trajectory/session header (e.g. a routing key or session id)? This determines whether session-affinity scorers (sticky session, consistent hash) are usable; without it, only engine-gauge and queue-based scorers apply.
4. **Environment**: Bare **VM / container** or **Kubernetes**? Which OS family? (The build commands below assume Debian/Ubuntu.)

*Answer 1 gates the whole skill. Answers 2–3 select the launch flags and policy configuration (Sections 3.3, 4). Answer 4 adjusts the build commands (Section 3.1).*

---

## 2. Architectural Blueprint (Code Boundaries)

### 2.1 Control Flow (Launch & Routing)

1. **Entry point**: `python -m integration.vllm_router` ([__main__.py](../../../integration/vllm_router/__main__.py)) parses the two integration flags (`--external-scheduler-config`, `--external-metrics-interval-ms`), exports them as environment variables (`ROUTER_CONFIG_PATH`, `RLS_METRICS_INTERVAL_MS`), sets `--external-policy-factory integration.vllm_router.factory:make_policy` on the router args, and hands off to the fork's `launch_router`. All other flags pass through to vllm-router unchanged.
2. **Factory (once, at startup)**: the fork imports and calls `make_policy(router_args)` ([factory.py](../../../integration/vllm_router/factory.py)). It builds the `Scheduler` from the yaml config, constructs `VllmRouterSchedulerAdapter`, seeds statically known workers from `--worker-urls`, starts the metrics poller, and returns `adapter.select`. It raises on missing/invalid config — launch fails closed; the router's fallback policy is reserved for per-request failures.
3. **Per request**: the router calls `select(workers, request_text, headers) -> int | None` ([adapter.py](../../../integration/vllm_router/adapter.py)) on a Rust thread holding the GIL. The adapter upserts the offered worker dicts into its endpoint registry (TTL-pruned, 60s default), runs the `Scheduler`, and returns the chosen worker's position in `workers`.
4. **Fallback**: `None` or any exception routes that request via the router's built-in `--external-fallback-policy` (default `round_robin`). The scheduler can never fail a request, only route it.

### 2.2 Data Flow (Metrics)

- **Engine gauges**: a background `MetricsPoller` thread scrapes each registered worker's `GET /metrics` (Prometheus text) off the request path and parses `vllm:num_requests_waiting`, `vllm:num_requests_running`, and `vllm:kv_cache_usage_perc` into each endpoint's `routing_stats`.
- **Inflight counts**: the router tracks per-worker inflight requests itself; the count arrives inside every `select` call as the `load` key of each worker dict and is written to the endpoint's `queue_len`. This value is live even when engine gauges have not moved yet.
- **Worker discovery**: workers register with the router, not with the scheduler. The adapter learns the worker set lazily from each `select` call, plus optional startup seeding from `--worker-urls`. There is no deregistration signal; the TTL prune substitutes for it.

---

## 3. Pre-Flight Checklist

### 3.1 Fork build prerequisites

1. Rust toolchain (rustup stable).
2. C compiler, `pkg-config`, and OpenSSL headers: `apt-get update && apt-get install -y build-essential pkg-config libssl-dev`. In containers, `apt-get update` must run first or apt reports "Unable to locate package".
3. The extension compiles against the Python interpreter that runs `pip install .`. Install the fork and the scheduler into the **same venv**, and treat the built wheel as interpreter-specific regardless of its `abi3` filename.
4. The build compiles silently for several minutes. This is normal.

### 3.2 Scheduler install

- `pip install -e . --no-deps`, then `pip install aiohttp prometheus-client pyyaml setproctitle`.
- Running from the repo root **without installing** does not work (src layout).
- pip prints unmet-dependency warnings for ray/fastapi/uvicorn — expected and harmless with `--no-deps`; those belong to other integrations.

### 3.3 Policy configuration

Base config: [integration/slime/examples/scheduler.yaml](../../../integration/slime/examples/scheduler.yaml). Selecting and weighting scorers is workload tuning — defer to the [Scheduler Customization Guide](../../../docs/scheduler_customization.md) for the available scorers, pickers, and flow-control plugins, and work through it with the user rather than prescribing a profile.

- The guide does not yet document the `saturation` filter on main ([plugins/filters/saturation.py](../../../src/py_inference_scheduler/plugins/filters/saturation.py)); read its source when configuring filters.
- One constraint is integration-specific, not workload tuning: session-affinity scorers (sticky session, consistent hash) require the per-request identity header from triage answer 3. Omit them if the framework provides none.

### 3.4 Worker contract

- `GET /health` → 200. The router blocks before serving until every `--worker-urls` entry answers (up to 600s).
- `GET /metrics` → Prometheus text containing the three gauges in Section 2.2. **Missing gauge names parse as silent zeros, not errors** — a wrong name (e.g. `vllm:gpu_cache_usage_perc`) is indistinguishable from an idle worker.
- The proxied inference endpoints: `/v1/completions`, `/v1/chat/completions`, `/generate`, `/inference/v1/generate`. Real vLLM servers provide all of the above out of the box.

---

## 4. Framework Wiring

Point the framework's rollout/generation HTTP client at the router's `--host:--port` instead of at a worker. Any HTTP dialect the workers speak passes through; the router extracts routing text from text prompts or token-id payloads automatically.

- **Static workers** (triage answer 1 = static): pass them as `--worker-urls` at router launch. The adapter seeds them so the first requests are scored on real stats.
- **Self-registering engines** (triage answer 1 = runtime): engines `POST /workers {"url": "..."}` at boot and may deregister with `DELETE /workers/{url}`. Start the router **before** the framework job. The router's worker registry is in-memory: if the router restarts, engines must re-register (usually by restarting the run).

---

## 5. Self-Healing Diagnostic Flow

Follow this progressive diagnostic tree to isolate and fix the exact failure point:

### Step 5.1: Does the fork build fail? (Install Phase)

- **Symptom**: `pip install .` of the fork fails; `cargo ... failed with code 101` and `openssl-sys` "Could not find directory of OpenSSL installation" buried in the pip output.
- **Root Cause**: missing system packages (Section 3.1).
- **Fix**: `apt-get update && apt-get install -y build-essential pkg-config libssl-dev`, then reinstall.

### Step 5.2: Does the router fail at launch? (Startup Phase)

- **Symptom**: `SystemExit: vllm-router (the Rust router fork) is not installed`.
  **Root Cause**: the fork wheel is missing from the active venv (or was installed into a different one).
  **Fix**: repeat README Step 1 inside the same environment; verify with `pip show vllm-router`.
- **Symptom**: `ValueError: ROUTER_CONFIG_PATH must point to a scheduler yaml config`.
  **Root Cause**: the factory was invoked without the launcher setting the config path (launch fails closed by design).
  **Fix**: launch via `python -m integration.vllm_router --external-scheduler-config <path>`.
- **Symptom**: panic `FailedToCreateHTTPListener("Address already in use")`.
  **Root Cause**: the router binds `:29000` for its own Prometheus exporter regardless of `--port`; a second or stale router already holds it.
  **Fix**: kill the stale `vllm::router` process (see Step 5.6) or pass a distinct `--prometheus-port`.

### Step 5.3: Does the router hang before serving? (Health-Gate Phase)

- **Symptom**: startup produces no listening message for a long time (up to 600s).
- **Root Cause**: at least one `--worker-urls` entry is not answering `GET /health`.
- **Fix**: `curl <worker>/health` for each entry; correct or remove unreachable workers.

### Step 5.4: Is routing happening only via fallback? (Selection Phase)

- **Symptom**: `Selection failed, deferring to router fallback` in the router log. Requests still succeed because the fallback policy is serving.
- **Diagnostic**: the exception traceback is logged with that message; read it. Treat a streak of these as an incident even though traffic flows — the scheduler is erroring on every request.

### Step 5.5: Are metrics missing or stuck at zero? (Scraping Phase)

- **Diagnostic 1 (scrape errors)**: run with `RLS_DECISION_LOG=1` and inspect each decision line's per-endpoint stats. A scrape failure appears as `error: <message>` in that endpoint's `routing_stats`.
- **Diagnostic 2 (gauge-name mismatch)**: `error: None` with zeros under **sustained** load means the worker's `/metrics` does not expose the exact gauge names in Section 2.2. `curl <worker>/metrics` and compare.
- **Diagnostic 3 (burst timing)**: zeros at the instant of burst arrival are normal — gauges lag dispatch; this is not a scrape failure. Inflight `queue_len` from the router's load counters is the signal that stays live through a burst (Section 2.2).
- **Diagnostic 4 (staleness)**: poller staleness warnings mean **no** endpoint yielded usable stats in the interval; check per-endpoint `error` values to find out why.

### Step 5.6: Can you not find or kill the router process? (Operations)

- **Symptom**: `pkill python` does not terminate the router; or two routers appear to run.
- **Root Cause**: the router renames its process to `vllm::router` (setproctitle).
- **Fix**: match `vllm::router` with `pkill`/`pgrep`. In minimal containers without pkill, scan `/proc/[0-9]*/comm` for `vllm::router`.

---

## 6. Verification (in order)

1. **Startup**: `Loaded scheduler config: {...}`, then `Assigning policy external`, then `Seeded worker <url>` for each static worker (or `Tracking worker <url>` for runtime registrations).
2. **Traffic**: send one completion through the router; a 200 response proves the proxy path.
3. **Decisions**: with `RLS_DECISION_LOG=1`, each request logs `request <id> candidates: {url: {queue_len, stats}}`. Confirm `error: None` for every endpoint and, under sustained load, nonzero `num_running_reqs`/`kv`. Note the toggle adds routing latency on every request while enabled — leave it off for benchmarks.
