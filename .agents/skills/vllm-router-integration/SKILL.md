---
name: vllm-router-integration
description: Wire py-inference-scheduler into vllm-router as the routing policy via our external-policy fork. Use when a user runs vllm-router (any platform, any RL framework or serving stack) and wants this repo's scorers deciding which worker serves each request. Covers build, launch, framework wiring, verification, and the failure modes that are invisible from the README.
---

# vllm-router integration (external policy)

The canonical steps live in [integration/vllm_router/README.md](../../../integration/vllm_router/README.md) —
follow it for the happy path. This skill adds the operational knowledge around it: what breaks, why,
and how the pieces behave under real traffic. Assume nothing about the user's platform: no Kubernetes,
no specific cloud, any RL framework or none.

## Mental model (30 seconds)

Our [vllm-router fork](https://github.com/aniketmohanty82/router/tree/external-policy) (base v0.1.15)
adds `--external-policy-factory MODULE:FUNCTION`. At launch it imports the factory and calls it once;
the returned callable is invoked per request as `select(workers, request_text, headers) -> int | None`
on a Rust thread holding the GIL. `integration.vllm_router.factory:make_policy` is that factory: it
builds a `Scheduler` from yaml and returns the adapter's `select`. `None` or any exception falls back
to `--external-fallback-policy` (default round_robin) — the scheduler can never fail a request, only
route it. Engine gauges arrive via a background poller thread scraping worker `/metrics`; router-side
inflight counts arrive live inside each `select` call.

## Build & install — failure modes first

1. **System packages**: the fork's Rust build needs a C compiler, `pkg-config`, and OpenSSL headers.
   Missing → cargo exit 101 with `openssl-sys` "Could not find directory of OpenSSL installation"
   buried in pip output. Debian/Ubuntu: `apt-get update && apt-get install -y build-essential
   pkg-config libssl-dev`. (Containers usually need the `update` — stale apt lists say
   "Unable to locate package".)
2. **Rust toolchain**: `rustup` stable is fine. The build compiles silently for several minutes.
3. **Python version binds at install**: the extension compiles against the interpreter that runs
   `pip install .` — despite the wheel's `cp38-abi3` filename, treat it as version-specific. Install
   fork and scheduler into the SAME venv.
4. **Scheduler install**: `pip install -e . --no-deps` then
   `pip install aiohttp prometheus-client pyyaml setproctitle`. Plain "run from repo root" does NOT
   work (src layout). pip prints scary unmet-dependency warnings for ray/fastapi/uvicorn — expected
   and harmless with `--no-deps`; those are for other integrations.

## Worker contract (what the router expects of vLLM workers)

- `GET /health` → 200. The router **blocks before serving** until every `--worker-urls` entry answers
  (default up to 600s). A hung startup usually means an unreachable worker, not a router bug.
- `GET /metrics` → Prometheus text. Scorers read `vllm:num_requests_waiting`,
  `vllm:num_requests_running`, `vllm:kv_cache_usage_perc`. **Missing gauges parse as silent zeros**
  (no error) — wrong metric names (e.g. `vllm:gpu_cache_usage_perc`) look exactly like idle workers.
- The inference endpoints being proxied (`/v1/completions`, `/v1/chat/completions`, `/generate`,
  `/inference/v1/generate`). Real vLLM serves all of this out of the box.
- Workers can also register at runtime: `POST /workers {"url": "..."}` and deregister
  `DELETE /workers/{url}`. Frameworks whose engines self-register need only the router's address.

## Wiring a custom RL framework

Point the framework's rollout/generation HTTP client at the router's `--host:--port` instead of a
worker. Any HTTP dialect the workers speak passes through — the router extracts routing text from
text prompts or token-id payloads automatically. Two patterns:
- **Static workers**: pass them as `--worker-urls` at router launch.
- **Self-registering engines** (slime/vime-style): engines `POST /workers` at boot; start the router
  BEFORE the framework job. Note the router keeps its registry in memory — restarting the router
  requires engines to re-register (usually: restart the run).

## Policy config fit — the expensive lesson

Reuse `integration/slime/examples/scheduler.yaml` as the base. Two rules learned on GPUs:
- **Always keep `least_queue` in the profile.** At burst arrival (RL rollouts dispatch hundreds of
  requests in ~1s), engine gauges truthfully read zero — all engine-side scorers tie, and without the
  live router-load tie-breaker the max_score picker sends the ENTIRE burst to one engine
  (observed: 131 requests on one worker at kv 0.995 while three sat idle).
- **Drop or down-weight `prefix_cache` for GRPO-style traffic** where all prompts share a chat
  template and samples are duplicated per prompt: universal shared prefixes make affinity herd
  instead of partition (observed: +67% rollout time vs round_robin). It earns its keep on
  multi-session serving traffic with distinct prefixes.

## Operations

- **Process name**: the router retitles itself to `vllm::router` (setproctitle) — `pkill python`
  misses it; match `vllm::router`. In minimal containers without pkill, scan `/proc/[0-9]*/comm`.
- **Prometheus exporter port**: the router binds `:29000` for its own metrics regardless of `--port`.
  A second router on the same host panics with `FailedToCreateHTTPListener("Address already in use")`
  — pass a distinct `--prometheus-port`.
- **Observability**: `RLS_DECISION_LOG=1` logs each request's per-endpoint stats at decision time
  (adds routing latency while on — never during benchmarks). `--log-level debug` adds per-scorer raw
  scores and the selected endpoint. Poller staleness (`routing on stale metrics` warnings downstream)
  means NO endpoint yielded usable stats — per-endpoint failures live in each endpoint's
  `routing_stats["error"]`, visible in the decision log.
- **Fallback streaks**: `Selection failed, deferring to router fallback` in the log means the
  scheduler is erroring and round_robin is silently serving — treat a streak as an incident even
  though requests succeed.

## Verify (in order)

1. Startup: `Loaded scheduler config: {...Profiles: [...]}` then `Assigning policy external` as
   workers register; `Seeded worker <url>` for static workers, `Tracking worker <url>` for runtime
   registrations.
2. Traffic: send a completion through the router; 200 proves the proxy path.
3. Decisions: with `RLS_DECISION_LOG=1`, each request logs
   `request <id> candidates: {url: {queue_len, stats}}` — confirm `error: None` and, under sustained
   load, NONZERO `num_running_reqs`/`kv` (zeros under load = gauge-name or scrape problem; zeros at
   burst instant = normal, see policy-fit section).

## Known-good validation reference

Mechanical overhead measured below run-to-run noise vs stock round_robin on 8xH100 (synthetic 4-arm
inference-perf A/B and a vime GRPO A/B with identical decisions); README cold-tested end-to-end by a
context-free agent and re-validated against real vLLM workers.
