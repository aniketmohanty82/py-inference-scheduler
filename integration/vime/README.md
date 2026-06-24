# vime Integration with py-inference-scheduler

## Compatibility Notice

**This integration is designed specifically for [vime v0.3.0](https://github.com/vllm-project/vime/tree/v0.3.0)** (commit `08f4a0c`, built on vLLM `v0.22.0`).

vime is the vLLM team's fork of slime: it keeps slime's decoupled train↔rollout design but swaps the
rollout backend to vLLM + `vllm-router`. This integration implements vime's `vllm-router` HTTP dialect
(`POST /workers` with a JSON body, `POST /inference/v1/generate`). It may require updates for other
vime / vllm-router versions.

### Scope (v1)

Supports the **default text-only, non-streaming rollout** (`vime.rollout.vllm_rollout`). Not yet
supported (the rollout only uses these for specific modes — add if you need them):
- **Streaming** (`vllm_streaming_rollout`, `"stream": true` SSE).
- **Multimodal** (`POST /v1/chat/completions/render` for image inputs).
- Engine **`pause`/`resume`** abort — those go engine-direct, not through this router; we only expose
  the `/workers` (and `/list_workers`) enumeration the abort path needs.

## Architecture

vime manages its own vLLM rollout engines and, by default, launches its own `vllm-router` to load
balance across them. When you set `--vllm-router-ip/--vllm-router-port`, vime skips that router
(`vime/ray/rollout.py` `_start_router` returns early) and instead each engine self-registers with ours
(`POST /workers`) and the rollout posts generations to it (`POST /inference/v1/generate`). On each
request the router scrapes the engines' vLLM Prometheus `/metrics` and delegates the routing decision
to `py-inference-scheduler`. vime keeps full ownership of the rollout lifecycle; we only decide which
engine serves each request.

Key components:
- [server.py](./server.py): the router — worker registry + the scheduled `/inference/v1/generate` proxy.
- [`__main__.py`](./__main__.py): the `python -m integration.vime` launcher.
- [datalayer/metrics/vime/](../../datalayer/metrics/vime): per-request vLLM Prometheus `/metrics` scrape.

---

## Integration Configuration

The routing policy reuses slime's engine-agnostic
[`scheduler.yaml`](../slime/examples/scheduler.yaml) (default `backpressure`: prefix-cache affinity +
queue/KV-pressure load balancing). The scorers read engine-neutral routing stats
(`num_waiting_reqs`/`num_running_reqs`/`kv`), so the same config drives vLLM engines. To customize,
copy that file, edit it, and pass `--config /path/to/your.yaml`. See the
[Scheduler Customization Guide](../../docs/scheduler_customization.md) for the available scorers,
pickers, and flow-control plugins.

## Running a Training Job

First, clone this repo onto the VM (the head node for multi-node) and install the router's
dependencies on top of the vime image:

```bash
git clone https://github.com/llm-d-incubation/py-inference-scheduler.git
cd py-inference-scheduler
pip install fastapi uvicorn aiohttp prometheus-client pyyaml setproctitle
```

**Start the router** — CPU-only, run from the repo root (so `integration` / `datalayer` /
`scheduling` import). It must be up **before** the vime job (engines register at boot), and it
renames its process to `router` so vime's example scripts' `pkill -9 python` cleanup won't kill it.
Run it on the single VM for single-node, or on **node 0 (the head)** for multi-node; `--host 0.0.0.0`
makes it reachable both locally and from worker nodes.

```bash
python -m integration.vime --host 0.0.0.0 --port 8000
```

It uses slime's bundled `scheduler.yaml` by default; pass `--config /path/to/your.yaml` to override
with a custom policy.

Then point vime at it — the **only** change to vime's launch is two flags:

```bash
    --vllm-router-ip   127.0.0.1     # single-node; use the head node's IP (${MASTER_ADDR}) for multi-node
    --vllm-router-port 8000
```

For single-node, the router and engines share the VM, so `127.0.0.1` works. For multi-node, set
`--vllm-router-ip` to the **head node's IP** (the same value you gave `ray start --head`) so engines
on worker nodes can reach it.

## Verifying Results

The router prints its routing decisions to **stdout** — the terminal where you started it. Watch for
`Selected endpoint …` lines as the rollout flows through the scheduler.

The scheduler's impact shows in vime's per-step rollout timing — the router only affects the
**rollout** (generation) phase, so rollout wall-clock and per-GPU generation throughput are the most
direct signals (lower rollout time / higher throughput is better routing).
