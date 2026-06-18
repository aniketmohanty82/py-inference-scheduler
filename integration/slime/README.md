# slime Integration with py-inference-scheduler

## Compatibility Notice

**This integration is designed specifically for [slime v0.3.0](https://github.com/THUDM/slime/releases/tag/v0.3.0).**

slime v0.3.0 ships sgl-router 0.3.2, whose `/workers` HTTP API this integration implements; the
older sgl-router endpoints (`/add_worker`, `/list_workers`, `/remove_worker`) are intentionally not
implemented. It may require updates for other slime / sgl-router versions.

## Architecture

slime manages its own SGLang rollout engines and, by default, launches its own sgl-router to load
balance across them. When you set `--sglang-router-ip/--sglang-router-port`, slime skips that router
and instead each engine self-registers with ours (`POST /workers`) and the rollout posts generations
to it (`POST /generate`). On each request the router scrapes the engines' Prometheus `/metrics` and
delegates the routing decision to the `py-inference-scheduler`. slime keeps
full ownership of the rollout lifecycle; we only decide which engine serves each request.

Key components:
- [server.py](./server.py): the router — worker registry + the scheduled `/generate` proxy.
- [__main__.py](./__main__.py): the `python -m integration.slime` launcher.
- [datalayer/metrics/slime/](../../datalayer/metrics/slime): per-request Prometheus `/metrics` scrape.

---

## Prerequisites (Step 1)

This integration follows and has been tested against slime's
[Quick Start](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/get_started/quick_start.md) guide. For all non scheduler integration steps, please follow the guide as directed below:

| Task | slime Quick Start |
|---|---|
| Environment / image / install slime | [§ Basic Environment Setup — L6–L50](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/get_started/quick_start.md#L6-L50) |
| Download model + dataset | [§ Model and Dataset Download — L51–L67](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/get_started/quick_start.md#L51-L67) |
| Convert HF → Megatron checkpoint | [§ Model Weight Conversion — L68–L107](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/get_started/quick_start.md#L68-L107) |

> [!NOTE]
> Rather than using the ```slimerl/slime:latest```image, we would prefer if you used ```slimerl/slime:v0.3.0``` image.


## Integration Configuration (Step 2)

The routing policy lives in [`examples/scheduler.yaml`](./examples/scheduler.yaml) (default
`backpressure`: prefix-cache affinity + queue/KV-pressure load balancing). This project isn't
packaged yet, so to customize it just **edit that file directly** inside your VM and restart the router. See the
[Scheduler Customization Guide](../../docs/scheduler_customization.md) for the available scorers,
pickers, and flow-control plugins.

## Running a Training Job (Step 3)

**Start the router** — CPU-only, from this repo's root (so that the required `integration` / `datalayer` /
`scheduling` libraries import). It must be up **before** the slime job (engines register at boot), and it
renames its process to `router` so slime's example scripts `pkill -9 python` cleanup won't kill it.
The command is the same for single- and multi-node — run it on the single VM for single-node, or on
**node 0 (the head)** for multi-node; `--host 0.0.0.0` makes it reachable both locally and from
worker nodes.

First install the router's dependencies into the pod (on top of the slime image):

```bash
pip install fastapi uvicorn aiohttp prometheus-client pyyaml setproctitle
```

Then start it:

```bash
python -m integration.slime --host 0.0.0.0 --port 8000
```

It uses the bundled `examples/scheduler.yaml` by default; pass `--config /path/to/your.yaml` to
override with a custom policy.

Then point slime at it — the **only** change to slime's launch is two flags:

### Single node — [§ Training Script — L108–L115](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/get_started/quick_start.md#L108-L115)

Run `bash scripts/run-<model>.sh` as documented, adding two flags to its `SGLANG_ARGS`. The router
and the engines run on the same VM, so the engines reach the router at `127.0.0.1`:

```bash
    --sglang-router-ip   127.0.0.1
    --sglang-router-port 8000
```

### Multi node — [§ Multi-Node Training — L551–L593](https://github.com/THUDM/slime/blob/v0.3.0/docs/en/get_started/quick_start.md#L551-L593)

Follow the Ray cluster and `ray job submit` exactly as documented. Start the router on **node 0** (as
above), and add the two flags to the `python3 train.py` args. Set `--sglang-router-ip` to the **head
node's IP** (the same value you gave `ray start --head` as `${MASTER_ADDR}`), so engines on the
worker nodes can reach it:

```bash
   -- python3 train.py \
   --... \                              # your normal Megatron/SGLang/slime args
   --sglang-router-ip   ${MASTER_ADDR} \
   --sglang-router-port 8000
```

## Verifying Results (Step 4)

The router prints its routing decisions to **stdout** — the terminal where you started it in Step 3.
Watch that terminal for `Selected endpoint …` lines as the rollout flows through the scheduler. This shows the actual decisions made by the scheduler and metrics it uses to make them.

The scheduler's impact shows in slime's per-step `perf` line in the job log. This is what a step looks like:
```
perf 448: {'perf/step_time': 89.0, 'perf/train_wait_time': 74.8, 'perf/wait_time_ratio': 0.84, ...}
```
Use `perf/step_time` or `perf/train_wait_time` to see the scheduler's effect. `perf/step_time` is
the entire step time; `perf/train_wait_time` is the entire non-training time within a step (rollout,
weight sync, log-probs, etc.).

For the actual sampling throughput, read the SGLang engine logs directly — each engine prints
`gen throughput (token/s)`.
