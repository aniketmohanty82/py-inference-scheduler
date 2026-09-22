# miles Integration with py-inference-scheduler

[miles](https://github.com/radixark/miles) is a fork of slime, so this integration **reuses the
slime router (`python -m integration.slime`) unchanged**. To learn how it works, see the [Architecture section of the slime README](./README.md#architecture).

> [!NOTE]
> Leave `--use-miles-router` **unset** so miles' engines self-register with our router instead of
> miles' own `MilesRouter`.

## Compatibility Notice

miles publishes no releases, so this integration is pinned by validation rather than by tag:

| | Validated (2026-06-22) | miles `main` (2026-09-22) |
|---|---|---|
| miles commit | `713d99d` | moving |
| SGLang engine | 0.5.14 | v0.5.20 ([radixark/miles#3321](https://github.com/radixark/miles/pull/3321), merged 2026-09-21) |
| sglang-router | 0.3.2 | 0.3.2 |

The router-facing contract is unchanged at SGLang v0.5.20 (see the
[slime README](./README.md#sglang-v0520)), and miles ports the v0.5.20 `ServerArgs` → `msgspec`
change itself ([#3321](https://github.com/radixark/miles/pull/3321),
[#3581](https://github.com/radixark/miles/pull/3581)), so `python -m integration.slime` needs no
change. Two things to watch when running against current miles:

- **SGLang CLI flag renames.** The v0.5.20 bump renamed flags SGLang removed
  (`--cuda-graph-max-bs` → `--cuda-graph-max-bs-decode`, `--cuda-graph-bs` → `--cuda-graph-bs-decode`,
  `--nsa-*-backend` → `--dsa-*-backend`, `--mamba-scheduler-strategy` →
  `--mamba-radix-cache-strategy`). `SGLANG_ARGS` copied from older run scripts are rejected at engine
  start; the two router flags below are unaffected.
- **Not yet re-validated on GPUs.** The v0.5.20 check is source-level plus unit tests, not a training
  run. Before a long run on a v0.5.20 miles image, confirm the engines appear under `GET /workers`
  and `Selected endpoint` lines flow (Step 4).

---

## Prerequisites (Step 1)

This integration follows and has been tested against miles'
[Quick Start](https://radixark.mintlify.app/getting-started/quick-start) guide. For all non scheduler integration steps, please follow the guide as directed below:

| Task | miles Quick Start |
|---|---|
| Environment / image / install miles | [§ Start the container](https://radixark.mintlify.app/getting-started/quick-start#1-start-the-container) |
| Download model + dataset | [§ Download model and data](https://radixark.mintlify.app/getting-started/quick-start#2-download-model-and-data) |
| Convert HF → Megatron checkpoint | [§ Convert to Megatron format](https://radixark.mintlify.app/getting-started/quick-start#3-convert-to-megatron-format) |

> [!NOTE]
> miles' quick-start downloads the prompt dataset (`DAPO-Math-17K`) as **parquet**, but `scripts/run-qwen3-4B.sh` expects a **`prompt`/`label` jsonl** - and the quick-start does not cover this conversion. You'll need to prepare the dataset into that format yourself before launching.


## Integration Configuration (Step 2)

The routing policy lives in [`examples/scheduler.yaml`](./examples/scheduler.yaml) (default
`backpressure`: prefix-cache affinity + queue/KV-pressure load balancing). This project isn't
packaged yet, so to customize it just **edit that file directly** inside your VM and restart the router. See the
[Scheduler Customization Guide](../../docs/scheduler_customization.md) for the available scorers,
pickers, and flow-control plugins.

## Running a Training Job (Step 3)

First, clone this repo onto the VM and install the router's dependencies on top of the miles image:

```bash
git clone https://github.com/llm-d-incubation/py-inference-scheduler.git
cd py-inference-scheduler
pip install fastapi uvicorn aiohttp prometheus-client pyyaml setproctitle
```

**Start the router** — CPU-only, run from the repo root (so `integration` / `datalayer` /
`scheduling` import). It must be up **before** the miles job (engines register at boot), and it
renames its process to `router` so miles' run scripts' `pkill -9 {python,ray,sglang}` cleanup won't
kill it. `--host 0.0.0.0` makes it reachable locally.

```bash
python -m integration.slime --host 0.0.0.0 --port 8000
```

It uses the bundled `examples/scheduler.yaml` by default; pass `--config /path/to/your.yaml` to
override with a custom policy. You can also just change `examples/scheduler.yaml` as mentioned above.

Then point miles at it — the **only** change to miles' launch is two flags (and leave
`--use-miles-router` unset). Run `bash scripts/run-qwen3-4B.sh` as documented, adding two flags to
its `SGLANG_ARGS`. The router and the engines run on the same VM, so the engines reach the router at
`127.0.0.1`:

```bash
    --sglang-router-ip   127.0.0.1
    --sglang-router-port 8000
```
This differs for multi-node, and we did not find a miles doc detailing that process — we suggest following the [Multi node section of the slime README](./README.md#multi-node---multi-node-training--l551l593), which the router has been tested against.

## Verifying Results (Step 4)

See the [Verifying Results section of the slime README](./README.md#verifying-results-step-4).
