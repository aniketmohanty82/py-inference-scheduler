# slime Integration with py-inference-scheduler

## Compatibility Notice

**This integration targets [slime v0.3.0](https://github.com/THUDM/slime/releases/tag/v0.3.0)**
(which ships `sgl-router 0.3.2`). It implements the modern sgl-router `/workers` HTTP
surface that slime v0.3.0 actually emits; older sgl-router endpoints (`/add_worker`,
`/list_workers`, `/remove_worker`) are intentionally not implemented.

## Architecture

slime normally launches its own `sgl-router` inside the training job and routes rollout
traffic through it. When you set `--sglang-router-ip/--sglang-router-port`, slime **skips
launching its own router** (`slime/ray/rollout.py`) and instead:

- each SGLang engine self-registers with our router (`POST /workers`), and
- the rollout fires generations at our router (`POST /generate`).

We run as a **standalone HTTP process** that delegates the routing decision to the
`py-inference-scheduler` engine (`scheduling/`). Unlike the verl integration, **no slime
code is touched and nothing is injected into the slime image** — it's purely two CLI flags.

slime owns the rollout lifecycle (batching, partial rollout, aborts — aborts go *directly*
to the engines, not through us). The router only owns "which worker serves this request".

Key components:
- [server.py](./server.py): the FastAPI app — worker registry + the scheduled `/generate` proxy.
- [`__main__.py`](./__main__.py): `python -m integration.slime` launcher.
- [`datalayer/metrics/slime/`](../../datalayer/metrics/slime): scrapes each worker's
  Prometheus `/metrics` (parsed via `prometheus_client`) on the scheduling path.

### Router HTTP surface (slime v0.3.0)
| Endpoint | Caller | Purpose |
|---|---|---|
| `POST /workers` | engine → router | register `{ "url", "worker_type" }`; we assign an `id` |
| `GET /workers` | engine / slime → router | list `{ "workers": [ { "url", "id" } ] }` |
| `DELETE /workers/{id}` | engine → router | deregister by `id` |
| `POST /generate` | slime rollout → router | scheduled, proxied to the chosen worker |

The router also scrapes `GET {worker_url}/metrics` (router → engine) on the scheduling path.

---

## Running a job

The only differences from a normal slime run are (1) starting this router first and
(2) adding two flags. Everything else (Ray cluster, `train.py`, args) is unchanged.

### 1. Start the router (before submitting the slime job)
On a host reachable from the Ray cluster (the head node is fine — CPU only), from the
repository root (so `integration`/`datalayer`/`scheduling` are importable):
```bash
python -m integration.slime --host 0.0.0.0 --port 8000 \
    --config integration/slime/examples/scheduler.yaml
```
Host, port, and config are all flags — change them inline as needed.

### 2. Point slime at it
Add to your slime `train.py` invocation (e.g. into `SGLANG_ARGS`):
```bash
    --sglang-router-ip   $ROUTER_HOST \
    --sglang-router-port 8000
```
- **Single node / `--colocate`:** `--sglang-router-ip 127.0.0.1`.
- **Multinode:** bind the router to `0.0.0.0` and pass the head node's IP.

The router must be up before the job starts, because the engines register at boot.

## Running on KubeRay (GKE)

[examples/slime-inference-scheduler.yaml](./examples/slime-inference-scheduler.yaml) brings up
a stock slime `RayCluster` (slime image, GPU workers) plus the router as a separate CPU-only
`Deployment` + `ClusterIP Service`. The router pod clones this repo's `slime-integration`
branch via an initContainer and `pip install`s its deps at startup (for releases, swap in a
baked image and drop the initContainer).

```bash
# 1. publish the routing profile as a ConfigMap (separate from any verl one)
kubectl create configmap slime-scheduler-config \
    --from-file=scheduler.yaml=integration/slime/examples/scheduler.yaml

# 2. bring up the cluster + router
kubectl apply -f integration/slime/examples/slime-inference-scheduler.yaml

# 3. submit the slime job (engines reach the router by Service DNS)
kubectl port-forward svc/slime-inference-scheduler-head-svc 8265:8265 &
export RAY_ADDRESS=http://127.0.0.1:8265
ray job submit --address "$RAY_ADDRESS" -- \
    python3 train.py ... \
    --sglang-router-ip slime-router --sglang-router-port 8000
```

The router needs no GPU and scrapes engine `/metrics` + proxies `/generate` over the in-cluster
pod network; the `RayCluster` carries **no** scheduler ConfigMap or metrics volume (unlike the
verl manifest), since all of that is centralized in the router.

## Configuration

The routing policy lives in [examples/scheduler.yaml](./examples/scheduler.yaml) — the same
plugin profile format as the verl integration. The default `backpressure` profile combines
prefix-cache affinity with queue/KV-pressure-aware load balancing. See the
[Scheduler Customization Guide](../../docs/scheduler_customization.md) for the available
scorers, pickers, and flow-control plugins.
