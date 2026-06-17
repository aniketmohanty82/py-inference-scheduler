# slime quickstart, with the py-inference-scheduler router

This is slime's own [Quick Start](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md)
run **verbatim**, with our scheduler added as the only delta: **one extra process + two
flags**. It is the simplest, most doc-faithful way to see the router routing a real GRPO run.

We run slime as a **single self-contained node**. On a GPU box that's literally slime's
`docker run`; on GKE (no `docker run --gpus all`) the equivalent is one Pod
([`examples/slime-node.yaml`](./examples/slime-node.yaml)) — same image, same interactive
shell, same GPUs. slime starts its own Ray *inside* the pod, so the stock run script works
unchanged.

> The only differences from slime's quickstart are **Step 4 (start the router)** and the
> **two `--sglang-router-*` flags** in Step 5. Steps 1–3 are the quickstart, unchanged.

---

## Step 0 — Environment (≡ quickstart "Pull and Start Docker Container")

| slime quickstart | here |
|---|---|
| `docker run --gpus all --ipc=host --shm-size=16g -it slimerl/slime /bin/bash` | the Pod below + `kubectl exec -it` |

```bash
kubectl apply -f integration/slime/examples/slime-node.yaml
kubectl exec -it slime-node -- bash
```
Everything below runs **inside that one shell** (exactly as the quickstart runs inside the
`docker run` shell).

## Step 1 — Download model + dataset (≡ quickstart "Model and Dataset Download")

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024      --local-dir /root/aime-2024
```

## Step 2 — Convert HF → Megatron (≡ quickstart "Model Weight Conversion")

```bash
cd /root/slime && source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/Qwen3-4B --save /root/Qwen3-4B_torch_dist
```

## Step 3 — Start the router (the ONE added step)

This is the entire integration footprint — a single CPU process that slime will route
through instead of launching its own sgl-router.

```bash
pip install fastapi uvicorn aiohttp prometheus-client pyyaml setproctitle
git clone -b slime-integration https://github.com/aniketmohanty82/py-inference-scheduler.git /root/router
cd /root/router   # run from the repo root so `python -m integration.slime` resolves the package
python -m integration.slime --host 127.0.0.1 --port 8000 \
    --config integration/slime/examples/scheduler.yaml > /root/router.log 2>&1 &
sleep 3 && cat /root/router.log && curl -s http://127.0.0.1:8000/workers   # -> {"workers":[]}
```

> Run the router and the job in **this same kept-open shell** — as a background job (`&`) it
> stays alive as long as the shell is open. The router renames itself ("router") so slime's
> stock `pkill -9 python` cleanup won't kill it (that, not the session, was the earlier
> failure). If you'd rather launch it from a *separate* `kubectl exec` or have it survive a
> disconnect, prefix with `setsid … </dev/null` to detach it from the exec session.

## Step 4 — Train (≡ quickstart "Training Script", + 2 flags)

The stock `scripts/run-qwen3-4B.sh`, with the **only** change being two flags in
`SGLANG_ARGS` that point slime at our router:

```bash
cd /root/slime
sed -i 's#--rollout-num-gpus-per-engine 2#--rollout-num-gpus-per-engine 2\n   --sglang-router-ip 127.0.0.1\n   --sglang-router-port 8000#' \
    scripts/run-qwen3-4B.sh
NUM_GPUS=8 bash scripts/run-qwen3-4B.sh
```

## Step 5 — Verify routing

In a second terminal:
```bash
kubectl exec slime-node -- grep -E "Registered worker|Selected endpoint" /root/router.log | tail
```
Success looks like **4× `Registered worker …`** (8 GPUs ÷ TP-2 = 4 engines), then
`Selected endpoint …` lines as slime's rollout flows through the scheduler, while slime
prints `step:N`. That confirms: stock slime quickstart + 1 process + 2 flags → routed.

## Teardown

```bash
kubectl delete pod slime-node
```

---

### Step-by-step equivalence to slime's quickstart

| slime quickstart section | here | same / delta |
|---|---|---|
| Pull and Start Docker Container | Step 0 (Pod + `exec`) | same (GKE translation) |
| Install slime | (pre-installed in image) | same |
| Model and Dataset Download | Step 1 | identical commands |
| Model Weight Conversion | Step 2 | identical commands |
| — | Step 3 (start router) | **added** (1 process) |
| Training Script | Step 4 | same **+ 2 flags** |

### Config

The routing policy is [`examples/scheduler.yaml`](./examples/scheduler.yaml) — a `backpressure`
profile combining prefix-cache affinity with queue/KV-pressure load balancing. See the
[Scheduler Customization Guide](../../docs/scheduler_customization.md) to change it.
