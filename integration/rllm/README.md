# rllm (DeepSWE-style) RL training × py-inference-scheduler — 3-arm convergence A/B

Full RL loop (rollout → reward → weight update) with our scheduler and Mooncake
overscheduling in the rollout path, to prove the layer does not impact
convergence. Plan of record: `.claude/plans/squishy-hatching-iverson.md`
(approved 2026-08-12). Pins + upstream facts: `patches/PINS.md`.

## Architecture

```
rllm AgentTrainer (verl backend, Qwen3-32B + LoRA r32, GRPO)
  └─ Model Gateway (subprocess, sqlite trace store, injects return_token_ids)
       └─ routing_policy = integration.rllm.policy.SchedulerRoutingPolicy   ← arms B/C
            └─ trainer-launched `vllm serve` replicas (TP=2, 8 replicas / 2 nodes)
                 └─ arm C only: DecodeKVSavingConnector + OverschedulingScheduler
                              + MOONCAKE_OVERSCHEDULING_OFFLOAD_ALL=1
mini-swe-agent runs INSIDE each R2E-Gym task container and calls the gateway
(OPENAI_API_BASE); reward = task tests executed in the same container.
Weight sync (verl CheckpointEngineManager) is trainer→replica, NOT via gateway.
```

Arms: A = stock (StickyLeastLoadedPolicy) · B = A + our policy · C = B + Mooncake
KV + evict-on-turn-end. Only `routing_policy` + engine kwargs + env flags differ.

## Status (2026-08-12)

- **Gate 0 ✅** branch consolidated (poller + vllm_router adapter cherry-picked,
  k8s assets committed), 83 unit tests green.
- **Gate 1 ✅** vendored gateway patch validated end-to-end locally (env-var
  policy load, URL-path session id delivered, selection honored); full stack
  resolves only with `image/overrides.txt` (see PINS.md).
- **Gate 4 (local half) ✅** `SchedulerRoutingPolicy` ran inside the patched
  gateway against stub workers: poller scraping, waiting_queue/kv_cache/
  sticky_session scoring live, session affinity stable. Offload-all flag +
  tests merged.
- **Gate 2/3 (GPU) — next**: `smoke/train_smoke.sh` on a GPU node using the
  training image, then the engine-kwargs passthrough spike.
- Image `rllm-verl-mooncake:dev` building (flash-attn compiles from source,
  sm90-only; see `image/Dockerfile`).

## Plan deviations (vs approved plan)

1. **Pinned fork → vendored patch.** Pushing to a public GitHub fork needs
   explicit approval; identical effect via `patches/*.diff` applied onto the
   pinned SHA in the image build. Local branch `gateway-routing-policy` in the
   working clone mirrors it.
2. **R2E-Gym k8s env backend does not exist in rllm's agent_flow path**
   (sandbox backends: docker/local/modal/daytona only; the k8s runtime is in
   the R2E-Gym repo, which rllm does not use). Cluster options: DOCKER_HOST →
   remote docker daemon, or write a `kubernetes` Sandbox backend for rllm
   (carried as a second vendored patch). Smoke uses local docker. DECISION
   NEEDED before Phase 5 cluster bring-up.
3. **Fully-async caveat is benign**: only the sandbox warm-pool prefetch is
   disabled under `rllm.async_training.enable=true` (and snapshots are no-ops
   on docker anyway); the separated 1-train + 2-rollout topology stands.

## Files

- `policy.py` — gateway RoutingPolicy wrapping Scheduler + MetricsPoller; env
  config: `ROUTER_CONFIG_PATH` (required), `RLS_METRICS_INTERVAL_MS`,
  `RLS_SESSION_HEADER`. Duck-types WorkerInfo; never raises (least-loaded
  fallback).
- `configs/scheduler.yaml` — arm B/C profile: waiting_queue 1.0 + kv_cache 0.5
  + sticky_session 4.0 (`x-rls-session-id`); no prefix_cache (policy hook has
  no request body), no saturation filter (not on this branch lineage).
- `smoke/` — Gate-2 assets: `prepare_r2egym_subset.py` (8 frozen orange3
  tasks, `r2egym_smoke_manifest.json`), `train_smoke.py|.sh` (Qwen3-4B, LoRA,
  colocated sync, sqlite traces).
- `image/` — `Dockerfile` (+`build.sh <tag>`): ray-llm-mooncake:2.56.0 base +
  pinned/patched rllm[verl] (overrides mandatory) + docker-py + cupy + this
  package. Import-gate + pip check baked in.
- `k8s/base/` — mooncake master/config/RDMA claim templates seeded from the
  tau3 west deploy (secrets are placeholders).
- `patches/` — the rllm gateway diff + PINS.md.

## Gate-2 run (GPU node)

```bash
docker run --gpus all --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
  us-south1-docker.pkg.dev/aniket-gke-dev/llm-images/rllm-verl-mooncake:dev bash
python /opt/py-inference-scheduler/integration/rllm/smoke/prepare_r2egym_subset.py
# pre-pull the 8 images from the manifest, then:
bash /opt/py-inference-scheduler/integration/rllm/smoke/train_smoke.sh
# observables: ≥2 steps, non-constant reward; sqlite traces with token IDs
# grouped by rollout_id; CheckpointEngineManager sync lines each step
```

Arm B/C later add (see plan Phase 4 table):
`+rllm.gateway.routing_policy=integration.rllm.policy.SchedulerRoutingPolicy`
with `ROUTER_CONFIG_PATH=$REPO/integration/rllm/configs/scheduler.yaml`, and
arm C the mooncake engine kwargs + env flags (`kv.py` builds them from env).
