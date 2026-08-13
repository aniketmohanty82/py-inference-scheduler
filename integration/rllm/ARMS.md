# The three arms: vanilla vs +scheduler vs +scheduler+Mooncake

Reference for the convergence A/B. Everything not listed here is identical
across arms: image, rllm/verl/vllm pins, model (Qwen3-32B + LoRA r32), seed,
task list and data order, sampling params, topology (1 train + 2 rollout
nodes, 8 replicas at TP=2, fully-async), sandbox backend (kubernetes pods),
gateway store (sqlite). The mooncake master Deployment runs in all arms; A
and B simply never touch it.

## Arm A — vanilla (baseline)

- Gateway worker selection: rllm's stock `StickyLeastLoadedPolicy`
  (LRU session→worker map, least-loaded fallback).
- vLLM replicas: plain `vllm serve` as verl launches them (only the
  tool-choice flags every arm needs for mini-swe-agent).
- Measures: the reference reward curve, KL/entropy, throughput.

## Arm B — vanilla + our scheduler (routing only)

Delta vs A (one line in the launch config):
```
+rllm.gateway.routing_policy=integration.rllm.policy.SchedulerRoutingPolicy
```
plus env on the pods: `ROUTER_CONFIG_PATH=.../integration/rllm/configs/scheduler.yaml`,
`PYTHONPATH=/opt/py-inference-scheduler`, optional `RLS_DECISION_LOG=1`,
`RLS_METRICS_INTERVAL_MS` (default 100), `RLS_SESSION_HEADER` (default
x-rls-session-id).

- Same gateway, same workers - only the worker-selection function changes:
  py-inference-scheduler `Scheduler` scoring each request.
- KV/engine behavior identical to A (no Mooncake, no eviction).
- Isolates: does routing alone bend the reward curve?
- Status: VALIDATED live 2026-08-13 (32b-policy-smoke: 1,932 decisions,
  4 replicas, sticky per session).

## Arm C — scheduler + Mooncake overscheduling

Delta vs B, all through channels proven in the smokes:

1. Engine kwargs (the Gate-3 channel,
   `actor_rollout_ref.rollout.engine_kwargs.vllm.*`), values built by
   `py_inference_scheduler.datalayer.connectors.mooncake.kv`:
   - `kv_transfer_config` → `DecodeKVSavingConnector` (kv_both; saves decode
     KV to the Mooncake store, not just prefill) — nested-dict survival
     through OmegaConf→argv is the one channel variant not yet smoke-tested
     (scalars proven; fallback: JSON-string kwarg → pinned verl one-liner).
   - `scheduler_cls` → `OverschedulingScheduler` (evict-on-turn-end).
   - `prefix_caching_hash_algo=sha256_cbor`.
2. Env flags on rollout pods: `ENABLE_MOONCAKE_KV=1`,
   `ENABLE_MOONCAKE_OVERSCHEDULING=1`, `MOONCAKE_OVERSCHEDULING_OFFLOAD_ALL=1`
   (every finished turn offloads - RL clients can't tag per-request),
   `MOONCAKE_OVERSCHEDULING_PRESERVE_PREFIX_BLOCKS=<measured>`,
   `MOONCAKE_OVERSCHEDULING_MIN_KV_USAGE=<f>`, `MOONCAKE_CONFIG_PATH`,
   `PYTHONHASHSEED=0` (all replicas), `MC_FORCE_HCA=1`, `WITH_NVIDIA_PEERMEM=0`;
   pods mount the mooncake ConfigMap and hold `mooncake-rdma-8` claims.
- Behavior delta vs B: on each turn end, the request's private HBM blocks are
  evicted (KV already saved to the shared RDMA store); the next turn pulls
  KV back on whatever replica the scheduler picks - cross-replica, cross-node.
- Isolates: does turn-end eviction + store pulls bend the curve beyond
  routing effects?
- Status: connectors validated on the Ray Serve stack (tau3); NOT yet run
  under verl-launched vLLM - that is the arm-C smoke (Gate 6c).

## The scheduler profile (shared by B and C, frozen)

`configs/scheduler.yaml` is now the FULL tau3 champion profile: saturation
filter (kv 0.95 / waiting 16) + prefix_cache 4.0 (max_prefix_blocks 2048,
lru 262144) + waiting_queue 1.0 + kv_cache 0.5 + sticky_session 4.0,
max_score picker. The two former gaps were closed on 2026-08-13:

- `prefix_cache`: the vendored gateway patch threads the parsed request
  body into body-aware policies (a policy opting in via a `request_body`
  parameter; stock policies untouched). The policy hashes the growing
  `messages` list - validated live: turn 2 of a session scored 0.667 prefix
  match on exactly the worker that served turn 1.
- `saturation` filter: cherry-picked from the saturation-filter lineage
  (482cde1 + 39dd12e + tests).

Weights are the tau3-proven set, NOT re-tuned for RL-rollout traffic -
deliberately: the A/B compares arms under one fixed profile; tuning on this
workload is a separate experiment after convergence parity is established.
If tuning ever happens, it must land in BOTH B and C identically or the
B/C isolation breaks.

## Current overall status (2026-08-13)

Mechanics all proven: full RL loop through the gateway (Gate 2),
engine-kwargs channel (Gate 3), scheduler routing at 32B (arm-B preview).
Remaining before full runs: task-difficulty screen (reward is 0.0 on the
frozen 8-task subset even at 32B - GRPO needs pass@4 in (0,1)), 3-node
separated-topology RayJobs, arm-C smoke.
