# Arm-C mooncake smoke: training loop PASSED, store path MISWIRED (2026-08-13/14)

RayJob `rllm-32b-mooncake-smoke` SUCCEEDED as a training run in SEPARATED
async mode (4 train GPUs + 2 rollout replicas TP=2 on one node,
raise_on_error=false), with the full overscheduling stack loaded on the
verl-launched replicas. Branch through c2abb28+, image digest c70edd2e.

## What is proven

- **Nested-dict engine_kwargs passthrough** (hydra dict → OmegaConf → verl →
  vllm serve): kv_transfer_config with nested extra_config parsed and
  applied; EngineCore logged the custom OverschedulingScheduler on every
  replica. The last untested Gate-3 variant - no verl fork needed.
- **Separated/fully-async mode works** end to end with the gateway path and
  our routing policy (this is the 3-arm topology, minus node count). The
  colocated attempts OOMed at weight sync / sleep-wake: pinned Mooncake RDMA
  buffers don't participate in vLLM's cumem sleep - separated mode is not
  just preferred but REQUIRED for arm C.
- Mooncake transfer engine initialized on workers (protocol=rdma; rdma_utils
  device auto-selection warnings - set device_name CSV for the real arms).
- The connector's LOOKUP path fired: workers attempted batched GETs.

## What is broken (the next debugging session)

- **Zero puts on our mooncake master** (all master_* counters 0) while
  workers logged "Failed to get 38 Mooncake keys ... rc=-800" - gets were
  attempted against a store that never received saves, and our master saw
  no RPCs at all.
- **Prime suspect: store-config collision with verl 0.8's TransferQueue**,
  which runs its OWN MooncakeStore on localhost (config table showed
  transfer_queue.backend.MooncakeStore master localhost:50124, protocol
  tcp). Check whether verl sets/overrides MOONCAKE_CONFIG_PATH or the
  mooncake client singleton in-process, redirecting our connector's
  master_server_address away from mooncake-master.default.svc:50051.
- No `overschedule: evicted` lines and no offload-all init line anywhere in
  engine logs; INFO-level suppression in EngineCore is possible, but with
  the save path dead, evictions would be unsafe anyway (evicting unsaved KV)
  - resolve the store wiring first, then re-verify offload.

## Repro/debug entry points

- engine-mooncake-lines.log here (200 lines of worker/engine mooncake output)
- verl transfer_queue config keys: `transfer_queue.backend.MooncakeStore*`
  (can it be disabled or pointed at our master?)
- vllm 0.22.1 pip MooncakeStoreConnector vs the ray-llm-mooncake image's
  0.22.0 build (save-trigger semantics may differ; tau3 validation was on
  the image build)
- mooncake master metrics: mooncake-master.default.svc:9003/metrics
