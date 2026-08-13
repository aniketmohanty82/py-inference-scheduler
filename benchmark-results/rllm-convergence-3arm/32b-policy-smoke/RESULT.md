# 32B + SchedulerRoutingPolicy smoke: PASSED 2026-08-13 (arm-B preview)

RayJob `rllm-32b-policy-smoke` SUCCEEDED (branch bbb308b, image
rllm-verl-mooncake:dev). Qwen3-32B + LoRA r32 at TP=2 on one 8xH200 spot
node = 4 vLLM replicas; SchedulerRoutingPolicy live in the gateway;
kubernetes sandbox pods for R2E-Gym tasks. Full metrics: submitter.log.

## Observables

- **Scheduler routed the whole run**: 4 workers tracked (one per TP=2
  replica), **1,932 scorer decision lines** (waiting_queue + kv_cache +
  sticky_session per request), sticky_session rendezvous-hashing different
  sessions to different replicas with per-session stickiness - the
  per-trajectory affinity design working against live verl-managed vLLM.
- **32B trained**: global_step 1 + validation completed, weight sync 6.2s
  (vs 2.5s at 4B), no OOM at gpu_memory_utilization 0.8 alongside LoRA
  FSDP with param/optimizer offload.
- routing_policy reached the gateway subprocess via the vendored
  `rllm.gateway.routing_policy` key; policy import needs
  `PYTHONPATH=/opt/py-inference-scheduler` (integration/ is repo-root code,
  not in the installed package) - now in the RayJob env.

## Open item for the arms (last blocker)

**Reward is still 0.0 at 32B** (solve_none 1.0, all groups "too_hard") on
the 8 frozen orange3 tasks with the 4096-token output cap. Task-selection
problem, not integration: next step is a pass-rate screen over a larger
r2egym slice (or r2egym_lite) to pick a subset with pass@4 strictly
between 0 and 1, which GRPO needs for signal (rllm already computes the
too_easy/too_hard fractions we can screen on).
