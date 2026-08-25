# Handoff: prefix-cache hit-rate collapse under KV saturation (RL rollout workload)

Self-contained summary for continuation work. Measured 2026-08-13 through
2026-08-18 on GKE cluster `rls-ab-west` (us-west1-c), branch
`rllm-convergence` of py-inference-scheduler (worktree
`.claude/worktrees/rllm-convergence`).

## One-paragraph result

DeepSWE-style RL rollouts (rllm + verl, Qwen3-32B + LoRA, mini-swe-agent on
R2E-Gym tasks in k8s sandbox pods, vLLM 0.22.1 engines behind the rllm Model
Gateway running our SchedulerRoutingPolicy with the tau3 champion profile)
hold an **87.7-91.9% vLLM prefix-cache hit rate with zero preemptions** as
long as aggregate KV demand fits the replica pools. When demand is
concentrated past pool capacity (single TP=2 replica, 64 concurrent
trajectories), KV pins at the ceiling (peaks 99.9-100%, 124-141 preemptions
per ~1h window) and the **hit rate collapses to 1.4-2.4%** - reproduced at
identical config, dose-responsive to pool size, and present even at the
largest single-replica pool tested. This is the recompute-cascade regime
that Mooncake overscheduling (evict-on-turn-end + shared-store pulls) is
designed to relieve; the arm-C comparison in this regime is the next
experiment once the store wiring is fixed.

## The data

Unsaturated (4 replicas at TP=2 on one 8xH200 node):

| run | tasks x samples | gmu | peak KV | preempt | hit rate |
|---|---|---|---|---|---|
| v1 | 8 x 8 (GRPO siblings) | 0.40 | 26% | 0 | 91.9% |
| v3 | 64 distinct x 2 | 0.30 | 72% | 0 | 87.7% |

Saturated (SINGLE TP=2 replica, separated mode: 4 trainer GPUs + 2 rollout
GPUs; 32 distinct tasks x n=2, n_parallel_tasks=64; hit rate computed ONLY
inside the saturated window = first kv>=0.95 sample to end of load):

| run | gmu | window | kv mean/max | preempt | lookups | hit rate |
|---|---|---|---|---|---|---|
| v4 | 0.30 | 66 min | 86.5% / 99.9% | 133 | 2.44B | 2.4% |
| v5 (replicate) | 0.30 | 70 min | 88.4% / 99.7% | 141 | 2.92B | 2.4% |
| v6b | 0.27 | 68 min | 79.5% / 100% | 125 | 3.42B | 1.4% |
| v7 | 0.35 | 56 min | 90.0% / 99.9% | 124 | 1.48B | 2.3% |

- gmu 0.25 does not boot: 2.25 GiB KV < the 4 GiB single-request floor at
  max_model_len 32768. Hard lower bound of the knob.
- Collapse is a cliff in demand/pool ratio, not a slope in pool size: all
  single-replica points land in the 1.4-2.4% band.
- Requeue churn scales inversely with pool size (3.42B lookups at the
  smallest pool vs 1.48B at the largest, same nominal workload).

## Metric semantics (important caveat)

`vllm:prefix_cache_queries_total` / `hits_total` count block lookups per
SCHEDULING ATTEMPT. Under thrash, preempted and requeued requests re-query
on every scheduling pass, inflating the denominator (billions of lookups vs
low-hundreds-of-millions of plausible prefill tokens). Read "2.4%" as "the
cache answered almost nothing during saturation", not as a token-weighted
prefill-savings percentage. `vllm:num_preemptions_total` and
`vllm:kv_cache_usage_perc` are unaffected by this caveat.

Also: verl launches vLLM with log stats DISABLED - the engines expose no
vllm:* metrics unless you pass
`++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_log_stats=false`.

## Why this workload resists saturation at multi-replica scale

Each agent turn is a separate HTTP request: KV frees at every turn end
(tool-call gap ~4.2s mean vs generation ~6.6s per turn), so ACTIVE memory
stays low and pressure lands on the evictable cache tier. GRPO sampling
(n>1) makes sibling trajectories share long task prefixes, shrinking the
unique working set. Sticky + prefix-affinity routing (our champion profile)
concentrates each trajectory's blocks on one replica. All three effects
protect the cache until demand is deliberately concentrated.

## Repro

- Manifest: `integration/rllm/k8s/rayjob-32b-pressure-smoke.yaml`
  (single-replica separated config; gmu is the pressure knob on the
  `gpu_memory_utilization=` line). Image:
  `us-south1-docker.pkg.dev/aniket-gke-dev/llm-images/rllm-verl-mooncake:dev`.
- Driver: job-dir `tmp/run_pressure.sh <gmu> <outfile>` - patches the
  manifest, deletes the previous RayJob AND orphaned `app=rllm-sandbox`
  pods (relaunches leak them; enough orphans deadlock the head pod's own
  scheduling), applies, polls at 60s, auto-harvests the metrics sidecar at
  terminal state.
- Metrics: in-pod `metrics-scraper` sidecar (must be listed AFTER
  `ray-worker` - KubeRay derives its wait-gcs-ready init container from
  containers[0]). Harvest anytime: `kubectl logs <worker-pod> -c metrics-scraper`.
- Analysis: job-dir `tmp/analyze_v4.py <raw-log>` - windowed deltas of
  cumulative counters (window = first kv>=0.95 through last kv>=0.5,
  drain excluded). Single-replica assumption in the script.
- Raw scrapes: `benchmark-results/rllm-convergence-3arm/pressure-test/*_metrics_raw.log`.
  Full narrative: RESULT.md in the same directory.
- Sandbox pods need `RLLM_K8S_CPU_REQUEST=50m` / `RLLM_K8S_MEMORY_REQUEST_MB=256`
  or the 2-node CPU pool caps at ~16 concurrent task pods.

## Open follow-ups

1. **Arm-C rerun of this sweep** once the Mooncake store path works (as of
   08-14 it is miswired under verl: zero puts on our master, gets fail
   rc=-800; suspect verl 0.8's TransferQueue running its own localhost
   MooncakeStore - see `benchmark-results/rllm-convergence-3arm/32b-mooncake-smoke/RESULT.md`). Expected claim:
   overscheduling holds hit rate (or replaces recompute with store pulls)
   in exactly this collapse regime.
2. A token-weighted hit-rate metric (dedupe requeue lookups) if a
   publishable number is needed.
3. Preemption-count ceiling: all saturated runs land at 124-141 - likely
   bounded by rollout count (64) x retry structure; more preemptions per
   run would need more trajectories or longer contexts.
