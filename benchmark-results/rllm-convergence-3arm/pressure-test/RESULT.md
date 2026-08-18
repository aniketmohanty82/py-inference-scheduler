# Prefix-cache hit rate under KV pressure (rllm 32B rollouts), 2026-08-14/17

Question: run concurrency/batch far above per-replica HBM and measure cache
hit rate. Setup: champion-profile scheduler, 4 replicas TP=2 on one 8xH200
node, mini-swe-agent on r2egym, contexts to 16k, output cap 4096.

## v1 - 64 trajectories, 8 distinct tasks (GRPO n=8), gmu 0.40, 43 min

| replica | queries | hits | hit rate | preempt | peak KV% |
|---|---|---|---|---|---|
| A | 4.20M | 3.92M | 93.4% | 0 | 19.5% |
| B | 2.42M | 2.25M | 92.9% | 0 | 21.2% |
| C | 2.13M | 1.89M | 88.9% | 0 | 15.1% |
| D | 3.54M | 3.23M | 91.3% | 0 | 25.9% |
| **agg** | **12.29M** | **11.30M** | **91.9%** | **0** | - |

First half 89.9% -> second half 94.5%. NOT saturated: GRPO siblings share
task prefixes (8 real contexts, not 64) and turn-end frees keep active KV
tiny.

## v3 - 64 DISTINCT tasks (n=2), gmu 0.30, 30 min of rollouts
(v2 = same config, measurement lost to scraper failure; v3 adds an in-pod
metrics-scraper sidecar - the reliable pattern for all future runs)

| replica | queries | hits | hit rate | preempt | peak KV% |
|---|---|---|---|---|---|
| A | 4.35M | 4.08M | 93.7% | 0 | 71.8% |
| B | 2.48M | 2.26M | 91.3% | 0 | 57.3% |
| C | 4.91M | 4.72M | 96.1% | 0 | 51.8% |
| D | 26.49M | 22.45M | **84.7%** | 0 | 71.1% |
| **agg** | **38.22M** | **33.51M** | **87.7%** | **0** | - |

First half 85.0% -> second half 96.3%. Run ended on the known
runaway-trajectory context overflow (3 retries exhausted) after the heavy
rollout phase - counters cover the full pressure window.

## Read

- **3x the pressure (peak KV 52-72% vs 15-26%), still zero preemptions.**
  Aggregate hit rate dropped 91.9% -> 87.7%; the hot replica dropped to
  84.7%. Cache eviction pressure is visible but far from collapse.
- **Replica D is the story**: it absorbed 69% of all queried tokens (26.5M)
  - the runaway trajectory's 19-turn retries re-prefilled its whole history
  repeatedly, and sticky+prefix affinity (correctly) kept that on one
  replica. The prefix cache absorbed 84.7% of that flood; without affinity
  those re-prefills would have been full recomputes on cold replicas.
- **This workload shape resists HBM saturation by construction**: each turn
  is a separate request, so KV frees at every tool call; the 4096 output
  cap bounds decode length; active memory stays low and the fight is over
  the evictable cache tier. To force preemption-level pressure you need
  32k+ contexts, gmu <=0.25, or single-replica concentration - noted for
  the overscheduling A/B design: the arm-C benefit case on THIS task mix
  is cache-hit protection (avoiding re-prefill of evicted history), not
  active-memory relief.

Raw scrapes: v3_metrics_raw.log (61 snapshots, 30s cadence). Analysis
script: job tmp analyze_hitrate.py (final-snapshot cumulative counters).

## v4 - SINGLE TP=2 replica, gmu 0.30, 64-way concurrency: sustained thrash

Separated mode (trainer 4 GPUs + rollout 2 GPUs), 32 distinct tasks x n=2,
n_parallel_tasks=64, ~85k-token KV pool vs multi-hundred-k demand.
Requires sandbox requests dropped to 50m/256Mi (CPU pool caps at ~16 pods
otherwise) and orphaned-sandbox cleanup between runs (relaunches leak the
previous run's 64 pods and can deadlock the head pod's own scheduling).

Saturated window (first kv>=0.95 to end of load, 66 min, 132 samples):

| metric | value |
|---|---|
| kv usage | mean 86.5%, max 99.9%, >=90% for 52% of samples, >=95% for 36% |
| preemptions | **133** (first nonzero of the whole series) |
| cache-lookup queries in window | 2.44B token-lookups |
| lookup hit rate in window | **2.4%** (2.5% whole-run) |

The pressure gradient across the series is the result:

| run | peak KV | preemptions | hit rate |
|---|---|---|---|
| v1 (8 tasks, 4 replicas, gmu .40) | 26% | 0 | 91.9% |
| v3 (64 tasks, 4 replicas, gmu .30) | 72% | 0 | 87.7% |
| v4 (32 tasks, 1 replica, gmu .30) | 99.9%, pinned | 133 | **2.4%** |

Metric caveat: prefix_cache_queries counts block lookups per SCHEDULING
attempt; under thrash, preempted/requeued requests re-query repeatedly, so
the denominator includes requeue storms (query volume 64x v3 while compute
capacity halved - implausible as real prefill tokens). Read 2.4% as "the
cache answered almost nothing during saturation", not as a token-weighted
prefill-savings figure. Either way this is the recompute cascade
overscheduling targets: every evicted history re-prefills from scratch,
preemption restarts multiply the load, and the engine spends the window
re-doing work it had already done. This is the regime for the arm-C
comparison once the Mooncake store wiring is fixed.
