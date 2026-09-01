# Pull-from-Mooncake vs straight recompute under sustained KV saturation

## RETRACTION (2026-08-31): BOTH store-arm runs were wedged - the 2.6x
## same-work claim is WITHDRAWN

Per-snapshot timeline dumps of the store raw logs (dump via
pb2_analyze/dump tooling) show both store runs' engine counters FROZEN
minutes after saturation: store #1 pinned at gen=115,981 /
computed=820,941 / kv=0.98 from +4.5 min to log end; store #2 pinned at
gen=126,334 from +6 min. This is the engine deadlock documented in
WEDGE-BUG.md (preemption race in the save path - present in EVERY store
run since 08-28, silent because the catch logged one line and the RayJob
still reported SUCCEEDED). The celebrated "stable 97-99% KV" WAS the
deadlock. The store arms completed only a fraction of the 64-trajectory
workload; comparing their "windows" against fully-completed recompute
runs was invalid.

What remains valid in this file: the recompute-arm measurements (v4, v5,
and the 08-28 measured rerun - continuous progression, full completion,
collapse replicated 3x). The only valid store-vs-recompute comparison to
date is the pb2 2-node pair with the FIXED connector (PB2.md): wall-clock
parity with 13.4x less true prefill compute (by_source-corrected
semantics: store arm local_compute 1.83M vs recompute 24.6M prompt
tokens; the store+cache served 94.4% of the store arm's prompt tokens).

Counter-semantics correction that also affects numbers below: in this
vLLM build `prompt_tokens_total` = local_compute + local_cache_hit +
external_kv_transfer (verified: by_source labels sum exactly to it), so
every "computed" figure in the original text is actually "processed";
true compute is by_source{local_compute} only.

Original text preserved below for audit; read store-arm numbers as
retracted.

2026-08-28, cluster rls-ab-west, branch rllm-convergence. Question: in the
preemption-heavy regime (single TP=2 replica, gmu 0.30, 64 concurrent
trajectories, 32 distinct r2egym tasks x n=2, Qwen3-32B), does pulling
evicted KV from the Mooncake store beat recomputing it?

Arms differ by EXACTLY one thing: the store arm adds
`kv_transfer_config={kv_connector: DecodeKVSavingConnector, kv_role:
kv_both, save_decode_kv: true}` (+ sha256_cbor hashing). NO
OverschedulingScheduler in either arm - vLLM's own pressure eviction does
the evicting; the store either rescues the evicted history or the engine
recomputes it. Same image, seed, task set, topology, scheduler profile.
Two runs per arm.

## Results (saturated window: first kv>=0.95 sample to end of load)

| | recompute #1 (v4) | recompute #2 (v5) | store #1 | store #2 |
|---|---|---|---|---|
| saturated window (same 64-trajectory workload) | 66 min | 70 min | **26 min** | **26 min** |
| preemptions in window | 133 | 141 | **7** | **7** |
| KV usage mean | 86.5% (osc. 28-100%) | 88.4% (osc. 53-100%) | 97.6% (stable 94-98) | 98.5% (stable 94-100) |
| local prefix-cache lookup hit rate | 2.4% | 2.4% | 6.6% | 6.2% |
| store-tier hit rate (external_prefix) | - | - | **66.6%** (270k/405k) | **60.3%** (229k/380k) |
| master batch_put_end (saves) | 0 (no store) | 0 | 83k+ | (cumulative) |

## Read

- **~2.6x same-work throughput with the store** (66/70 min -> 26/26 min for
  the identical trajectory workload), replicated exactly. Preemptions drop
  19-20x (133/141 -> 7/7).
- Mechanism: the store answers ~2/3 of external lookups, so evicted
  histories reload instead of re-prefilling; requests finish sooner; the
  requeue storm never builds; the engine sits at a STABLE 97-99% KV
  (healthy full utilization) instead of the fill/evict/collapse oscillation
  of the recompute arm.
- The local (HBM) hit rate stays low in both arms - under this pressure the
  HBM cache cannot hold anything; the store tier is where reuse survives.
- Lookup-volume caveat: per-scheduling-attempt query counters vary 40x
  between store runs (retry-storm sensitivity in the drain tail) - use
  window duration, preemptions, and store hit rate as the robust metrics,
  not lookup counts.
- Tail variance: total job wall differs run to run (29-80 min) due to
  runaway-trajectory retries in the drain phase (both arms suffer it; the
  planned turn cap addresses it). Window metrics exclude the drain.

## Diagnosis correction (supersedes 32b-mooncake-smoke/RESULT.md)

The 08-14 "store path miswired" finding was a measurement artifact: the
master metrics grep was truncated (head -8) before the `master_batch_*`
counters - the engines always used batch RPCs (batch_put_end 83k+,
batch_exist 96k+). The rc=-800 gets were cold-store NOT_FOUND during a run
whose saves had not yet accumulated. In-pod RDMA probes (CPU pod over tcp,
GPU worker pod over rdma with device auto-selection) both pass setup/put/get
cleanly. No device_name pinning was needed for same-node traffic; pin it
for the 3-node arms per the rdma_utils warning.

## Implications for arm C

The store connector ALONE (no eviction policy) already delivers the win in
the collapse regime, via cache-hit protection - exactly the mechanism
HANDOFF.md predicted. Evict-on-turn-end (OverschedulingScheduler) is
expected to shift the win earlier (below-saturation proactive freeing);
that comparison is arm C vs this store arm, and is now unblocked.

## Measured token throughput (2026-08-28 rerun of the recompute arm with
## token counters - no imputation)

Recompute arm rerun (rayjob-32b-pullbench-recompute.yaml = store manifest
minus ONLY the kv_transfer_config line). Its saturated window, measured:

| | store #1 | store #2 | recompute (measured) |
|---|---|---|---|
| window | 25.5 min (all 64 trajectories done) | 25.5 min (all 64) | 55.5 min (56/64 done, tail still crawling) |
| computed prompt tokens | 245k | 269k | **19.8M** |
| computed prompt tok/s | 160 | 176 | **5,956** |
| generation tok/s | 25 | 35 | 318 |
| combined engine tok/s | 234 | 238 | 6,274 |
| computed as % of served | 77% | 87% | **99.9%** |
| preemptions | 7 | 7 | 128 |
| local hit rate | 6.6% | 6.2% | 2.3% |

(Counter note: in this vLLM build `request_prompt_tokens_sum` equals
`prompt_tokens_total` - both count COMPUTED prompt tokens.)

Correct reading - the raw tok/s comparison INVERTS naively: the recompute
arm shows 27x higher engine token throughput because it is 27x busier doing
redundant work. Same 64-trajectory workload:

- recompute: **19.8M prompt tokens computed** (99.9% redundant re-prefill
  of evicted/preempted history), 5,956 tok/s of sustained wasted compute,
  and still 8 trajectories unfinished at 55.5 min.
- store: **~0.25M prompt tokens computed** (the store + cache supplied the
  rest), all 64 trajectories done in 25.5 min.

=> ~75-80x less prefill compute for the same delivered trajectories, which
converts to >=2.2x faster same-work sampling in this measured pair (2.5-2.7x
vs the v4/v5 baselines). "Throughput" for RL sampling must be counted in
delivered trajectories per wall-clock, not engine tokens per second - engine
tok/s under thrash measures the size of the fire, not the output.

This rerun is also the third replication of the recompute collapse
(2.3% hit rate, 128 preemptions - v4: 2.4%/133, v5: 2.4%/141).

Raw: pullbench_store_v1/v2_raw.log, pullbench_recompute_v1_raw.log;
earlier recompute baselines: v4/v5_metrics_raw.log.
Manifests: rayjob-32b-pullbench-store.yaml / -recompute.yaml.
