# Pull-from-Mooncake vs straight recompute under sustained KV saturation

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
| saturated window (same 64-rollout workload) | 66 min | 70 min | **26 min** | **26 min** |
| preemptions in window | 133 | 141 | **7** | **7** |
| KV usage mean | 86.5% (osc. 28-100%) | 88.4% (osc. 53-100%) | 97.6% (stable 94-98) | 98.5% (stable 94-100) |
| local prefix-cache lookup hit rate | 2.4% | 2.4% | 6.6% | 6.2% |
| store-tier hit rate (external_prefix) | - | - | **66.6%** (270k/405k) | **60.3%** (229k/380k) |
| master batch_put_end (saves) | 0 (no store) | 0 | 83k+ | (cumulative) |

## Read

- **~2.6x same-work throughput with the store** (66/70 min -> 26/26 min for
  the identical rollout workload), replicated exactly. Preemptions drop
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

Raw: pullbench_store_v1_raw.log / v2 alongside; recompute arm raws are
v4/v5_metrics_raw.log. Manifest: integration/rllm/k8s/rayjob-32b-pullbench-store.yaml.
