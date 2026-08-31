# pb2: 2-node cross-node RDMA pullbench (store vs recompute)

2026-08-31, cluster rls-ab-west, branch rllm-convergence. Question: does the
store's win survive when pulls actually cross the node fabric? The
single-node pullbench (PULLBENCH.md) only exercised same-node loopback
(METHODOLOGY.md T13); here each TP=2 replica sits on its OWN H200 node and
the Mooncake master spreads saves across all 4 segments (2 per node), so
~half of all saves and pulls are remote RDMA by construction.

## Setup deltas vs single-node pullbench

| | single-node | pb2 |
|---|---|---|
| nodes / replicas | 1 node, 1x TP=2 | 2 nodes, 2x TP=2 (one per node, k8s-forced) |
| workload | 32 tasks x n=2 = 64 rollouts | 64 tasks x n=2 = 128 rollouts |
| concurrency | 64 (n_parallel_tasks) | 128 |
| mounted segments | 2 x 32gb (one node) | 4 x 96gb (2 per node) |
| topology enforcement | n/a | 2-GPU rollout pods + hostname anti-affinity; 4-GPU trainer pod |
| weight-sync NCCL | in-pod | NCCL_IB_DISABLE=1 + NCCL_SOCKET_IFNAME=eth0 (cross-pod) |

Arms differ by EXACTLY the kv_transfer_config line (plus job names) -
`git diff` of rayjob-32b-pb2-store.yaml vs rayjob-32b-pb2-recompute.yaml.
Same image, seed, task set, scheduler champion profile, both replicas
routed by SchedulerRoutingPolicy through the rllm gateway.

## Results (per-replica window: first kv>=0.95 sample to last kv>=0.5)

| | store @ s2qr | store @ lczl | recompute @ s2qr | recompute @ lczl |
|---|---|---|---|---|
| window | 20.0 min | 21.5 min | 37.5 min | 37.5 min |
| KV usage mean | 80.1% | 92.6% | 86.0% | 84.3% |
| preemptions | 4 | 7 | 53 | 45 |
| computed prompt tok | 256k | 827k | 13.81M | 14.28M |
| computed prompt tok/s | 213 | 641 | 6,136 | 6,344 |
| computed as % of served | 90.6% | 92.2% | 99.9% | 99.9% |
| local (HBM) hit rate | 1.1% | 3.0% | 4.5% | 3.7% |
| store-tier hit rate | 85.0% | 84.6% | - | - |
| tokens loaded FROM store | 182k | 553k | 0 | 0 |

Aggregate (union windows): store 21.5 min / 11 preemptions / 1.08M computed
prompt tokens / 84.7% store hit / 736k tokens pulled from store; recompute
39.0 min / 98 preemptions / 28.09M computed prompt tokens / 99.9% of served
prompt tokens recomputed.

## Read

- **1.8x same-work sampling speed with the store across the fabric**
  (union 39.0 -> 21.5 min for the identical 128-rollout workload).
- **9x fewer preemptions** (98 -> 11) and **26x less prefill compute**
  (28.1M -> 1.08M computed prompt tokens).
- **Cross-node RDMA did not blunt the store**: store-tier hit rate was
  84.7% (vs 66-80% single-node - higher because the 3x larger store never
  evicted), with ~half of the 736k pulled tokens crossing the wire by
  construction (master allocation was uniform across all 4 segments,
  verified live: 27GB +/- 0.3 per segment at plateau in attempt 1,
  ~14.4GB +/- 0.2 mid-ramp in the measured run).
- The engine tok/s inversion replicates: recompute arms sustained
  6.1-6.3k computed-prompt tok/s of 99.9%-redundant re-prefill - the same
  fire-size-not-output signature as the single-node runs (5,956 tok/s).
- The speedup ratio is smaller than single-node (1.8x vs 2.6x) because the
  RECOMPUTE arm improved with two replicas: per-replica preemptions fell
  from 128-141 (single-node) to 45-53 - with a second endpoint available,
  the saturation filter can actually shed load instead of being trivially
  degenerate (METHODOLOGY.md T11). The store arm's absolute windows
  (20-21.5 min) match the single-node store windows (25.5 min).

## Store-full wedge (attempt 1) - real bug, worth upstreaming a fix

With the original 4x32gb segments the store filled mid-window; master
eviction kicked in (13 sweeps, 38,695 keys, 81.1GB) and every save then
crash-looped with `Error in KVCacheStoreSendingThread: list index out of
range` (vllm mooncake store connector scaffolding). Failed saves never mark
requests finished, so BOTH engines deadlocked at ~96% KV with queued
requests and frozen token counters. Mitigation here: 96gb segments so
eviction never triggers (measured run: 0 evictions, 0 put failures, 31,920
batch puts). The failure mode - engine deadlock when the store is
capacity-bound under load - reproduces the moment eviction starts and needs
a connector-side fix (skip-on-failure + mark-finished), independent of this
benchmark.

## Notes / caveats

- Same T2/T5/T8-style caveats as METHODOLOGY.md: trajectory stochasticity,
  per-scheduling-attempt counters (use windows/preemptions/computed tokens
  as primary), drain tails excluded by the window.
- "Computed as % of served" mixes counters with different attempt semantics
  (request sums count finished requests once; computed/by_source count
  every scheduling attempt) - compare across arms, don't read as exact
  fractions.
- device_name still unpinned (T9): Mooncake auto-selected NICs; transfers
  likely concentrated on one HCA per process rather than striping across 8.
  Wire-byte counters were not accessible in-pod (DRANET exposes no IB port
  counters); the cross-node claim rests on master allocation uniformity,
  which the segment gauges verify directly.
- Weight sync rides TCP (NCCL pinned to eth0) - outside the measured
  sampling window; Mooncake's ibverbs data path is unaffected.

Raw: pb2_store_v1_raw.log, pb2_recompute_v1_raw.log (gateway-side sidecar,
both replicas' series), pb2_master_snapshots.log (timestamped master
counters incl. the attempt-1 eviction/wedge evidence). Analyzer:
pb2_analyze.py (per-replica + aggregate windows). Manifests:
integration/rllm/k8s/rayjob-32b-pb2-{store,recompute}.yaml @ 5a81cbb.
