# pb2: 2-node cross-node pullbench - valid results only

2026-08-31, cluster rls-ab-west, branch rllm-convergence. Question: does
pull-from-store keep its win when pulls cross the node fabric? The
single-node pullbench (PULLBENCH.md) exercised only same-node loopback
(METHODOLOGY.md T13). Trajectory lifecycle and shared methodology:
METHODOLOGY.md sec. 3.0.

Per the documentation policy, this file compares ONLY runs confirmed
reliable (continuous counter progression, proper drain, workload
completed). Flawed runs are excluded from all tables; the store-arm
engine-deadlock bug that has so far prevented a valid 2-node store run is
documented separately in WEDGE-BUG.md.

## Run validity ledger

| run | status | where |
|---|---|---|
| single-node store x2, recompute x3 (08-28) | VALID | PULLBENCH.md |
| pb2 recompute (2-node, 08-31) | VALID | this file |
| pb2 store attempts 1-3 (08-31) | invalid (engine deadlock) | WEDGE-BUG.md |
| pb2 store attempt 4 | in flight, instrumented | WEDGE-BUG.md |

## Setup (deltas vs single-node pullbench)

| | single-node | pb2 |
|---|---|---|
| nodes / replicas | 1 node, 1x TP=2 | 2 nodes, 2x TP=2 (one per node, k8s-forced) |
| workload | 32 tasks x n=2 = 64 rollouts | 64 tasks x n=2 = 128 rollouts |
| concurrency | 64 | 128 |
| mounted segments | 2 x 32gb (one node) | 4 x 96gb (2 per node) |
| topology enforcement | n/a | 2-GPU rollout pods + hostname anti-affinity; 4-GPU trainer pod |
| weight-sync NCCL | in-pod | NCCL_IB_DISABLE=1 + NCCL_SOCKET_IFNAME=eth0 |

## Valid result: pb2 recompute arm (baseline for the pending store arm)

Per-replica saturated windows (first kv>=0.95 to last kv>=0.5) plus the
whole sampling phase (generation counters moving; drain included):

| metric | replica A | replica B | aggregate |
|---|---|---|---|
| saturated window | 37.5 min | 37.5 min | 39.0 min (union) |
| KV usage mean | 86.0% | 84.3% | - |
| preemptions | 53 | 45 | 98 |
| computed prompt tokens | 13.81M | 14.28M | 28.09M |
| computed as % of served | 99.9% | 99.9% | 99.9% |
| local (HBM) hit rate | 4.5% | 3.7% | 4.1% |
| sampling phase (whole) | - | - | 51.5 min |
| served tokens in phase | - | - | 32.8M (10,610 tok/s, 99.9% redundant re-prefill) |
| tail: last 10% of prompt work | - | - | 11.5 min |

Cross-node observation that stands on its own: per-replica thrash is
MILDER than single-node recompute at the same per-replica load (45-53
preemptions vs 128-141). With a second endpoint available, the saturation
filter can actually shed load instead of being trivially degenerate
(METHODOLOGY.md T11) - a scheduler effect, measured incidentally.

## Pending

The 2-node store number requires a run that clears WEDGE-BUG.md. Until
then, no cross-node store-vs-recompute ratio is claimed. (Mechanism-level
evidence that cross-node pulls function - 84%+ store hit rate while
healthy, uniform segment fill across both nodes - is recorded in
WEDGE-BUG.md as bug-run context, not as a benchmark result.)

Raw: pb2_recompute_v1_raw.log (gateway-side sidecar, both replicas).
Analyzer: pb2_analyze.py. Manifests: rayjob-32b-pb2-{store,recompute}.yaml;
single-node controls at rayjob-32b-pb2-1n-{store,recompute}.yaml (queued).
