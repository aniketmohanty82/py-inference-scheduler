# Store-arm engine deadlock ("the wedge") - bug note, not a benchmark doc

Failure that has invalidated every 2-node store attempt so far (3/3).
Kept separate from the results docs per the valid-runs-only policy.

## Symptom

10-15 min into sampling, both replicas' worker processes start
crash-looping:

```
ERROR [worker.py:393] Error in KVCacheStoreSendingThread: list index out of range
```

(vLLM mooncake store connector transfer-thread scaffolding,
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py).
The thread survives (run() catches), but each failed save never marks its
request finished; vLLM holds requests until the connector reports the
save done, so requests accumulate in limbo, KV blocks never free, and
BOTH engines stop serving: gen/prompt counters frozen, KV gauge pinned
(0.79/0.93), queues full, GPUs idle. Pods stay Running and the RayJob
later reports SUCCEEDED because rllm.workflow.raise_on_error=false
absorbs aborted trajectories - job status is NOT evidence of a completed
workload; counter drain is.

## Reproductions

| attempt | segments | onset | evictions at onset | evidence |
|---|---|---|---|---|
| 1 (08-31) | 4x32gb | ~20 min, after store filled | 13 sweeps, 81.1GB | error spam in driver log; counters froze; salvaged logs |
| 2 | 4x96gb | ~7.5 min | ZERO | full timeline dump shows 18 min frozen, no drain |
| 3 | 4x96gb | ~13 min | zero | 771 error lines across both replicas in streamed driver log |

Attempt 2 kills the store-full theory: the wedge is not capacity- or
eviction-triggered. The same connector config ran clean 5/5 on
single-node (2 store + 3 recompute runs, 64 rollouts, one replica); the
wedge appears only in the 2-replica / 128-rollout / cross-node regime.

## Candidate fault sites (from reading _handle_request)

- `req_meta.block_hashes[chunk_idx]` - chunk loop indexes request block
  hashes by token-db chunk index.
- `db.prepare_value(s, e, block_ids_per_group[g_idx])` - save uses block
  ids captured at enqueue; a preemption frees/reshapes them before the
  async save drains (both attempts wedge shortly after preemptions
  begin).

## Instrumentation now in place (attempt 4, in flight)

| channel | mechanism |
|---|---|
| full traceback | image patches the transfer-thread catch to logger.exception (Dockerfile) |
| live stacks | freeze_catcher.sh: 2-min counter polls; on frozen streak, py-spy every engine process on both rollout pods via `kubectl debug --profile=sysadmin` (privileged ephemeral container - bypasses in-pod ptrace denial) |
| driver stream | kubectl logs -f from launch (stream_driver.sh) |

Lesson recorded the hard way: attempt 3's wedged pods were torn down
before dumping stacks - `kubectl debug` could have produced the diagnosis
from the live process in minutes. Never tear down a wedged pod before
py-spy.

## Fix direction (once the line is confirmed)

Connector-side: on save failure, mark the request finished (skip/drop the
block) instead of leaving it pending - the store is a cache, a failed
save must never wedge the engine. Upstream to the PR #46 lineage.

Context values while healthy (attempt-1/3 runs, bug-run context only):
store answered 84%+ of lookups; master allocation uniform across all 4
segments (2 per node), so cross-node pulls demonstrably function.
Raw evidence in job tmp: pb2_store_attempt{1,3}_driver.log,
pb2_store_attempt3_scraper.log, pb2_master_snapshots.log,
pb2_store_v1_raw.log (attempt-2 sidecar; retracted from results).
