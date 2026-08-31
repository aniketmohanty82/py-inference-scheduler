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

## Root cause (CONFIRMED 08-31, attempt 4: full traceback + py-spy stacks)

Traceback (exc_info image patch) lands on:

```
_handle_request
  addr, size, _ = db.prepare_value(s, e, block_ids_per_group[g_idx])
    block_id = block_ids[start // self.block_size]
IndexError: list index out of range
```

A request PREEMPTED while its async save is still queued has its physical
KV blocks freed; the queued ReqMeta still describes the pre-preemption
token range, so prepare_value indexes past the (now shorter) block_ids
list. run()'s catch logged and continued WITHOUT the
dec_stored_request/task_done completion that every other path (including
tolerated put failures) performs - so the request never reports
save-finished, vLLM holds it forever, held requests pin KV, and the
engine stalls completely. py-spy confirms: sending threads idle on empty
queues post-error; the engine simply waits on completions that will
never arrive.

Why only the pb2 regime: the race needs a preemption to land between
save-enqueue and save-drain. 128-rollout queues are deeper and cross-node
puts drain slower than single-node loopback, widening the window;
single-node 64-rollout runs (5/5 clean) stayed ahead of it.

## Fix (shipped in image 08-31; upstream to vLLM mooncake store connector)

run()'s except now completes the failed request exactly like the skip
paths: dec_stored_request (when present) + task_done, with its own
guard. Semantics: the store is a cache - a failed save drops the blocks
and releases the request; it must never wedge the engine. A follow-up
upstream refinement can clamp the save range to the live block count and
salvage the still-valid prefix chunks instead of dropping the batch.

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
