# Mooncake store connector: silent-engine-death fault tree

2026-09-10. Written after two wedged store arms (pair-2 regime), one
engine killed by our own v1 fix (pair-3 smoke), and admission starvation
at 256-wide - to replace patch-and-rerun with a verified failure model.
Sources: vendored vLLM 0.22.1 connector + scheduler (read line-by-line),
recorded counters/py-spy from every incident (operator scratch
`jobs/7c396abe/tmp/pressure/`).

## The invariant that everything hangs off

A request that used the store holds GPU KV blocks past its own lifetime
until a worker-side COMPLETION SIGNAL arrives:

- send side: finish with pending saves -> blocks delay-freed until the
  request appears in `finished_sending` (scheduler
  `_update_from_kv_xfer_finished` is the ONLY release path).
- recv side: external-hit request parks in WAITING_FOR_REMOTE_KVS holding
  its full allocated context until it appears in `finished_recving`; if
  aborted while loading, its blocks stay delay-freed pending that same
  signal, and the request leaves every queue (invisible to running AND
  waiting gauges).

vLLM core has NO timeout, reaper, or fallback on either direction
(verified by grep and read). Therefore: one dropped signal = blocks
pinned until process death. Every incident below is one way to drop the
signal.

## Fault tree (signal-drop paths)

| # | side | path | evidence | status |
|---|---|---|---|---|
| 1 | send | exception in send `_handle_request` (preemption freed blocks under a queued save -> `prepare_value` IndexError); old `run()` catch skipped completion | pb2 traceback + py-spy (08-31) | FIXED 08-31, in image since swe2 |
| 2 | send | preemption deletes the request's `stored_requests` counter; finish-with-counter-None matches no branch in `_get_and_clear_finished_sending`; nothing emitted, no exception raised | static proof; terminal state matched twice (98% KV pinned, 0 running/waiting, counters bit-identical for hours); rescue precondition observed firing 2x in pair-3 smoke | rescue in swe3; made safe in swe4 (below) |
| 2b | send | our v1 rescue for #2 emitted `finished_sending` for requests the scheduler had ALREADY freed -> `assert req_id in self.requests` -> EngineDeadError | rescue warnings and engine death on the SAME engine in the SAME second (16:48:53, pid 1878) | FIXED in swe4: scheduler treats unknown done-ids as drop-and-log (an unknown id means the blocks are already freed - assert was wrong) |
| 3 | recv | exception in recv `_handle_request` BEFORE its inner try: `prepare_value` IndexError (abort/preempt racing a queued load), `tp_rank % len(key_list)` ZeroDivision on fully-masked loads, `strict=True` zip; shared `run()` catch completes send accounting only - `set_finished_request` never called | static proof (read); NOT yet reproduced live; wedges 1-2 are consistent with #2 or #3 and cannot be retroactively distinguished (ray logs lost with the pod) | OPEN - fix F1 below |

Distinct from the wedges, and NOT a bug:

| phenomenon | mechanism | evidence |
|---|---|---|
| admission starvation at 256-wide (71% of pressured samples: running=0, waiting 40+, KV 90%+) | parked load-waiters legitimately hold allocated blocks while ONE recv thread per TP worker drains ~45ms get-ops; save side mirrors it with 1-key/2MB puts paying ~3.9ms overhead each (>98% overhead on this RDMA fabric) | drain-timeout failsafe NEVER fired (so parked SAVES are not the holder); load-waiter accounting read from scheduler source; op sizes/timings recorded |

## Fixes: minimal and sustainable set

Layer 1 - correctness (make completion signals undroppable):

- F1 (recv, REQUIRED): recv `_handle_request` must end in
  `set_finished_request` + `task_done` on EVERY path, with all of the
  meta's block ids marked load-failed first when an exception occurred
  (signaling done without invalidating blocks would let the request
  resume on garbage KV - worse than the wedge). Mirror of the 08-31 send
  fix; ~10 lines.
- F2 (send, DONE in swe4): keep the #2 rescue + tolerant scheduler pair.
  Optional later simplification: with the scheduler tolerant, the rescue
  no longer needs the `_preemption_cleared` gate at all (over-emission is
  safe by construction).
- F3 (upstream): file ONE issue for the class - "async KV connector
  completion signals are droppable and core has no timeout" - carrying
  paths 1/2/2b/3, the recorded terminal states, and our patches; propose
  a core-side watchdog (free + warn after T) as belt. Paths 1 and 2 were
  never upstreamed; check vLLM main before filing, the connector moved.

Layer 2 - capacity (the governor; policy, in OUR repo code, no vendored
patches):

- F4: decode-save aggregation in `decode_save.py` - emit a save meta at
  >= N unsaved full blocks (env knob) instead of every block; kills the
  1-key/2MB put pattern (4-8x fewer ops).
- F5: minimum-pull threshold in `get_num_new_matched_tokens` - below N
  matched external tokens, recompute locally instead of parking the
  request behind a 45ms-per-op pull; under pull-storms small hits are a
  bad trade.
- F6 (deferred until F4/F5 measured): send/recv thread pools.

Layer 3 - validation gate BEFORE any further GPU pair:

- V1: deterministic unit tests against the vendored classes with mocked
  store (no GPU; runnable in the head pod): (a) finish-after-preemption
  emits done; (b) done-signal for an already-freed request is dropped
  without assert; (c) recv exception mid-handle still signals done AND
  invalidates the meta's blocks; (d) drain failsafe fires only past the
  timeout; (e) F4 flush cadence; (f) F5 threshold.
- V2: one 2-step store smoke at 256-wide gating rc/entropy/turns/tool +
  preemptions>0 + zero EngineDeadError + all engines alive, reading the
  new warning counters (rescues, drops, drain fires) as diagnostics.

## Resolution (2026-09-10, image swe8)

Fix set as built, all unit-tested at image build time (19 tests):

| fix | what | where |
|---|---|---|
| completion-on-error hook | every dequeued transfer completes on ANY exception path; recv marks its blocks invalid BEFORE signalling | `patches/connector_v3.py` (vendored vLLM) |
| tolerant done-signals | unknown ids drop-and-log; the old `assert req_id in self.requests` turned a stale signal into EngineDeadError | same |
| liveness watchdog | runs EVERY step in `_update_from_kv_xfer_finished`: force-frees stale finished requests, and for remote-KV waiters PROMOTES those whose load already landed (keeping the KV) or force-fails the rest past `VLLM_KV_FREE_WATCHDOG_S` | same |
| decode-save aggregation | `RLS_DECODE_SAVE_MIN_BLOCKS` batches decode saves instead of one put per 2MB block | `decode_save.py` (ours) |
| minimum pull | `RLS_MIN_PULL_TOKENS` recomputes tiny external matches instead of parking on a pull | ours |
| in-flight load cap | `RLS_MAX_INFLIGHT_LOADS` bounds concurrent async pulls | ours |

**Why the watchdog matters more than expected:** promotion of blocked
waiting requests happens inside `schedule()`'s waiting-queue loop, which is
gated on `if not preempted_reqs`. Under chronic preemption that loop is
skipped for long stretches, so loads that COMPLETED SUCCESSFULLY still
stranded their requests (observed re-firing on the same ids 3x at 300s
intervals). At 256-wide, 2,300+ promotions per smoke came from the
watchdog - i.e. essentially every pull. This is an upstream-worthy bug in
its own right.

**THE SIZING LAW (the actual blocker, found last).** With a KV pool of
2,125 blocks (34k tokens) and trajectory contexts up to 28k tokens, a
single request parked on an async pull holds ~82% of the pool. Unbounded
admission then deadlocked all four engines simultaneously - `running=0`,
`waiting~52`, KV 91-97%, counters frozen for 33+ min - because releasing
blocks requires running and running requires blocks. Capping concurrent
pulls (6, then 2) did NOT fix it: the pool was below one request's
working set, so no cap can help.

Raising `gpu_memory_utilization` 0.317 -> 0.45 (pool 2,125 -> **11,645
blocks = 186k tokens**, ~6-7 full contexts) resolved it completely:

| metric | gmu 0.317 | gmu 0.45 |
|---|---|---|
| requests running per engine | **0 (deadlock)** | 17-26 |
| admission-starved samples | **71-77%** | **0%** |
| steps completed | none, ever | step 1 in 1,146s gen |
| save ops | 1.0 key/op (pre-F4) | **21.5 keys/op** |
| load ops | - | 546.8 keys/op |
| by_source external share | n/a (frozen) | **26.2%** |
| promotions / force-fails / engine deaths | - | 2,300 / **0** / **0** |

Rule to carry forward: **the KV pool must exceed a single request's full
context by a healthy multiple (>= ~5x) or async KV loading cannot work at
all**, independent of connector correctness. Pressure regimes must be
built by raising concurrency, never by shrinking the pool below one
working set - the 0.317 pool was tuned purely to manufacture preemptions
and produced a configuration no serving system would run.

## Standing lessons

- Any state that waits on a cross-component signal needs either an
  undroppable signal (complete-on-every-path) or a timeout. This
  connector had neither, three times.
- A fix that ASSERTS its own precondition remotely (v1 rescue) is a bug
  with better intentions. Tolerant-receiver + at-least-once signaling is
  the stable shape.
- Gauge blind spots cost us a day: WAITING_FOR_REMOTE_KVS requests are
  invisible in `num_requests_running`/`waiting` once aborted; starvation
  and wedge states look identical from those two gauges alone. The
  discriminator is counters ADVANCING vs bit-identical.
