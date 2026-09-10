"""Mooncake store connector: structural fix (v3). Applies to the swe2 base.

Replaces the v1/v2 leaf patches (preemption-cleared rescue, drain-timeout
machinery) with the two structural repairs from
integration/verl/CONNECTOR-INVESTIGATION.md:

1. UNDROPPABLE COMPLETION ON ERROR. Block release is gated on a worker
   completion signal; today its emission is scattered across every path of
   two thread classes, and three separate holes have each pinned an engine.
   The transfer threads' shared run() loop now routes every exception
   through one per-subclass hook:
     - send: decrement save accounting, task_done (what the 08-31 fix did
       for one path, now for all of them);
     - recv: mark ALL of the meta's blocks load-failed, signal
       set_finished_request, task_done - signaling done without
       invalidating would resume the request on garbage KV.

2. SCHEDULER LIVENESS WATCHDOG (tolerant receiver + deadline). The
   scheduler keeps finished requests alive awaiting `finished_sending`,
   and remote-KV waiters awaiting `finished_recving`, with no timeout. Now:
     - unknown ids in finished_sending/recving are dropped with a warning
       (an unknown id means the blocks were already freed; the old assert
       turned a stale signal into EngineDeadError);
     - any finished request still held past VLLM_KV_FREE_WATCHDOG_S
       (default 300s - orders of magnitude above any healthy save) is
       force-freed with a warning; remote-KV waiters that old are warned
       about (never force-promoted: their recovery path needs worker-side
       block invalidation, which hook 1 guarantees).
   300s deliberately trades a ~zero-probability in-flight-DMA race for
   converting silent engine death into a logged 5-minute blip.

Exact-anchor replacements; fails loudly on drift.
"""

import os
import py_compile
import sys

import vllm

BASE = os.path.dirname(vllm.__file__)
STORE = os.path.join(BASE, "distributed/kv_transfer/kv_connector/v1/mooncake/store")


def patch(path: str, pairs: list[tuple[str, str, str]]) -> None:
    src = open(path).read()
    for old, new, name in pairs:
        if new.strip() and new in src:
            print(f"  {name}: already applied")
            continue
        if src.count(old) != 1:
            print(f"PATCH_FAILED: {path} anchor '{name}' count={src.count(old)}")
            sys.exit(1)
        src = src.replace(old, new)
        print(f"  {name}: applied")
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"{os.path.basename(path)} OK")


# ------------------- worker.py: completion-on-error hook ----------------
WORKER = os.path.join(STORE, "worker.py")
patch(
    WORKER,
    [
        (
            """            except Exception as e:
                logger.exception("Error in %s: %s", self.name, e)
                try:
                    req_id = getattr(request_data, "req_id", None)
                    if req_id is not None:
                        if hasattr(self, "dec_stored_request"):
                            self.dec_stored_request(req_id)
                        self.request_queue.task_done()
                except Exception:
                    logger.exception(
                        "Failed to release request after error in %s", self.name
                    )
""",
            """            except Exception as e:
                logger.exception("Error in %s: %s", self.name, e)
                try:
                    if request_data is not None:
                        self._complete_on_error(request_data)
                except Exception:
                    logger.exception(
                        "Failed to release request after error in %s", self.name
                    )
""",
            "run() error hook dispatch",
        ),
        (
            """    def _handle_request(self, req_meta: Any):
        pass
""",
            """    def _handle_request(self, req_meta: Any):
        pass

    def _complete_on_error(self, req_meta: Any) -> None:
        \"\"\"Release everything a dropped request would otherwise pin.

        The scheduler holds KV blocks until this thread's completion signal
        arrives and has no timeout of its own, so EVERY dequeued item must
        complete - including ones whose _handle_request raised. Subclasses
        override with their side's completion semantics.
        \"\"\"
        self.request_queue.task_done()
""",
            "base hook",
        ),
        (
            """    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1
""",
            """    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def _complete_on_error(self, req_meta: Any) -> None:
        req_id = getattr(req_meta, "req_id", None)
        if req_id is not None:
            self.dec_stored_request(req_id)
        self.request_queue.task_done()
""",
            "send hook",
        ),
        (
            """    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_block_ids = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_block_ids
""",
            """    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_block_ids = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_block_ids

    def _complete_on_error(self, req_meta: Any) -> None:
        # The request will be promoted out of WAITING_FOR_REMOTE_KVS the
        # moment we signal done; every block this load was going to fill
        # must be marked invalid FIRST or it resumes on garbage KV.
        block_ids = getattr(req_meta, "block_ids", None) or ()
        flat: list[int] = []
        for group in block_ids:
            if group:
                flat.extend(group)
        if flat:
            self._add_load_error_block_ids(flat)
        req_id = getattr(req_meta, "req_id", None)
        if req_id is not None:
            self.set_finished_request(req_id)
        self.request_queue.task_done()
""",
            "recv hook",
        ),
    ],
)

# load ledger: one INFO line per load enqueue and per rank done-signal, so
# a lost finished_recving is attributable (never-issued vs one-rank-silent
# vs aggregation) from logs instead of live heap forensics.
patch(
    WORKER,
    [
        (
            """            assert self.kv_recv_thread is not None
            self.kv_recv_thread.add_request(request)
""",
            """            assert self.kv_recv_thread is not None
            if os.getenv("RLS_KV_LEDGER"):
                # warning-level: INFO from vLLM worker subprocesses does not
                # reach the driver log, and this is bug-hunt instrumentation.
                logger.warning(
                    "KV load enqueued: req=%s tokens=%d tp_rank=%d",
                    request.req_id,
                    load_spec.token_len,
                    self.tp_rank,
                )
            self.kv_recv_thread.add_request(request)
""",
            "load enqueue ledger",
        ),
        (
            """    def set_finished_request(self, req_id: str):
        with self.done_task_lock:
            self.finished_requests.add(req_id)
""",
            """    def set_finished_request(self, req_id: str):
        if os.getenv("RLS_KV_LEDGER"):
            logger.warning("%s done-signal for req=%s", self.name, req_id)
        with self.done_task_lock:
            self.finished_requests.add(req_id)
""",
            "done-signal ledger",
        ),
    ],
)

# ------------- vllm scheduler: tolerant receiver + watchdog -------------
SCHED = os.path.join(BASE, "v1/core/sched/scheduler.py")
patch(
    SCHED,
    [
        (
            """        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
""",
            """        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            if req_id not in self.requests:
                # Done-signals are async; an unknown id means the request's
                # blocks were already freed (including by the watchdog
                # below) and there is nothing left to release.
                logger.warning("Dropping finished_recving for unknown %s", req_id)
                continue
""",
            "tolerant finished_recving",
        ),
        (
            """        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])
""",
            """        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                # See finished_recving above: unknown id == already freed.
                logger.warning("Dropping finished_sending for unknown %s", req_id)
                continue
            self._free_blocks(self.requests[req_id])

        # Liveness watchdog, run every step (unlike the waiting-queue loop,
        # which the scheduler skips entirely on any step that preempted -
        # under chronic preemption that starves promotion for minutes and
        # was observed re-triggering on the same requests 3x).
        #
        # Three states are rescued here, all of which otherwise hold KV
        # blocks with no other release path in vLLM:
        #  a) finished request awaiting finished_sending -> force-free;
        #  b) remote-KV waiter whose load ALREADY completed -> promote now,
        #     keeping the loaded KV (this is pure bookkeeping latency, so it
        #     is not deadline-gated);
        #  c) remote-KV waiter with no signal past the deadline -> force the
        #     failed-load path so it recomputes instead of hanging.
        # (b) and (c) complete recovery inline rather than setting flags for
        # the scheduling loop to consume, precisely because that loop is the
        # thing that may never run.
        deadline = float(os.getenv("VLLM_KV_FREE_WATCHDOG_S", "300"))
        now = time.monotonic()
        watchdog = self._kv_free_watchdog
        stale: list[Request] = []
        promote: list[Request] = []
        force_fail: list[Request] = []
        for req_id, request in self.requests.items():
            if RequestStatus.is_finished(request.status):
                if now - watchdog.setdefault(req_id, now) > deadline:
                    stale.append(request)
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                if req_id in self.finished_recving_kv_req_ids:
                    promote.append(request)
                elif now - watchdog.setdefault(req_id, now) > deadline:
                    force_fail.append(request)
            else:
                watchdog.pop(req_id, None)

        def _resume(request: Request) -> None:
            # Mirror of the remote-KV arm of
            # _try_promote_blocked_waiting_request.
            self._update_waiting_for_remote_kv(request)
            request.status = (
                RequestStatus.PREEMPTED
                if request.num_preemptions
                else RequestStatus.WAITING
            )
            self._kv_free_watchdog.pop(request.request_id, None)

        for request in promote:
            logger.warning(
                "KV watchdog: promoting %s whose KV load completed but whose "
                "scheduler promotion was starved",
                request.request_id,
            )
            _resume(request)
        for request in force_fail:
            logger.warning(
                "KV watchdog: force-failing load of %s after %.0fs in "
                "WAITING_FOR_REMOTE_KVS without a finished_recving signal; "
                "request will recompute",
                request.request_id,
                deadline,
            )
            request.num_computed_tokens = 0
            self.failed_recving_kv_req_ids.add(request.request_id)
            self.finished_recving_kv_req_ids.add(request.request_id)
            _resume(request)
        for request in stale:
            logger.warning(
                "KV watchdog: force-freeing blocks of finished request %s "
                "after %.0fs without a connector done-signal",
                request.request_id,
                deadline,
            )
            watchdog.pop(request.request_id, None)
            self._free_blocks(request)
        if len(watchdog) > len(self.requests):
            for req_id in list(watchdog):
                if req_id not in self.requests:
                    watchdog.pop(req_id, None)
""",
            "tolerant finished_sending + watchdog",
        ),
        (
            """        self.finished_recving_kv_req_ids: set[str] = set()
""",
            """        self.finished_recving_kv_req_ids: set[str] = set()
        # first-seen times for the KV-free liveness watchdog (see
        # _update_from_kv_xfer_finished).
        self._kv_free_watchdog: dict[str, float] = {}
""",
            "watchdog state init",
        ),
    ],
)

ssrc = open(SCHED).read()
if "\nimport os\n" not in ssrc:
    if ssrc.count("\nimport time\n") != 1:
        print("PATCH_FAILED: scheduler import anchor")
        sys.exit(1)
    open(SCHED, "w").write(ssrc.replace("\nimport time\n", "\nimport os\nimport time\n", 1))
    py_compile.compile(SCHED, doraise=True)
    print("scheduler.py os import added")

print("ALL_PATCHES_OK")
