"""Mooncake store connector fixes v2 (applies on top of swe3's v1 patch).

Three changes, each judged against the failure it answers:

1. Tolerant done-signal handling in vLLM's scheduler. The v1 worker-side
   rescue (release preemption-orphaned delayed-free requests) is correct
   when the scheduler delayed the free, but the worker cannot observe that
   decision; when the request took the immediate-free path instead, the
   rescue's done-signal tripped `assert req_id in self.requests` and killed
   an engine (caught by the pair-3 smoke). An unknown id in
   finished_sending/recving means the blocks are already freed - the
   correct handling is drop-and-log, not crash. v1's rescue stays.

2. Save-drain failsafe. At 256-wide rollouts, finished requests parked in
   delay-free behind the save queue held blocks long enough to
   admission-starve engines (71% of pressured samples). Past
   SAVE_DRAIN_TIMEOUT_S (default 30), the request's remaining QUEUED saves
   are skipped so its counter drains to zero through the normal done path.
   In-flight transfers are never touched: freeing blocks mid-DMA writes
   corrupt KV into the store.

3. The drain marks live in a dedicated `_drain_skip` set, NOT the existing
   store-pressure machinery: `_clear_store_pressure()` wipes that state on
   any successful save batch, which would evaporate timeout marks.

Exact-anchor replacements; fails loudly if the code drifted.
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


# ---- vLLM scheduler: unknown done-signal ids are drop-and-log ----------
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
                # Connector done-signals are async; an unknown id means the
                # request's blocks were already freed (e.g. immediate-free
                # finish) and there is nothing left to release.
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
""",
            "tolerant finished_sending",
        ),
    ],
)

# ---- store worker: drain-skip set + timeout drain of parked requests ---
WORKER = os.path.join(STORE, "worker.py")
patch(
    WORKER,
    [
        (
            """        # Pause store requests when CPU/disk offloading is under pressure.
        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()
""",
            """        # Pause store requests when CPU/disk offloading is under pressure.
        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()
        # Queued saves for these requests complete-as-skipped: the
        # save-drain failsafe marks a finished request here when it has
        # been holding its KV blocks past SAVE_DRAIN_TIMEOUT_S. Separate
        # from _skip_store_requests because _clear_store_pressure() wipes
        # that set on any successful batch.
        self._drain_skip: set[str] = set()
""",
            "drain_skip init",
        ),
        (
            """    def _should_skip_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return self._store_pressure_active and req_id in self._skip_store_requests
""",
            """    def _should_skip_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return self._store_pressure_active and req_id in self._skip_store_requests

    def _drain_should_skip(self, req_id: str) -> bool:
        with self.done_task_lock:
            return req_id in self._drain_skip

    def mark_drain_skip(self, req_id: str) -> bool:
        \"\"\"Returns False if the request was already marked.\"\"\"
        with self.done_task_lock:
            if req_id in self._drain_skip:
                return False
            self._drain_skip.add(req_id)
            return True
""",
            "drain_skip methods",
        ),
        (
            """        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return
        if token_len == 0:
""",
            """        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return
        if self._drain_should_skip(req_id):
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return
        if token_len == 0:
""",
            "drain_skip check in _handle_request",
        ),
        (
            """    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            self._skip_store_requests.discard(req_id)
""",
            """    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            self._skip_store_requests.discard(req_id)
            self._drain_skip.discard(req_id)
""",
            "drain_skip cleanup",
        ),
        (
            """        self.finished_store_req: set[str] = set()
        # req ids whose save-accounting entry was deleted by preemption;
        # see _get_and_clear_finished_sending for why they need a
        # done-signal of their own.
        self._preemption_cleared: set[str] = set()
""",
            """        self.finished_store_req: set[str] = set()
        # req ids whose save-accounting entry was deleted by preemption;
        # see _get_and_clear_finished_sending for why they need a
        # done-signal of their own.
        self._preemption_cleared: set[str] = set()
        # first-parked time per finished request awaiting saves; drives
        # the save-drain failsafe in _get_and_clear_finished_sending.
        self._store_req_parked_at: dict[str, float] = {}
""",
            "parked_at init",
        ),
        (
            """            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)
            elif req_id in self._preemption_cleared:
""",
            """            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)
                self._store_req_parked_at.setdefault(req_id, time.monotonic())
            elif req_id in self._preemption_cleared:
""",
            "parked_at tracking",
        ),
        (
            """                logger.warning(
                    "Releasing preemption-orphaned delayed-free request %s",
                    req_id,
                )

        return finished_sending
""",
            """                logger.warning(
                    "Releasing preemption-orphaned delayed-free request %s",
                    req_id,
                )

        # Save-drain failsafe: a finished request parked behind the save
        # queue holds its KV blocks and stalls admission; past the timeout,
        # skip its remaining QUEUED saves (in-flight transfers are never
        # touched) so the counter drains to zero through the normal path.
        drain_timeout = float(os.getenv("SAVE_DRAIN_TIMEOUT_S", "30"))
        now = time.monotonic()
        for req_id in list(self._store_req_parked_at):
            if req_id not in self.finished_store_req:
                self._store_req_parked_at.pop(req_id, None)
                continue
            if now - self._store_req_parked_at[req_id] > drain_timeout:
                if self.kv_send_thread.mark_drain_skip(req_id):
                    logger.warning(
                        "Save-drain timeout (%.0fs): skipping queued saves "
                        "for finished request %s",
                        drain_timeout,
                        req_id,
                    )

        return finished_sending
""",
            "drain loop",
        ),
    ],
)

# worker.py needs os for the env knob
wsrc = open(WORKER).read()
if "\nimport os\n" not in wsrc:
    if wsrc.count("\nimport time\n") != 1:
        print("PATCH_FAILED: worker import anchor")
        sys.exit(1)
    open(WORKER, "w").write(wsrc.replace("\nimport time\n", "\nimport os\nimport time\n", 1))
    py_compile.compile(WORKER, doraise=True)
    print("worker.py os import added")

print("ALL_PATCHES_OK")
