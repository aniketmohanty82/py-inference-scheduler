"""Patch the mooncake store connector's preemption/finish accounting hole.

Wedge mechanism (observed twice, store arm only, silent): preemption deletes
a request's stored_requests entry; if the request then finishes without any
post-resume save recreating the entry, _get_and_clear_finished_sending's
finished loop matches neither `== 0` nor `is not None`, so no done-signal is
ever emitted - while the scheduler-side request_finished() delay-frees the
blocks on the strength of pre-preemption saved tokens and waits forever.
Result: one engine pinned at ~98% KV with zero running requests.

Exact-match string replacement; fails loudly if the file drifted.
"""

import os
import py_compile
import sys

import vllm

path = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
)
src = open(path).read()

MARKER = "preemption-orphaned"
if MARKER in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

INIT_OLD = "        self.finished_store_req: set[str] = set()\n"
INIT_NEW = (
    "        self.finished_store_req: set[str] = set()\n"
    "        # req ids whose save-accounting entry was deleted by preemption;\n"
    "        # see _get_and_clear_finished_sending for why they need a\n"
    "        # done-signal of their own.\n"
    "        self._preemption_cleared: set[str] = set()\n"
)

LOOP_OLD = """        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in self.kv_send_thread.stored_requests.copy():
            if (
                self.kv_send_thread.stored_requests[req_id] == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(req_id)
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)
"""

LOOP_NEW = """        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)
            self._preemption_cleared.add(req_id)

        for req_id in self.kv_send_thread.stored_requests.copy():
            if (
                self.kv_send_thread.stored_requests[req_id] == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)
                self._preemption_cleared.discard(req_id)

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(req_id)
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)
                self._preemption_cleared.discard(req_id)
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)
            elif req_id in self._preemption_cleared:
                # Preemption deleted this request's save-accounting entry and
                # no post-resume save recreated it, yet the scheduler is
                # delay-freeing its blocks on the strength of pre-preemption
                # saves - without a done-signal those blocks pin forever.
                finished_sending.add(req_id)
                self._preemption_cleared.discard(req_id)
                logger.warning(
                    "Releasing preemption-orphaned delayed-free request %s",
                    req_id,
                )
"""

for old, new, name in ((INIT_OLD, INIT_NEW, "init"), (LOOP_OLD, LOOP_NEW, "loop")):
    if src.count(old) != 1:
        print(f"PATCH_FAILED: {name} anchor count={src.count(old)}")
        sys.exit(1)
    src = src.replace(old, new)

open(path, "w").write(src)
py_compile.compile(path, doraise=True)
print("PATCHED_OK", path)
