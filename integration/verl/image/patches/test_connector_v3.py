"""Deterministic tests for connector_v3 (run inside the patched image, no GPU).

Covers every behavior the structural fix claims:
  1. send hook: exception -> save accounting decremented + task_done
  2. recv hook: exception -> ALL meta blocks invalidated BEFORE done-signal
  3. run() wiring: every raising item completes; the loop survives
  4. watchdog: finished request past deadline -> force-freed
  5. watchdog: fresh finished request -> untouched
  6. tolerant receiver: unknown done-id -> dropped, no assert
  7. remote-KV waiter past deadline -> warned, never force-freed
"""

import os
import queue
import threading
import time
import types
from collections import defaultdict
from unittest.mock import Mock

os.environ["VLLM_KV_FREE_WATCHDOG_S"] = "0.2"

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E402
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402


def mk_thread(cls):
    t = object.__new__(cls)
    threading.Thread.__init__(t)  # skip subclass __init__ only
    t.name = cls.__name__
    t.request_queue = queue.Queue()
    t.done_task_lock = threading.Lock()
    t.finished_requests = set()
    return t


def test_send_hook_decrements_and_completes():
    t = mk_thread(KVCacheStoreSendingThread)
    t.stored_requests = defaultdict(int)
    t.stored_requests["r1"] = 2
    t.request_queue.put(object())
    t.request_queue.get()
    t._complete_on_error(types.SimpleNamespace(req_id="r1"))
    assert t.stored_requests["r1"] == 1, t.stored_requests
    assert t.request_queue.unfinished_tasks == 0


def test_recv_hook_invalidates_before_signal():
    t = mk_thread(KVCacheStoreRecvingThread)
    t._invalid_block_ids_lock = threading.Lock()
    t._invalid_block_ids = set()
    t.request_queue.put(object())
    t.request_queue.get()
    t._complete_on_error(types.SimpleNamespace(req_id="r2", block_ids=([1, 2], [3])))
    assert t._invalid_block_ids == {1, 2, 3}, t._invalid_block_ids
    assert "r2" in t.finished_requests
    assert t.request_queue.unfinished_tasks == 0


def test_run_routes_every_exception_to_hook():
    calls = []

    class Boom(KVTransferThread):
        def _handle_request(self, m):
            raise RuntimeError("boom")

        def _complete_on_error(self, m):
            calls.append(m.req_id)
            self.request_queue.task_done()

    t = mk_thread(Boom)
    t.ready_event = threading.Event()
    threading.Thread(target=t.run, daemon=True).start()
    t.request_queue.put(types.SimpleNamespace(req_id="a"))
    t.request_queue.put(types.SimpleNamespace(req_id="b"))
    t.request_queue.join()  # only reachable if task_done ran for BOTH
    assert calls == ["a", "b"], calls


def mk_sched():
    s = object.__new__(Scheduler)
    s._update_waiting_for_remote_kv = Mock()
    s.requests = {}
    s.connector = None
    s.finished_recving_kv_req_ids = set()
    s.failed_recving_kv_req_ids = set()
    s._kv_free_watchdog = {}
    s.kv_cache_manager = Mock()
    return s


def mk_req(req_id, status):
    return types.SimpleNamespace(
        request_id=req_id,
        status=status,
        num_preemptions=0,
        num_computed_tokens=123,
        is_finished=lambda: RequestStatus.is_finished(status),
    )


OUT_EMPTY = types.SimpleNamespace(finished_recving=None, finished_sending=None)


def test_watchdog_frees_stale_finished_request():
    s = mk_sched()
    s.requests["x"] = mk_req("x", RequestStatus.FINISHED_STOPPED)
    s._update_from_kv_xfer_finished(OUT_EMPTY)  # registers first-seen
    assert "x" in s.requests
    time.sleep(0.25)
    s._update_from_kv_xfer_finished(OUT_EMPTY)
    assert "x" not in s.requests, "stale finished request must be force-freed"
    assert s.kv_cache_manager.free.called
    assert "x" not in s._kv_free_watchdog


def test_watchdog_leaves_fresh_finished_request():
    s = mk_sched()
    s.requests["y"] = mk_req("y", RequestStatus.FINISHED_STOPPED)
    s._update_from_kv_xfer_finished(OUT_EMPTY)
    assert "y" in s.requests
    assert not s.kv_cache_manager.free.called


def test_tolerant_unknown_done_ids():
    s = mk_sched()
    out = types.SimpleNamespace(finished_recving=["ghost1"], finished_sending=["ghost2"])
    s._update_from_kv_xfer_finished(out)  # must not raise


def test_watchdog_force_fails_stale_remote_kv_waiter():
    s = mk_sched()
    s.requests["z"] = mk_req("z", RequestStatus.WAITING_FOR_REMOTE_KVS)
    s._update_from_kv_xfer_finished(OUT_EMPTY)
    time.sleep(0.25)
    s._update_from_kv_xfer_finished(OUT_EMPTY)
    # recovery completes inline (the scheduling loop that would normally do
    # it is skipped on preempting steps)
    assert s.requests["z"].num_computed_tokens == 0
    assert "z" in s.failed_recving_kv_req_ids
    assert s._update_waiting_for_remote_kv.called
    assert s.requests["z"].status == RequestStatus.WAITING
    assert "z" not in s._kv_free_watchdog


def test_watchdog_promotes_completed_load_without_deadline():
    s = mk_sched()
    s.requests["p"] = mk_req("p", RequestStatus.WAITING_FOR_REMOTE_KVS)
    s.finished_recving_kv_req_ids.add("p")
    s._update_from_kv_xfer_finished(OUT_EMPTY)  # no sleep: not deadline-gated
    assert s._update_waiting_for_remote_kv.called
    assert s.requests["p"].status == RequestStatus.WAITING
    assert "p" not in s.failed_recving_kv_req_ids, "completed load must not recompute"


def test_watchdog_promotes_preempted_request_to_preempted_status():
    s = mk_sched()
    r = mk_req("q", RequestStatus.WAITING_FOR_REMOTE_KVS)
    r.num_preemptions = 2
    s.requests["q"] = r
    s.finished_recving_kv_req_ids.add("q")
    s._update_from_kv_xfer_finished(OUT_EMPTY)
    assert r.status == RequestStatus.PREEMPTED


def test_fresh_remote_kv_waiter_untouched():
    s = mk_sched()
    s.requests["w"] = mk_req("w", RequestStatus.WAITING_FOR_REMOTE_KVS)
    s._update_from_kv_xfer_finished(OUT_EMPTY)
    assert "w" not in s.failed_recving_kv_req_ids


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(
        (n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)
    ):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{'ALL_TESTS_PASSED' if fails == 0 else f'{fails} FAILURES'}")
    raise SystemExit(1 if fails else 0)
