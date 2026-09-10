"""Tests for decode_save.py policy knobs (run inside the image, no GPU).

F4 - should_flush_decode_save: decode saves aggregate to min_blocks batches.
F5 - min-pull threshold: tiny external matches recompute locally instead of
     parking the request behind an async pull.
"""

import types
from unittest.mock import patch

from py_inference_scheduler.datalayer.connectors.mooncake.decode_save import (
    DecodeKVSavingConnector,
    should_flush_decode_save,
)


def test_flush_legacy_per_block():
    assert should_flush_decode_save(
        token_len=16, num_saved_tokens=0, block_size=16, min_blocks=1
    )
    assert not should_flush_decode_save(
        token_len=15, num_saved_tokens=0, block_size=16, min_blocks=1
    )


def test_flush_aggregates_to_min_blocks():
    kw = dict(num_saved_tokens=64, block_size=16, min_blocks=4)
    assert not should_flush_decode_save(token_len=64 + 63, **kw)  # 3 full blocks
    assert should_flush_decode_save(token_len=64 + 64, **kw)  # 4 full blocks


def test_flush_partial_block_never_counts():
    assert should_flush_decode_save(
        token_len=64 + 17, num_saved_tokens=64, block_size=16, min_blocks=1
    )
    assert not should_flush_decode_save(
        token_len=64 + 15, num_saved_tokens=64, block_size=16, min_blocks=1
    )


def mk_connector(min_pull):
    c = object.__new__(DecodeKVSavingConnector)
    c._flush_generation = 0
    c._flush_generation_stamped = 0
    c.min_pull_tokens = min_pull
    c.max_inflight_loads = 0
    c._inflight_loads = set()
    c.connector_scheduler = None  # skip the local-covered short-circuit
    return c


def test_min_pull_threshold():
    c = mk_connector(1024)
    with patch(
        "py_inference_scheduler.datalayer.connectors.mooncake.decode_save."
        "MooncakeStoreConnector.get_num_new_matched_tokens",
        return_value=(512, True),
    ):
        assert c.get_num_new_matched_tokens(types.SimpleNamespace(request_id="x"), 0) == (0, False)
    with patch(
        "py_inference_scheduler.datalayer.connectors.mooncake.decode_save."
        "MooncakeStoreConnector.get_num_new_matched_tokens",
        return_value=(4096, True),
    ):
        assert c.get_num_new_matched_tokens(types.SimpleNamespace(request_id="x"), 0) == (4096, True)


def test_min_pull_disabled_passes_through():
    c = mk_connector(0)
    with patch(
        "py_inference_scheduler.datalayer.connectors.mooncake.decode_save."
        "MooncakeStoreConnector.get_num_new_matched_tokens",
        return_value=(512, True),
    ):
        assert c.get_num_new_matched_tokens(types.SimpleNamespace(request_id="x"), 0) == (512, True)


def mk_capped(cap):
    c = object.__new__(DecodeKVSavingConnector)
    c._flush_generation = 0
    c._flush_generation_stamped = 0
    c.min_pull_tokens = 0
    c.max_inflight_loads = cap
    c._inflight_loads = set()
    c.connector_scheduler = None
    return c


def _matched(c, req_id, value=(4096, True)):
    with patch(
        "py_inference_scheduler.datalayer.connectors.mooncake.decode_save."
        "MooncakeStoreConnector.get_num_new_matched_tokens",
        return_value=value,
    ):
        return c.get_num_new_matched_tokens(
            types.SimpleNamespace(request_id=req_id), 0
        )


def test_inflight_cap_declines_beyond_limit():
    c = mk_capped(2)
    assert _matched(c, "a") == (4096, True)
    assert _matched(c, "b") == (4096, True)
    assert _matched(c, "c") == (0, False), "third concurrent pull must recompute"
    assert c._inflight_loads == {"a", "b"}


def test_inflight_cap_disabled_by_default():
    c = mk_capped(0)
    for i in range(10):
        assert _matched(c, f"r{i}") == (4096, True)
    assert c._inflight_loads == set()


def test_sync_hits_do_not_count_against_cap():
    c = mk_capped(1)
    # load_kv_async False: no parking, no pre-allocated blocks held
    assert _matched(c, "s1", value=(4096, False)) == (4096, False)
    assert _matched(c, "s2", value=(4096, False)) == (4096, False)
    assert c._inflight_loads == set()


def test_scheduled_requests_release_cap_slots():
    c = mk_capped(2)
    _matched(c, "a")
    _matched(c, "b")
    so = types.SimpleNamespace(
        scheduled_new_reqs=[types.SimpleNamespace(req_id="a")],
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=["b"]),
    )
    with patch(
        "py_inference_scheduler.datalayer.connectors.mooncake.decode_save."
        "MooncakeStoreConnector.build_connector_meta",
        return_value=object(),
    ):
        c.build_connector_meta(so)
    assert c._inflight_loads == set(), "running requests must free their slot"
    assert _matched(c, "d") == (4096, True)


def mk_flusher(enabled=True, worker=None, seen=0):
    c = object.__new__(DecodeKVSavingConnector)
    c.flush_on_reset = enabled
    c._flush_generation = 0
    c._flush_generation_seen = seen
    c._flush_generation_stamped = 0
    c.connector_worker = worker
    return c


class FakeWorker:
    def __init__(self, tp_rank=0, rc=0):
        self.tp_rank = tp_rank
        self.store = FakeStore(rc)


class FakeStore:
    def __init__(self, rc=0):
        self.rc, self.calls = rc, 0

    def remove_all(self):
        self.calls += 1
        return self.rc


def test_reset_cache_arms_generation_without_touching_store():
    w = FakeWorker()
    c = mk_flusher(worker=w)
    assert c.reset_cache() is True
    assert c._flush_generation == 1
    assert w.store.calls == 0, "the wipe must happen worker-side, not here"


def test_worker_flushes_on_new_generation_once():
    w = FakeWorker()
    c = mk_flusher(worker=w)
    meta = types.SimpleNamespace(rls_flush_generation=1)
    with patch("py_inference_scheduler.datalayer.connectors.mooncake.decode_save."
               "MooncakeStoreConnector.bind_connector_metadata"):
        c.bind_connector_metadata(meta)
        c.bind_connector_metadata(meta)  # same generation: no second wipe
    assert w.store.calls == 1


def test_only_tp_rank_zero_issues_remove_all():
    w = FakeWorker(tp_rank=1)
    c = mk_flusher(worker=w)
    with patch("py_inference_scheduler.datalayer.connectors.mooncake.decode_save."
               "MooncakeStoreConnector.bind_connector_metadata"):
        c.bind_connector_metadata(types.SimpleNamespace(rls_flush_generation=1))
    assert w.store.calls == 0, "remove_all is global; one rank must issue it"


def test_worker_flush_survives_store_error():
    w = FakeWorker(rc=-1)
    c = mk_flusher(worker=w)
    with patch("py_inference_scheduler.datalayer.connectors.mooncake.decode_save."
               "MooncakeStoreConnector.bind_connector_metadata"):
        c.bind_connector_metadata(types.SimpleNamespace(rls_flush_generation=1))
    assert w.store.calls == 1  # logged, never raised


def test_lookups_suppressed_while_flush_is_armed():
    c = mk_capped(0)
    c._flush_generation, c._flush_generation_stamped = 1, 0
    assert _matched(c, "a") == (0, False), "must not match keys about to be wiped"
    c._flush_generation_stamped = 1
    assert _matched(c, "a") == (4096, True)


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
