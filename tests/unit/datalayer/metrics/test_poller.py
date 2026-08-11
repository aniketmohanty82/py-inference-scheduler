# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
import time

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.poller import MetricsPoller
from py_inference_scheduler.framework import Endpoint


def _wait_for(cond, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _counting_fetch(counter):
    async def fetch(ep, inflight, session):
        ep.attributes["routing_stats"] = {"tick": next(counter)}

    return fetch


def _failing_fetch():
    async def fetch(ep, inflight, session):
        raise RuntimeError("scrape failed")

    return fetch


def test_staleness_is_infinite_before_first_fetch():
    p = MetricsPoller(list, InflightStore(), _counting_fetch(itertools.count()))
    assert p.staleness() == float("inf")


def test_refreshes_stats_and_becomes_fresh():
    ep = Endpoint(name="a", attributes={})
    p = MetricsPoller(
        lambda: [ep], InflightStore(), _counting_fetch(itertools.count()), interval_ms=10
    )
    p.start()
    try:
        assert _wait_for(lambda: "routing_stats" in ep.attributes)
        assert _wait_for(lambda: p.staleness() < 1.0)
    finally:
        p.stop()


def test_raising_fetch_keeps_polling_but_stays_stale(caplog):
    # fetch callables must not raise. If one does, the poller logs the
    # contract violation and keeps polling, but the metrics stay stale
    calls = itertools.count(1)
    attempts = [0]

    async def fetch(ep, inflight, session):
        attempts[0] = next(calls)
        raise RuntimeError("scrape failed")

    ep = Endpoint(name="a", attributes={})
    p = MetricsPoller(lambda: [ep], InflightStore(), fetch, interval_ms=10)
    p.start()
    try:
        assert _wait_for(lambda: attempts[0] >= 3)
        assert p.staleness() == float("inf")
        assert any("despite contract" in r.message for r in caplog.records)
    finally:
        p.stop()


def test_all_soft_errors_stay_stale():
    # contract-compliant failure: fetch records the error and returns;
    # cycles where every endpoint errored must not refresh staleness
    calls = itertools.count(1)
    attempts = [0]

    async def fetch(ep, inflight, session):
        attempts[0] = next(calls)
        ep.attributes["routing_stats"] = {"error": "HTTP error 503"}

    ep = Endpoint(name="a", attributes={})
    p = MetricsPoller(lambda: [ep], InflightStore(), fetch, interval_ms=10)
    p.start()
    try:
        assert _wait_for(lambda: attempts[0] >= 3)
        assert p.staleness() == float("inf")
    finally:
        p.stop()


def test_partial_failure_still_counts_as_fresh():
    ok, bad = Endpoint(name="ok", attributes={}), Endpoint(name="bad", attributes={})
    counting = _counting_fetch(itertools.count())
    failing = _failing_fetch()

    async def fetch(ep, inflight, session):
        if ep.name == "bad":
            await failing(ep, inflight, session)
        else:
            await counting(ep, inflight, session)

    p = MetricsPoller(lambda: [ok, bad], InflightStore(), fetch, interval_ms=10)
    p.start()
    try:
        assert _wait_for(lambda: p.staleness() < 1.0)
        assert "routing_stats" not in bad.attributes
    finally:
        p.stop()


def test_stop_halts_polling():
    ep = Endpoint(name="a", attributes={})
    p = MetricsPoller(
        lambda: [ep], InflightStore(), _counting_fetch(itertools.count()), interval_ms=10
    )
    p.start()
    assert _wait_for(lambda: ep.attributes.get("routing_stats"))
    p.stop()
    assert _wait_for(lambda: not p._thread.is_alive())
    tick = ep.attributes["routing_stats"]["tick"]
    time.sleep(0.05)
    assert ep.attributes["routing_stats"]["tick"] == tick


def test_new_endpoints_are_picked_up_between_cycles():
    eps: list[Endpoint] = [Endpoint(name="a", attributes={})]
    p = MetricsPoller(
        lambda: list(eps), InflightStore(), _counting_fetch(itertools.count()), interval_ms=10
    )
    p.start()
    try:
        assert _wait_for(lambda: "routing_stats" in eps[0].attributes)
        late = Endpoint(name="late", attributes={})
        eps.append(late)
        assert _wait_for(lambda: "routing_stats" in late.attributes)
    finally:
        p.stop()
