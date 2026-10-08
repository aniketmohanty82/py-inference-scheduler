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

import queue
import re
from types import SimpleNamespace

import pytest

from py_inference_scheduler.datalayer.connectors.mooncake import save_diagnostics as diag


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(
        diag, "time", SimpleNamespace(monotonic=lambda: now["t"], time=lambda: 1.0e9)
    )
    return now


class Pool:
    def __init__(self, free: int, total: int) -> None:
        self.free = free
        self.num_gpu_blocks = total

    def get_num_free_blocks(self) -> int:
        return self.free


class SendThread:
    def __init__(self, clock, seconds: float = 0.0) -> None:
        self.clock = clock
        self.seconds = seconds
        self.handled: list[str] = []
        self.stored_requests = {"a": {4, 6}, "b": {5}}
        self.request_queue: queue.Queue[str] = queue.Queue()
        self._completed_saves = {4: 1}
        self.tp_rank = 2

    def _handle_request(self, req_meta) -> None:
        self.clock["t"] += self.seconds
        self.handled.append(req_meta)

    def is_alive(self) -> bool:
        return True


def scheduler(pinned, next_id=0, free=0, total=10, requests=None):
    return SimpleNamespace(
        _pinned_saves=pinned,
        _next_store_job_id=next_id,
        _gpu_block_pool=Pool(free, total),
        _unfinished_requests=requests or {},
    )


def step(new=(), cached=(), resumed=(), preempted=()):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new],
        scheduled_cached_reqs=SimpleNamespace(req_ids=list(cached), resumed_req_ids=set(resumed)),
        preempted_req_ids=set(preempted),
    )


def metadata(*jobs):
    return SimpleNamespace(requests=[SimpleNamespace(req_id=r, store_job_id=j) for r, j in jobs])


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", line))


def test_track_save_jobs_names_what_made_each_request_save(capsys, clock):
    pinned = {1: ([10, 11], 4), 2: ([11, 12], 4), 3: ([20], 4), 4: ([30], 4)}
    requests = {
        "a": (SimpleNamespace(num_preemptions=0), []),
        "b": (SimpleNamespace(num_preemptions=2), []),
    }
    born: dict[int, float] = {}
    diag.track_save_jobs(
        scheduler(pinned, requests=requests),
        # "b" is how the v2 model runner lists a resumed request, "d" how the v1 runner does.
        step(new=["a", "b"], cached=["c", "d"], resumed=["d"], preempted=["x", "y"]),
        metadata(("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", None)),
        born,
    )
    line = fields(capsys.readouterr().out)
    assert (line["emitted"], line["new"], line["resumed"], line["running"]) == ("4", "1", "2", "1")
    assert (line["blocks"], line["preempted"], line["scheduled"]) == ("5", "2", "4")
    assert born == {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0}


def test_track_save_jobs_is_quiet_without_saves_or_preemptions(capsys, clock):
    born = {1: 50.0, 2: 60.0}
    diag.track_save_jobs(scheduler({2: ([5], 1)}), step(cached=["a"]), metadata(("a", 2)), born)
    assert capsys.readouterr().out == ""
    # Job 1 was released; job 2 keeps its first stamp rather than counting as new again.
    assert born == {2: 60.0}


def test_print_pins_shows_turnover_age_and_lagging_ranks(capsys, clock):
    pinned = {7: ([1, 2, 3], 4), 9: ([3, 4], 1)}
    diag.print_pins(scheduler(pinned, next_id=12, free=5), {7: 97.0, 9: 99.5})
    line = fields(capsys.readouterr().out)
    assert (line["jobs"], line["blocks"], line["free"], line["total"]) == ("2", "4", "5", "10")
    assert (line["ids"], line["next"], line["oldest_s"]) == ("7-9", "12", "3.0")
    assert line["waiting_on"] == "1:1,4:1"


def test_print_pins_reports_instead_of_raising(capsys):
    class BrokenPool:
        def get_num_free_blocks(self) -> int:
            raise RuntimeError("pool gone")

    diag.print_pins(SimpleNamespace(_pinned_saves={}, _gpu_block_pool=BrokenPool()), {})
    assert capsys.readouterr().out.startswith("PINS unavailable: RuntimeError('pool gone')")


def test_save_timer_feeds_sendq_once_per_thread(capsys, clock):
    thread = SendThread(clock, seconds=1.5)
    assert diag.install_save_timer(thread)
    assert not diag.install_save_timer(thread)
    thread._handle_request("job")
    assert thread.handled == ["job"]
    diag.print_send_state(thread)
    line = fields(capsys.readouterr().out)
    assert (line["rank"], line["live"], line["unreported"]) == ("2", "3", "1")
    assert line["live_ids"] == "4-6"
    assert (line["finished"], line["slowest_s"], line["busy_s"]) == ("1", "1.50", "0.0")
    diag.print_send_state(thread)
    assert fields(capsys.readouterr().out)["slowest_s"] == "0.00"


def test_sendq_shows_how_long_the_running_job_has_run(capsys, clock):
    thread = SendThread(clock)
    diag.install_save_timer(thread)
    thread.rls_save_stats["started"] = clock["t"] - 4.0
    diag.print_send_state(thread)
    assert fields(capsys.readouterr().out)["busy_s"] == "4.0"


def test_save_timer_counts_a_job_that_raises(clock):
    class FailingThread(SendThread):
        def _handle_request(self, req_meta) -> None:
            raise RuntimeError("put failed")

    thread = FailingThread(clock)
    diag.install_save_timer(thread)
    with pytest.raises(RuntimeError, match="put failed"):
        thread._handle_request("job")
    assert (thread.rls_save_stats["finished"], thread.rls_save_stats["started"]) == (1, 0.0)


def test_print_send_state_without_a_send_thread_prints_nothing(capsys):
    diag.print_send_state(None)
    assert capsys.readouterr().out == ""
