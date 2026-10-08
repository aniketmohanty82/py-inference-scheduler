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

import asyncio

import pytest

from integration.verl.admission import Admission
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.framework import Endpoint, LLMRequest


def _admission(
    engines: list[Endpoint], scorer: str = "least_queue"
) -> tuple[Admission, InflightStore]:
    config = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "p": {
                "flow_control": {"type": "kv_saturation", "default_osl": 0},
                "scorers": [{"type": scorer, "weight": 1.0}],
                "picker": {"type": "max_score"},
            }
        },
    }
    scheduler = Scheduler.new_with_config(SchedulerConfig.from_dict(config))
    plugin = scheduler.get_flow_control_plugins()[0]
    counts = InflightStore()
    return Admission(scheduler, plugin, counts, lambda: engines, retry_interval_s=0.01), counts


def _engine(name: str, capacity: int | None = None) -> Endpoint:
    attributes: dict[str, object] = {"routing_stats": {}}
    if capacity is not None:
        attributes["kv_cache_size"] = capacity
    return Endpoint(name=name, attributes=attributes)


def _turn(trajectory: str, prompt_tokens: int) -> LLMRequest:
    return LLMRequest(request_id=trajectory, body=list(range(prompt_tokens)))


async def _queued(admission: Admission, request: LLMRequest) -> asyncio.Task[Endpoint]:
    task = asyncio.create_task(admission.admit(request))
    await asyncio.sleep(0)
    assert not task.done()
    return task


async def test_scorers_pick_among_engines_with_room():
    full, busy, idle = _engine("full", 100), _engine("busy", 100), _engine("idle", 100)
    admission, counts = _admission([full, busy, idle])
    for name in ("busy", "busy", "busy", "idle", "idle"):
        counts.increment(name)
    assert await admission.admit(_turn("filler", 60)) is full

    # full still has the shortest queue but only 40 tokens free.
    assert await admission.admit(_turn("t1", 50)) is idle
    assert counts.get("idle") == 3


async def test_a_request_waits_until_space_frees():
    engine = _engine("e1", 100)
    admission, counts = _admission([engine])
    await admission.admit(_turn("a", 60))
    waiter = await _queued(admission, _turn("b", 60))

    admission.finish("a", "e1", None)

    assert await waiter is engine
    assert counts.get("e1") == 1


async def test_waiting_requests_go_longest_prompt_first():
    engine = _engine("e1", 1000)
    admission, _ = _admission([engine])
    await admission.admit(_turn("a", 900))
    shorter = await _queued(admission, _turn("b", 400))
    longer = await _queued(admission, _turn("c", 700))

    admission.finish("a", "e1", None)
    assert await longer is engine
    assert not shorter.done()

    admission.finish("c", "e1", None)
    assert await shorter is engine


async def test_a_trajectory_returns_to_its_engine_over_a_less_loaded_one():
    home, other = _engine("home", 100), _engine("other", 100)
    admission, counts = _admission([home, other])
    counts.increment("other")
    assert await admission.admit(_turn("t1", 10)) is home
    admission.finish("t1", "home", 5)
    for _ in range(3):
        counts.increment("home")

    assert await admission.admit(_turn("t1", 30)) is home


async def test_fill_first_fills_one_engine_before_opening_the_next():
    engines = [_engine("a", 100), _engine("b", 100), _engine("c", 100)]
    admission, _ = _admission(engines, scorer="fill_first")
    placed = [(await admission.admit(_turn(f"t{i}", 25))).name for i in range(12)]
    assert placed == ["a"] * 4 + ["b"] * 4 + ["c"] * 4


async def test_fill_first_still_returns_a_trajectory_to_its_engine():
    home, fuller = _engine("home", 100), _engine("fuller", 100)
    admission, counts = _admission([home, fuller], scorer="fill_first")
    assert await admission.admit(_turn("t1", 10)) is home
    admission.finish("t1", "home", 5)
    for _ in range(3):
        counts.increment("fuller")

    assert await admission.admit(_turn("t1", 30)) is home


async def test_a_cancelled_waiter_gives_up_its_place():
    engine = _engine("e1", 100)
    admission, counts = _admission([engine])
    await admission.admit(_turn("a", 100))
    waiter = await _queued(admission, _turn("b", 10))
    waiter.cancel()

    admission.finish("a", "e1", None)

    assert counts.get("e1") == 0
    assert await admission.admit(_turn("c", 100)) is engine


async def test_capacity_from_the_first_poll_admits_waiting_requests():
    engine = _engine("e1")
    admission, _ = _admission([engine])
    waiter = await _queued(admission, _turn("t1", 10))

    engine.attributes["kv_cache_size"] = 100

    assert await asyncio.wait_for(waiter, timeout=1) is engine


async def test_a_waiting_caller_sees_why_placement_failed():
    engine = _engine("e1")
    admission, _ = _admission([engine])
    waiter = await _queued(admission, _turn("t1", 10))

    engine.attributes["routing_stats"] = {"num_running_reqs": 0}

    with pytest.raises(RuntimeError, match="KV capacity"):
        await asyncio.wait_for(waiter, timeout=1)
