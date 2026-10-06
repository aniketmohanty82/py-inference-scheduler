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

from integration.verl.admission import Admission
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.framework import Endpoint, LLMRequest


def _admission() -> Admission:
    config = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "p": {
                "flow_control": {"type": "kv_saturation", "default_osl": 0},
                "scorers": [{"type": "least_queue", "weight": 1.0}],
                "picker": {"type": "max_score"},
            }
        },
    }
    scheduler = Scheduler.new_with_config(SchedulerConfig.from_dict(config))
    return Admission(scheduler, scheduler.get_flow_control_plugins()[0])


def _engine(name: str, capacity: int, queue_len: int = 0) -> Endpoint:
    return Endpoint(name=name, attributes={"kv_cache_size": capacity, "queue_len": queue_len})


def _turn(trajectory: str, prompt_tokens: int) -> LLMRequest:
    return LLMRequest(request_id=trajectory, body=list(range(prompt_tokens)))


def test_scorers_pick_among_engines_with_room():
    admission = _admission()
    busy, idle, full = _engine("busy", 100, 5), _engine("idle", 100), _engine("full", 100)
    admission.place(_turn("filler", 100), [full])

    winner = admission.place(_turn("t1", 50), [busy, idle, full])

    assert winner is idle


def test_nothing_is_reserved_when_no_engine_has_room():
    admission = _admission()
    engine = _engine("e1", 100)
    admission.place(_turn("a", 60), [engine])

    assert admission.place(_turn("b", 60), [engine]) is None
    admission.release("a", "e1", None)
    assert admission.place(_turn("b", 60), [engine]) is engine


async def test_waiting_requests_go_longest_prompt_first_as_space_frees():
    admission = _admission()
    engine = _engine("e1", 1000)
    admission.place(_turn("a", 900), [engine])
    shorter = admission.wait(_turn("b", 400))
    longer = admission.wait(_turn("c", 700))

    admission.release("a", "e1", None)
    assert admission.retry([engine]) == [engine]
    assert longer.done()
    assert not shorter.done()
    assert admission.waiting == 1

    admission.release("c", "e1", None)
    assert admission.retry([engine]) == [engine]
    assert shorter.result() is engine


def test_a_trajectory_returns_to_its_engine_over_a_less_loaded_one():
    admission = _admission()
    home, other = _engine("home", 100, 0), _engine("other", 100, 0)
    assert admission.place(_turn("t1", 10), [home]) is home
    admission.release("t1", "home", 5)
    home.attributes["queue_len"] = 3

    assert admission.place(_turn("t1", 30), [home, other]) is home


async def test_a_cancelled_waiter_is_dropped():
    admission = _admission()
    engine = _engine("e1", 100)
    admission.place(_turn("a", 100), [engine])
    waiter = admission.wait(_turn("b", 10))
    waiter.cancel()

    admission.release("a", "e1", None)
    assert admission.retry([engine]) == []
    assert admission.waiting == 0
    await asyncio.sleep(0)
