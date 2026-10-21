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
from types import SimpleNamespace

import pytest

from integration.verl.fleet import admit
from py_inference_scheduler.framework import LLMRequest


def _fleet() -> SimpleNamespace:
    admitting: asyncio.Future[tuple[str, object]] = asyncio.get_running_loop().create_future()
    finished: list[tuple[object, ...]] = []
    return SimpleNamespace(
        admitting=admitting,
        finished=finished,
        admit=SimpleNamespace(remote=lambda request: admitting),
        finish=SimpleNamespace(remote=lambda *args: finished.append(args)),
    )


async def _cancelled_caller(fleet: SimpleNamespace) -> None:
    caller = asyncio.create_task(admit(fleet, LLMRequest(request_id="t1")))
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller


async def test_returns_the_engine_the_fleet_admitted_to():
    fleet = _fleet()
    fleet.admitting.set_result(("e1", "handle"))

    assert await admit(fleet, LLMRequest(request_id="t1")) == ("e1", "handle")
    assert fleet.finished == []


async def test_hands_back_an_admission_made_after_the_caller_gave_up():
    fleet = _fleet()
    await _cancelled_caller(fleet)

    fleet.admitting.set_result(("e1", "handle"))
    await asyncio.sleep(0)

    assert fleet.finished == [("t1", "e1", None)]


async def test_hands_back_nothing_when_admission_failed():
    fleet = _fleet()
    await _cancelled_caller(fleet)

    fleet.admitting.set_exception(RuntimeError("no capacity"))
    await asyncio.sleep(0)

    assert fleet.finished == []
