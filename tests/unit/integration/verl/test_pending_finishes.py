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

from integration.verl.fleet import PendingFinishes


def _fleet(*results: asyncio.Future[None]) -> SimpleNamespace:
    """A fleet whose finish calls return the given futures in order."""
    calls: list[tuple[object, ...]] = []
    pending = iter(results)

    def finish(*args: object) -> asyncio.Future[None]:
        calls.append(args)
        return next(pending)

    return SimpleNamespace(calls=calls, finish=SimpleNamespace(remote=finish))


def _future() -> asyncio.Future[None]:
    return asyncio.get_running_loop().create_future()


async def test_next_turn_waits_until_the_last_finish_has_run():
    done = _future()
    fleet = _fleet(done)
    finishes = PendingFinishes(fleet)
    finishes.start("t1", "e1", 32)
    next_turn = asyncio.create_task(finishes.wait("t1"))
    await asyncio.sleep(0)

    assert not next_turn.done()
    done.set_result(None)
    await next_turn
    assert fleet.calls == [("t1", "e1", 32)]


async def test_does_not_wait_without_a_running_finish():
    finishes = PendingFinishes(_fleet())

    await asyncio.wait_for(finishes.wait("t1"), timeout=1)


async def test_forgets_a_finish_once_it_has_run():
    done = _future()
    finishes = PendingFinishes(_fleet(done))
    finishes.start("t1", "e1", 32)
    done.set_result(None)
    await finishes.wait("t1")

    assert finishes._running == {}


async def test_an_older_finish_does_not_drop_a_newer_one():
    older, newer = _future(), _future()
    finishes = PendingFinishes(_fleet(older, newer))
    finishes.start("t1", "e1", 32)
    finishes.start("t1", "e2", 16)
    older.set_result(None)
    await asyncio.sleep(0)
    next_turn = asyncio.create_task(finishes.wait("t1"))
    await asyncio.sleep(0)

    assert not next_turn.done()
    newer.set_result(None)
    await next_turn
