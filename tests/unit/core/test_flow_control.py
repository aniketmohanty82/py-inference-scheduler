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
from typing import Sequence

import pytest

from py_inference_scheduler.core.flow_control import (
    FlowControlClosedError,
    FlowControlManager,
)
from py_inference_scheduler.framework import Endpoint, LLMRequest


class GatePlugin:
    """Flow-control stub whose gate opens and closes via a flag."""

    def __init__(self, *, open_: bool = False) -> None:
        self.open = open_
        self.reserved: list[str] = []
        self.released: list[str] = []

    def get_allowed_candidates(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[Endpoint]:
        return list(candidates) if self.open else []

    def reserve(self, request: LLMRequest, selected: Endpoint) -> None:
        self.reserved.append(selected.name)

    def release(self, request: LLMRequest, endpoint_name: str) -> None:
        self.released.append(endpoint_name)


def _eps(n: int = 2) -> list[Endpoint]:
    return [Endpoint(name=f"ep{i}") for i in range(n)]


def _req(rid: str = "r") -> LLMRequest:
    return LLMRequest(request_id=rid)


def _manager(
    plugin: GatePlugin | None, endpoints: list[Endpoint], **kwargs: float
) -> tuple[FlowControlManager, list[int]]:
    refreshes: list[int] = []

    async def get_endpoints() -> Sequence[Endpoint]:
        refreshes.append(1)
        return endpoints

    plugins = [plugin] if plugin else []
    manager = FlowControlManager(
        lambda: list(plugins), get_endpoints, poll_interval_s=0.02, **kwargs
    )
    return manager, refreshes


async def test_no_plugins_passes_through_without_watcher():
    manager, refreshes = _manager(None, _eps())
    eps = _eps()
    assert await manager.admit(_req(), eps) == eps
    assert manager._watcher is None
    assert refreshes == []


async def test_open_gate_fast_path_never_parks():
    manager, refreshes = _manager(GatePlugin(open_=True), _eps())
    allowed = await manager.admit(_req(), _eps(3))
    assert [ep.name for ep in allowed] == ["ep0", "ep1", "ep2"]
    assert manager._watcher is None
    assert refreshes == []


async def test_empty_endpoints_returns_empty_not_parked():
    manager, _ = _manager(GatePlugin(), _eps())
    assert await manager.admit(_req(), []) == []
    assert manager.queue_depth() == 0


async def test_parked_request_admitted_when_metrics_open_the_gate():
    plugin = GatePlugin()
    manager, refreshes = _manager(plugin, _eps())
    task = asyncio.ensure_future(manager.admit(_req(), _eps()))
    await asyncio.sleep(0.06)
    assert not task.done()
    assert manager.queue_depth() == 1
    assert refreshes  # the watcher is polling while the request is parked

    plugin.open = True
    allowed = await asyncio.wait_for(task, timeout=1.0)
    assert [ep.name for ep in allowed] == ["ep0", "ep1"]
    assert manager.queue_depth() == 0


async def test_release_does_not_wake_waiters():
    """Completions are plugin bookkeeping only; the watcher owns re-admission."""
    plugin = GatePlugin()
    manager, _ = _manager(plugin, _eps())
    task = asyncio.ensure_future(manager.admit(_req(), _eps()))
    await asyncio.sleep(0.03)
    manager.release(_req("done"), "ep0")
    await asyncio.sleep(0.06)
    assert not task.done()
    assert plugin.released == ["ep0"]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_fifo_order_preserved():
    plugin = GatePlugin()
    manager, _ = _manager(plugin, _eps(1), max_admissions_per_tick=1)
    order: list[str] = []

    async def admit(rid: str) -> None:
        await manager.admit(_req(rid), _eps())
        order.append(rid)

    tasks = [asyncio.ensure_future(admit(rid)) for rid in ("a", "b", "c")]
    await asyncio.sleep(0.03)
    plugin.open = True
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
    assert order == ["a", "b", "c"]


async def test_cancelled_waiter_leaves_the_queue():
    plugin = GatePlugin()
    manager, _ = _manager(plugin, _eps())
    task = asyncio.ensure_future(manager.admit(_req(), _eps()))
    await asyncio.sleep(0.03)
    assert manager.queue_depth() == 1
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert manager.queue_depth() == 0


async def test_close_fails_parked_requests_and_stops_watcher():
    plugin = GatePlugin()
    manager, _ = _manager(plugin, _eps())
    task = asyncio.ensure_future(manager.admit(_req(), _eps()))
    await asyncio.sleep(0.03)
    await manager.close()
    with pytest.raises(FlowControlClosedError):
        await task
    assert manager._watcher is None or manager._watcher.done()
    with pytest.raises(FlowControlClosedError):
        await manager.admit(_req(), _eps())


async def test_hot_reload_removing_plugins_drains_everyone_through():
    plugin = GatePlugin()
    plugins: list[GatePlugin] = [plugin]

    async def get_endpoints() -> Sequence[Endpoint]:
        return _eps()

    manager = FlowControlManager(lambda: list(plugins), get_endpoints, poll_interval_s=0.02)
    task = asyncio.ensure_future(manager.admit(_req(), _eps()))
    await asyncio.sleep(0.03)
    plugins.clear()
    allowed = await asyncio.wait_for(task, timeout=1.0)
    assert [ep.name for ep in allowed] == ["ep0", "ep1"]


async def test_watcher_exits_once_queue_drains():
    plugin = GatePlugin()
    manager, _ = _manager(plugin, _eps())
    task = asyncio.ensure_future(manager.admit(_req(), _eps()))
    await asyncio.sleep(0.03)
    plugin.open = True
    await asyncio.wait_for(task, timeout=1.0)
    await asyncio.sleep(0.05)
    assert manager._watcher is not None
    assert manager._watcher.done()


async def test_commit_and_release_fan_out_to_plugins():
    plugin = GatePlugin(open_=True)
    manager, _ = _manager(plugin, _eps())
    manager.commit(_req(), Endpoint(name="ep1"))
    manager.release(_req(), "ep1")
    assert plugin.reserved == ["ep1"]
    assert plugin.released == ["ep1"]


def test_invalid_config_rejected():
    async def get_endpoints() -> Sequence[Endpoint]:
        return []

    with pytest.raises(ValueError, match="poll_interval_s"):
        FlowControlManager(list, get_endpoints, poll_interval_s=0)
    with pytest.raises(ValueError, match="aimd_increase"):
        FlowControlManager(list, get_endpoints, aimd_increase=0)
    with pytest.raises(ValueError, match="aimd_decay"):
        FlowControlManager(list, get_endpoints, aimd_decay=1.0)
    with pytest.raises(ValueError, match="max_admissions_per_tick"):
        FlowControlManager(list, get_endpoints, max_admissions_per_tick=-1)


class BudgetGatePlugin(GatePlugin):
    """Gate that opens for a set number of evaluations, then shuts again."""

    def __init__(self) -> None:
        super().__init__()
        self.opens_remaining = 0

    def get_allowed_candidates(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[Endpoint]:
        if self.opens_remaining > 0:
            self.opens_remaining -= 1
            return list(candidates)
        return []


async def _drain(manager: FlowControlManager, tasks: list[asyncio.Task], timeout=3.0) -> None:
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)


async def test_each_admission_is_regated_on_fresh_stats():
    """The polls just before/after one admission inform the next one."""
    plugin = BudgetGatePlugin()
    manager, refreshes = _manager(plugin, _eps(1))
    tasks = [asyncio.ensure_future(manager.admit(_req(rid), _eps())) for rid in ("a", "b")]
    await asyncio.sleep(0.05)
    assert manager.queue_depth() == 2

    plugin.opens_remaining = 1  # capacity for exactly one gate evaluation
    await asyncio.sleep(0.08)
    assert manager.queue_depth() == 1  # one admitted, second re-gated and re-parked
    refreshes_after_first = len(refreshes)

    plugin.opens_remaining = 1
    await _drain(manager, tasks)
    # The second admission required its own endpoint refresh, not a shared snapshot.
    assert len(refreshes) > refreshes_after_first


async def test_window_seeds_from_admissible_count_and_ramps():
    plugin = GatePlugin()
    manager, _ = _manager(plugin, _eps(3))
    tasks = [asyncio.ensure_future(manager.admit(_req(f"r{i}"), _eps())) for i in range(5)]
    await asyncio.sleep(0.05)
    plugin.open = True
    await _drain(manager, tasks)
    # Seeded at 3 admissible endpoints; after a 3-admission streak it grew to 4.
    assert manager._window == 4
    assert manager._streak == 2


async def test_window_decays_while_gate_stays_shut():
    plugin = BudgetGatePlugin()
    manager, _ = _manager(plugin, _eps(4))
    tasks = [asyncio.ensure_future(manager.admit(_req(f"r{i}"), _eps())) for i in range(2)]
    await asyncio.sleep(0.05)
    plugin.opens_remaining = 1
    await asyncio.sleep(0.08)
    assert manager.queue_depth() == 1  # window seeded at 4 by the single admission
    for _ in range(40):
        if manager._window == 1:
            break
        await asyncio.sleep(0.02)
    assert manager._window == 1  # closed evaluations halved it down to the floor
    plugin.opens_remaining = 100
    await _drain(manager, tasks)


async def test_max_admissions_per_tick_caps_rate_and_paces_admissions():
    plugin = GatePlugin()
    manager, _ = _manager(plugin, _eps(4), max_admissions_per_tick=1)
    tasks = [asyncio.ensure_future(manager.admit(_req(f"r{i}"), _eps())) for i in range(3)]
    await asyncio.sleep(0.05)
    plugin.open = True
    t0 = asyncio.get_event_loop().time()
    await _drain(manager, tasks)
    elapsed = asyncio.get_event_loop().time() - t0
    assert manager._window == 1  # cap held despite ramp attempts
    # Rate 1 per 0.02s poll interval: 3 paced admissions cannot be instantaneous.
    assert elapsed >= 0.03
