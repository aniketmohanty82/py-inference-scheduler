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


class TestAimdWindow:
    """Drive _drain_tick directly: deterministic AIMD arithmetic, no timers."""

    def setup_method(self) -> None:
        self.plugin = GatePlugin(open_=True)

        async def get_endpoints() -> Sequence[Endpoint]:
            return []

        self.manager = FlowControlManager(
            lambda: [self.plugin], get_endpoints, poll_interval_s=0.02
        )

    def _park(self, n: int) -> list[asyncio.Future]:
        loop = asyncio.get_event_loop()
        futs = []
        for i in range(n):
            fut: asyncio.Future = loop.create_future()
            self.manager._waiters.append((_req(f"r{i}"), fut))
            futs.append(fut)
        return futs

    def _admitted(self, futs: list[asyncio.Future]) -> int:
        return sum(1 for f in futs if f.done())

    async def test_window_seeds_from_admissible_count_and_ramps(self):
        futs = self._park(9)
        window = self.manager._drain_tick([self.plugin], _eps(2), 0)
        assert self._admitted(futs) == 2  # seeded from 2 admissible endpoints
        assert window == 3  # +1 additive increase
        window = self.manager._drain_tick([self.plugin], _eps(2), window)
        assert self._admitted(futs) == 5
        assert window == 4

    async def test_window_halves_when_gate_shuts_mid_episode(self):
        self._park(3)
        self.plugin.open = False
        assert self.manager._drain_tick([self.plugin], _eps(2), 8) == 4
        assert self.manager._drain_tick([self.plugin], _eps(2), 1) == 1  # floor at 1

    async def test_window_resets_after_queue_drains(self):
        futs = self._park(2)
        window = self.manager._drain_tick([self.plugin], _eps(4), 0)
        assert self._admitted(futs) == 2
        assert window == 0  # reseeded on the next parked episode

    async def test_max_admissions_per_tick_caps_the_window(self):
        self.manager._max_admissions_per_tick = 1
        futs = self._park(5)
        window = self.manager._drain_tick([self.plugin], _eps(4), 0)
        assert self._admitted(futs) == 1
        assert window == 2  # capped to 1 this tick, still grows additively

    async def test_no_endpoints_counts_as_shut_gate(self):
        self._park(1)
        assert self.manager._drain_tick([self.plugin], [], 6) == 3
