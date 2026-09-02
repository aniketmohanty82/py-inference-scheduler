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
import contextlib
import logging
from collections import deque
from typing import Awaitable, Callable, Sequence

from py_inference_scheduler.framework import Endpoint, FlowControlPlugin, LLMRequest

logger = logging.getLogger(__name__)

GetEndpoints = Callable[[], Awaitable[Sequence[Endpoint]]]
PluginsProvider = Callable[[], Sequence[FlowControlPlugin]]

_Waiter = tuple[LLMRequest, "asyncio.Future[Sequence[Endpoint]]"]


class FlowControlClosedError(RuntimeError):
    """Raised to parked admit() callers when the manager shuts down."""


class FlowControlManager:
    """Metric-driven admission control shared by every integration.

    admit() gates a request on the flow-control plugins; when the whole fleet
    is inadmissible the request parks in a FIFO queue while a watcher task
    polls get_endpoints() for fresh routing stats and re-admits waiters as
    metrics allow. Request completions never drive wake-ups: release() only
    fans out to plugin bookkeeping.

    The watcher drains with an AIMD window so a large capacity release is not
    throttled to one admission per tick: each open tick admits up to `window`
    waiters then grows the window additively; a tick that finds the gate shut
    again shrinks it multiplicatively. The window is seeded from the number of
    admissible endpoints when a parked episode first reopens.

    Single event loop per manager; fresh requests can pass admit() directly
    while others are parked (no strict FIFO across the two paths).
    """

    def __init__(  # noqa: PLR0913
        self,
        plugins_provider: PluginsProvider,
        get_endpoints: GetEndpoints,
        *,
        poll_interval_s: float = 0.1,
        aimd_increase: int = 1,
        aimd_decay: float = 0.5,
        max_admissions_per_tick: int = 0,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive.")
        if aimd_increase < 1:
            raise ValueError("aimd_increase must be at least 1.")
        if not 0.0 < aimd_decay < 1.0:
            raise ValueError("aimd_decay must be in (0, 1).")
        if max_admissions_per_tick < 0:
            raise ValueError("max_admissions_per_tick must be >= 0 (0 disables the cap).")
        self._plugins = plugins_provider
        self._get_endpoints = get_endpoints
        self._poll_interval_s = poll_interval_s
        self._aimd_increase = aimd_increase
        self._aimd_decay = aimd_decay
        self._max_admissions_per_tick = max_admissions_per_tick
        self._waiters: deque[_Waiter] = deque()
        self._watcher: asyncio.Task[None] | None = None
        self._closed = False

    def has_plugins(self) -> bool:
        return bool(self._plugins())

    def queue_depth(self) -> int:
        return sum(1 for _, fut in self._waiters if not fut.done())

    async def admit(
        self, request: LLMRequest, endpoints: Sequence[Endpoint]
    ) -> Sequence[Endpoint]:
        """Endpoints the request may route to; parks until metrics allow any."""
        if self._closed:
            raise FlowControlClosedError("flow control manager is closed")
        plugins = list(self._plugins())
        # Without plugins or endpoints there is nothing to wait for.
        if not plugins or not endpoints:
            return endpoints
        allowed = self._run_gate(plugins, request, endpoints)
        if allowed:
            return allowed

        fut: asyncio.Future[Sequence[Endpoint]] = asyncio.get_running_loop().create_future()
        self._waiters.append((request, fut))
        self._ensure_watcher()
        try:
            return await fut
        except asyncio.CancelledError:
            # Client went away: give up the queue slot.
            with contextlib.suppress(ValueError):
                self._waiters.remove((request, fut))
            raise

    def commit(self, request: LLMRequest, selected: Endpoint) -> None:
        for plugin in self._plugins():
            plugin.reserve(request, selected)

    def release(self, request: LLMRequest, endpoint_name: str) -> None:
        """Plugin bookkeeping only: re-admission is the watcher's job."""
        for plugin in self._plugins():
            plugin.release(request, endpoint_name)

    async def close(self) -> None:
        """Stop the watcher and fail every parked request."""
        self._closed = True
        watcher = self._watcher
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        while self._waiters:
            _, fut = self._waiters.popleft()
            if not fut.done():
                fut.set_exception(FlowControlClosedError("flow control manager is closed"))

    @staticmethod
    def _run_gate(
        plugins: Sequence[FlowControlPlugin],
        request: LLMRequest,
        endpoints: Sequence[Endpoint],
    ) -> Sequence[Endpoint]:
        allowed = endpoints
        for plugin in plugins:
            allowed = plugin.get_allowed_candidates(request, allowed)
        return allowed

    def _ensure_watcher(self) -> None:
        if self._watcher is None or self._watcher.done():
            self._watcher = asyncio.get_running_loop().create_task(self._watch())

    async def _watch(self) -> None:
        window = 0  # 0 = unseeded; set from admissible-endpoint count at gate-open
        while self._waiters and not self._closed:
            await asyncio.sleep(self._poll_interval_s)
            try:
                endpoints = await self._get_endpoints()
            except Exception:
                logger.exception("flow control endpoint refresh failed; retrying")
                continue
            plugins = list(self._plugins())
            if not plugins:
                # Hot reload removed flow control: let everyone through.
                self._drain_all_through(endpoints)
                return
            window = self._drain_tick(plugins, endpoints, window)

    def _drain_all_through(self, endpoints: Sequence[Endpoint]) -> None:
        while self._waiters:
            _, fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(endpoints)

    def _drain_tick(
        self,
        plugins: Sequence[FlowControlPlugin],
        endpoints: Sequence[Endpoint],
        window: int,
    ) -> int:
        """Admit up to `window` waiters against this tick's stats; return new window."""
        admitted = 0
        gate_shut = not endpoints
        while self._waiters and not gate_shut:
            request, fut = self._waiters[0]
            if fut.done():
                self._waiters.popleft()
                continue
            allowed = self._run_gate(plugins, request, endpoints)
            if not allowed:
                gate_shut = True
                break
            if window == 0:
                window = len(allowed)
            if self._max_admissions_per_tick:
                window = min(window, self._max_admissions_per_tick)
            if admitted >= window:
                break
            self._waiters.popleft()
            fut.set_result(allowed)
            admitted += 1

        if gate_shut:
            return max(1, int(window * self._aimd_decay)) if window else 0
        if not self._waiters:
            return 0  # queue drained; reseed on the next parked episode
        return window + self._aimd_increase
