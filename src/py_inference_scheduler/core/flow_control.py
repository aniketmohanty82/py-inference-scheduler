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

    The watcher drains one admission at a time, each re-gated on the latest
    polled stats, at an AIMD-controlled rate: `window` admissions per poll
    interval, grown additively while the gate stays open and shrunk
    multiplicatively when it shuts. The window is seeded from the number of
    admissible endpoints when a parked episode first reopens, so a large
    capacity release ramps quickly without batching admissions onto a single
    metrics snapshot.

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
        self._window = 0  # current AIMD drain rate (admissions per poll interval)
        self._streak = 0  # consecutive admissions since the last window adjustment

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
        """Admit parked waiters one at a time, each against freshly polled stats.

        Admissions are paced, never batched: the AIMD window is a RATE (window
        admissions per poll interval, so the inter-admission delay is
        poll_interval_s / window). Every admission re-runs the gate on the
        latest endpoint stats, so the polls landing between admissions carry
        the effect of the previous one -- a reopened gate drains without
        herding a batch of waiters onto one snapshot's best replica.
        """
        self._window = 0  # 0 = unseeded; set from admissible-endpoint count at gate-open
        self._streak = 0
        while self._waiters and not self._closed:
            await asyncio.sleep(self._delay_s(self._window))
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
            while self._waiters and self._waiters[0][1].done():
                self._waiters.popleft()
            if not self._waiters:
                return
            request, fut = self._waiters[0]
            allowed = self._run_gate(plugins, request, endpoints) if endpoints else []
            if not allowed:
                self._window = max(1, int(self._window * self._aimd_decay)) if self._window else 0
                self._streak = 0
                continue
            if self._window == 0:
                self._window = self._capped(len(allowed))
            self._waiters.popleft()
            fut.set_result(allowed)
            self._streak += 1
            if self._streak >= self._window:
                self._window = self._capped(self._window + self._aimd_increase)
                self._streak = 0

    def _delay_s(self, window: int) -> float:
        return self._poll_interval_s / window if window > 1 else self._poll_interval_s

    def _capped(self, window: int) -> int:
        if self._max_admissions_per_tick:
            return min(window, self._max_admissions_per_tick)
        return window

    def _drain_all_through(self, endpoints: Sequence[Endpoint]) -> None:
        while self._waiters:
            _, fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(endpoints)
