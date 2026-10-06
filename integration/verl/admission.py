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
from typing import Sequence, cast

from py_inference_scheduler import Scheduler
from py_inference_scheduler.framework import Endpoint, FlowControlPlugin, LLMRequest


class Admission:
    """Places requests with a flow-control plugin and the profile's scorers.

    - Reserves in the same step as it places, so one instance never over-admits.
    - Holds requests that fit nowhere and retries them longest prompt first.
    """

    def __init__(self, scheduler: Scheduler, plugin: FlowControlPlugin) -> None:
        self._scheduler = scheduler
        self._plugin = plugin
        self._waiting: list[tuple[LLMRequest, asyncio.Future[Endpoint]]] = []

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    def place(self, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Endpoint | None:
        """Reserve the request on the scorers' pick among engines with room; None if none."""
        allowed = self._plugin.get_allowed_candidates(request, endpoints)
        if not allowed:
            return None
        picked = self._scheduler.run(request, allowed)
        winner = picked[0].endpoint if picked else allowed[0]
        self._plugin.reserve(request, winner)
        return winner

    def wait(self, request: LLMRequest) -> asyncio.Future[Endpoint]:
        """Queue a request that fit nowhere; the future resolves to its engine."""
        future: asyncio.Future[Endpoint] = asyncio.get_running_loop().create_future()
        self._waiting.append((request, future))
        return future

    def release(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        self._plugin.release(LLMRequest(request_id=request_id), endpoint_name, output_tokens)

    def retry(self, endpoints: Sequence[Endpoint]) -> list[Endpoint]:
        """Place waiting requests that now fit and return their engines."""
        # Longest prompt first: late turns are the step's critical path and the costliest to stall.
        waiting = sorted(
            (w for w in self._waiting if not w[1].done()),
            key=lambda w: len(cast("list[int]", w[0].body or [])),
            reverse=True,
        )
        self._waiting = []
        placed = []
        for request, future in waiting:
            winner = self.place(request, endpoints)
            if winner is None:
                self._waiting.append((request, future))
            else:
                future.set_result(winner)
                placed.append(winner)
        return placed
