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
from typing import Callable, Sequence

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.framework import Endpoint, FlowControlPlugin, LLMRequest
from py_inference_scheduler.framework.helpers import prefill_tokens


class Admission:
    """Admits requests with a flow-control plugin, in the order each worker ranked the engines.

    - Reserves and counts each admission in the same step, so later decisions see it.
    - Holds requests that fit nowhere and retries them longest prompt first.
    """

    def __init__(
        self,
        plugin: FlowControlPlugin,
        counts: InflightStore,
        endpoints: Callable[[], Sequence[Endpoint]],
        retry_interval_s: float,
    ) -> None:
        self._plugin = plugin
        self._counts = counts
        self._endpoints = endpoints
        self._retry_interval_s = retry_interval_s
        self._waiting: list[tuple[LLMRequest, Sequence[str], asyncio.Future[Endpoint]]] = []
        self._retrying: asyncio.Task[None] | None = None

    async def admit(self, request: LLMRequest, ranking: Sequence[str]) -> Endpoint:
        """Admit the request to its highest-ranked engine with room, waiting until one has it."""
        winner = self._try_admit(request, ranking)
        if winner is not None:
            return winner
        future: asyncio.Future[Endpoint] = asyncio.get_running_loop().create_future()
        self._waiting.append((request, ranking, future))
        if self._retrying is None or self._retrying.done():
            self._retrying = asyncio.get_running_loop().create_task(self._retry_until_empty())
        return await future

    def finish(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        """Release the request's reservation and retry the requests waiting for one."""
        self._counts.decrement(endpoint_name)
        self._plugin.release(LLMRequest(request_id=request_id), endpoint_name, output_tokens)
        self._retry()

    def _try_admit(self, request: LLMRequest, ranking: Sequence[str]) -> Endpoint | None:
        allowed = {
            ep.name: ep for ep in self._plugin.get_allowed_candidates(request, self._endpoints())
        }
        if not allowed:
            return None
        # An engine the worker has not discovered yet is not ranked, so it comes last.
        name = next((n for n in ranking if n in allowed), next(iter(allowed)))
        winner = allowed[name]
        self._plugin.reserve(request, winner)
        self._counts.increment(winner.name)
        return winner

    def _retry(self) -> None:
        # Longest prompt first: late turns are the step's critical path and the costliest to stall.
        waiting = sorted(
            (w for w in self._waiting if not w[2].done()),
            key=lambda w: prefill_tokens(w[0].body),
            reverse=True,
        )
        self._waiting = []
        for request, ranking, future in waiting:
            try:
                winner = self._try_admit(request, ranking)
            except Exception as e:  # noqa: BLE001 - the waiting caller must see the failure
                future.set_exception(e)
                continue
            if winner is None:
                self._waiting.append((request, ranking, future))
            else:
                future.set_result(winner)

    async def _retry_until_empty(self) -> None:
        # Also covers capacity that arrives with the first metrics poll, when no finish is coming.
        while self._waiting:
            await asyncio.sleep(self._retry_interval_s)
            self._retry()
