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

from typing import Callable, Sequence

from py_inference_scheduler import Scheduler
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.framework import Endpoint, FlowControlPlugin, LLMRequest
from py_inference_scheduler.framework.helpers import prefill_tokens

# Receives a queued request's admission, or the error that stopped it.
Deliver = Callable[["Endpoint | Exception"], None]


class Admission:
    """Admits requests with a flow-control plugin and the profile's scorers.

    - Reserves and counts each admission in the same step, so later decisions see it.
    - Keeps requests that fit nowhere in one queue, retries them longest prompt first, and hands
      each admission to the callback its request queued with.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        plugin: FlowControlPlugin,
        counts: InflightStore,
        endpoints: Callable[[], Sequence[Endpoint]],
    ) -> None:
        self._scheduler = scheduler
        self._plugin = plugin
        self._counts = counts
        self._endpoints = endpoints
        self._waiting: dict[str, tuple[LLMRequest, Deliver]] = {}

    def admit(self, request: LLMRequest, ticket: str, deliver: Deliver) -> Endpoint | None:
        """Admit the request to an engine with room, or queue it under its ticket until one has."""
        winner = self._try_admit(request)
        if winner is None:
            self._waiting[ticket] = (request, deliver)
        return winner

    def cancel(self, ticket: str) -> None:
        """Drop a queued request whose caller gave up."""
        self._waiting.pop(ticket, None)

    def has_waiting(self) -> bool:
        return bool(self._waiting)

    def finish(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        """Release the request's reservation and retry the requests waiting for one."""
        self._counts.decrement(endpoint_name)
        self._plugin.release(LLMRequest(request_id=request_id), endpoint_name, output_tokens)
        self.retry()

    def _try_admit(self, request: LLMRequest) -> Endpoint | None:
        candidates = self._endpoints()
        for ep in candidates:
            ep.attributes["queue_len"] = self._counts.get(ep.name)
        allowed = self._plugin.get_allowed_candidates(request, candidates)
        if not allowed:
            return None
        picked = self._scheduler.run(request, allowed)
        if not picked:
            raise RuntimeError("the profile picked no engine among those with room")
        winner = picked[0].endpoint
        self._plugin.reserve(request, winner)
        self._counts.increment(winner.name)
        return winner

    def retry(self) -> None:
        """Admit the queued requests that now fit."""
        # Longest prompt first: late turns are the step's critical path and the costliest to stall.
        waiting = sorted(
            self._waiting.items(), key=lambda item: prefill_tokens(item[1][0].body), reverse=True
        )
        for ticket, (request, deliver) in waiting:
            try:
                winner = self._try_admit(request)
            except Exception as e:  # noqa: BLE001 - the waiting caller must see the failure
                del self._waiting[ticket]
                deliver(e)
                continue
            if winner is not None:
                del self._waiting[ticket]
                deliver(winner)
