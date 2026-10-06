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

from collections import OrderedDict
from typing import Mapping

from py_inference_scheduler.framework import (
    CycleState,
    Endpoint,
    LLMRequest,
    ScorerPlugin,
    register_scorer,
)


@register_scorer("request_affinity")
class RequestAffinityScorer(ScorerPlugin):
    """Votes for the endpoint that served the previous request with the same request_id.

    - Scores that endpoint 1.0 and every other 0.0; an unseen request_id scores all 0.0.
    - Remembers up to `capacity` request ids and forgets the least recently routed first.
    - A vote, not a pin: when a filter drops the holder, later turns follow where this one ran.
    """

    def __init__(self, capacity: int = 20000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._holders: OrderedDict[str, str] = OrderedDict()

    def score(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> dict[str, float]:
        holder = self._holders.get(request.request_id)
        return {name: 1.0 if name == holder else 0.0 for name in pods}

    def pre_request(
        self, cycle_state: CycleState, request: LLMRequest, selected_endpoint: Endpoint
    ) -> None:
        self._holders[request.request_id] = selected_endpoint.name
        self._holders.move_to_end(request.request_id)
        while len(self._holders) > self.capacity:
            self._holders.popitem(last=False)
