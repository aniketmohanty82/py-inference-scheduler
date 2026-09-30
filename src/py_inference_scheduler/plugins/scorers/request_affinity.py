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
    """Votes for the endpoint that served the previous turn of this request_id.

    A multi-turn rollout resubmits its whole context every turn, and the KV
    for that context lives on the engine that ran the last turn; any other
    engine must prefill it from scratch. A request_id with no history scores
    every endpoint 0.0, which the profile treats as a tie, so first-turn
    placement is left to the load scorers.

    A weighted vote, not a pin: a filter can still drop the holder, and the
    next turn then follows the endpoint that actually served this one. The
    table holds `capacity` request ids and evicts the least recently routed.
    """

    def __init__(self, capacity: int = 20000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._holder: OrderedDict[str, str] = OrderedDict()

    def holder(self, request_id: str) -> str | None:
        return self._holder.get(request_id)

    def score(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> dict[str, float]:
        scores = dict.fromkeys(pods.keys(), 0.0)
        holder = self._holder.get(request.request_id)
        if holder in scores:
            scores[holder] = 1.0
        return scores

    def pre_request(
        self, cycle_state: CycleState, request: LLMRequest, selected_endpoint: Endpoint
    ) -> None:
        self._holder[request.request_id] = selected_endpoint.name
        self._holder.move_to_end(request.request_id)
        while len(self._holder) > self.capacity:
            self._holder.popitem(last=False)
