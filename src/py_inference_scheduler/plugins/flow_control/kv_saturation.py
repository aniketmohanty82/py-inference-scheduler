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
from typing import Sequence

from py_inference_scheduler.framework import (
    Endpoint,
    FlowControlPlugin,
    LLMRequest,
    register_flow_control,
)
from py_inference_scheduler.framework.helpers import prefill_tokens

# Trajectories remembered for their last engine and output; a rollout's worth many times over.
_REMEMBERED = 65536


@register_flow_control("kv_saturation")
class KVSaturationPlugin(FlowControlPlugin):
    """Admits a request only where its tokens fit in an engine's KV budget.

    - A request needs its prefill plus its trajectory's last output, default_osl on turn one.
    - An engine's budget is its KV capacity minus what unfinished admitted requests reserved.
    - Offers the engine that served the trajectory's last turn if it fits, else all that fit.
    - Offers nothing when no engine fits, so the caller queues the request.
    """

    def __init__(self, default_osl: int = 1024) -> None:
        if default_osl < 0:
            raise ValueError("default_osl must be >= 0.")
        self.default_osl = default_osl
        self._reserved: dict[str, int] = {}
        self._holds: dict[str, tuple[str, int]] = {}
        self._last_turns: OrderedDict[str, tuple[str, int]] = OrderedDict()

    def get_allowed_candidates(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[Endpoint]:
        if candidates and all(_reports_no_capacity(ep) for ep in candidates):
            raise RuntimeError("kv_saturation needs each engine's KV capacity; none reports it.")
        need = self._need(request)
        fitting = [ep for ep in candidates if self._fits(ep, need, request.request_id)]
        last = self._last_turns.get(request.request_id)
        home = [ep for ep in fitting if last and ep.name == last[0]]
        return home or fitting

    def reserve(self, request: LLMRequest, selected: Endpoint) -> None:
        # A retried decision moves the hold instead of stacking a second one.
        self._drop_hold(request.request_id)
        tokens = min(self._need(request), _capacity(selected))
        self._reserved[selected.name] = self._reserved.get(selected.name, 0) + tokens
        self._holds[request.request_id] = (selected.name, tokens)

    def release(
        self, request: LLMRequest, endpoint_name: str, output_tokens: int | None = None
    ) -> None:
        engine = self._drop_hold(request.request_id)
        if engine is None or output_tokens is None:
            return
        self._last_turns[request.request_id] = (engine, output_tokens)
        self._last_turns.move_to_end(request.request_id)
        if len(self._last_turns) > _REMEMBERED:
            self._last_turns.popitem(last=False)

    def _drop_hold(self, request_id: str) -> str | None:
        held = self._holds.pop(request_id, None)
        if held is None:
            return None
        engine, tokens = held
        self._reserved[engine] -= tokens
        return engine

    def _need(self, request: LLMRequest) -> int:
        last = self._last_turns.get(request.request_id)
        output = self.default_osl if last is None else last[1]
        return prefill_tokens(request.body) + output

    def _fits(self, ep: Endpoint, need: int, request_id: str) -> bool:
        capacity = _capacity(ep)
        # A retried decision moves the request's own hold, so that hold does not count against it.
        hold = self._holds.get(request_id)
        own = hold[1] if hold and hold[0] == ep.name else 0
        reserved = self._reserved.get(ep.name, 0) - own
        # An oversize request reserves the whole engine, so it waits for an idle one.
        return capacity > 0 and reserved + min(need, capacity) <= capacity


def _capacity(ep: Endpoint) -> int:
    value = ep.attributes.get("kv_cache_size")
    return int(value) if isinstance(value, (int, float)) else 0


def _reports_no_capacity(ep: Endpoint) -> bool:
    # Engines not yet polled have empty stats and are simply waited for.
    stats = ep.attributes.get("routing_stats")
    return isinstance(stats, dict) and bool(stats) and not stats.get("error") and _capacity(ep) <= 0
