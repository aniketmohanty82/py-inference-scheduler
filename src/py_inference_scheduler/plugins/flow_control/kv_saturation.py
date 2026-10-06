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
from typing import Sequence, TypeVar

from py_inference_scheduler.framework import (
    Endpoint,
    FlowControlPlugin,
    LLMRequest,
    register_flow_control,
)

# Trajectories remembered for their last engine and output; a rollout's worth many times over.
_REMEMBERED = 65536
# Text prompts carry no token ids, so their size is estimated from their length.
_CHARS_PER_TOKEN = 4

_V = TypeVar("_V")


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
        self._held: dict[str, tuple[str, int]] = {}
        self._holder: OrderedDict[str, str] = OrderedDict()
        self._last_output: OrderedDict[str, int] = OrderedDict()

    def get_allowed_candidates(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[Endpoint]:
        need = self._need(request)
        fitting = [ep for ep in candidates if self._fits(ep, need)]
        holder = self._holder.get(request.request_id)
        for ep in fitting:
            if ep.name == holder:
                return [ep]
        return fitting

    def reserve(self, request: LLMRequest, selected: Endpoint) -> None:
        tokens = min(self._need(request), _capacity(selected))
        self._reserved[selected.name] = self._reserved.get(selected.name, 0) + tokens
        self._held[request.request_id] = (selected.name, tokens)
        _remember(self._holder, request.request_id, selected.name)

    def release(
        self, request: LLMRequest, endpoint_name: str, output_tokens: int | None = None
    ) -> None:
        held = self._held.pop(request.request_id, None)
        if held is not None:
            engine, tokens = held
            self._reserved[engine] -= tokens
        if output_tokens is not None:
            _remember(self._last_output, request.request_id, output_tokens)

    def _need(self, request: LLMRequest) -> int:
        output = self._last_output.get(request.request_id, self.default_osl)
        return prefill_tokens(request.body) + output

    def _fits(self, ep: Endpoint, need: int) -> bool:
        capacity = _capacity(ep)
        # An oversize request reserves the whole engine, so it waits for an idle one.
        return capacity > 0 and self._reserved.get(ep.name, 0) + min(need, capacity) <= capacity


def prefill_tokens(body: object) -> int:
    """Tokens a request's prompt occupies: exact for token ids, estimated for text."""
    if isinstance(body, list) and body and isinstance(body[0], int):
        return len(body)
    if isinstance(body, (str, bytes)):
        return len(body) // _CHARS_PER_TOKEN
    if isinstance(body, list):
        contents = (
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "") for m in body
        )
        return sum(len(str(c)) for c in contents) // _CHARS_PER_TOKEN
    return 0


def _capacity(ep: Endpoint) -> int:
    value = ep.attributes.get("kv_cache_size")
    return int(value) if isinstance(value, (int, float)) else 0


def _remember(table: OrderedDict[str, _V], key: str, value: _V) -> None:
    table[key] = value
    table.move_to_end(key)
    if len(table) > _REMEMBERED:
        table.popitem(last=False)
