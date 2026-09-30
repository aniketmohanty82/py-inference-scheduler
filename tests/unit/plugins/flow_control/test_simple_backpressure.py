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

import pytest

from py_inference_scheduler.framework import Endpoint, LLMRequest
from py_inference_scheduler.framework.registry import build_flow_control
from py_inference_scheduler.plugins import SimpleBackpressurePlugin


def _ep(name: str, kv: float = 0.0, waiting: int = 0) -> Endpoint:
    return Endpoint(
        name=name,
        attributes={"routing_stats": {"kv": kv, "num_waiting_reqs": waiting}},
    )


def _allowed(plugin: SimpleBackpressurePlugin, eps: list[Endpoint]) -> set[str]:
    return {ep.name for ep in plugin.get_allowed_candidates(LLMRequest(request_id="r"), eps)}


def test_drops_saturated_keeps_healthy():
    plugin = SimpleBackpressurePlugin(kv_threshold=0.9, waiting_threshold=6)
    eps = [_ep("hot_kv", kv=0.95), _ep("deep_queue", waiting=9), _ep("ok", kv=0.4, waiting=3)]
    assert _allowed(plugin, eps) == {"ok"}


def test_thresholds_are_inclusive():
    plugin = SimpleBackpressurePlugin(kv_threshold=0.9, waiting_threshold=6)
    eps = [_ep("at_kv", kv=0.9), _ep("at_waiting", waiting=6), _ep("under", kv=0.89, waiting=5)]
    assert _allowed(plugin, eps) == {"under"}


def test_all_saturated_returns_empty():
    """The empty result is the park signal -- the inverse of the filter's fail-open."""
    plugin = SimpleBackpressurePlugin()
    eps = [_ep(name, kv=0.99, waiting=99) for name in ("a", "b", "c")]
    assert plugin.get_allowed_candidates(LLMRequest(request_id="r"), eps) == []


def test_missing_stats_counts_as_healthy():
    plugin = SimpleBackpressurePlugin()
    eps = [Endpoint(name="fresh"), _ep("hot", kv=0.99)]
    assert _allowed(plugin, eps) == {"fresh"}


def test_reserve_and_release_are_stateless_noops():
    plugin = SimpleBackpressurePlugin()
    before = vars(plugin).copy()
    plugin.reserve(LLMRequest(request_id="r"), _ep("ep1"))
    plugin.release(LLMRequest(request_id="r"), "ep1")
    assert vars(plugin) == before


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="kv_threshold"):
        SimpleBackpressurePlugin(kv_threshold=0.0)
    with pytest.raises(ValueError, match="waiting_threshold"):
        SimpleBackpressurePlugin(waiting_threshold=0)


def test_registered_as_simple_backpressure():
    plugin = build_flow_control("simple_backpressure", kv_threshold=0.5, waiting_threshold=2)
    assert isinstance(plugin, SimpleBackpressurePlugin)
    assert _allowed(plugin, [_ep("a", kv=0.6), _ep("b", kv=0.4)]) == {"b"}
