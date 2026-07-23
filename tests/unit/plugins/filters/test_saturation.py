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

from scheduling.framework import CycleState, Endpoint, LLMRequest
from scheduling.plugins import SaturationFilter


def _ep(name: str, kv: float = 0.0, waiting: int = 0) -> Endpoint:
    return Endpoint(
        name=name,
        attributes={"routing_stats": {"kv": kv, "num_waiting_reqs": waiting}},
    )


def _run(flt: SaturationFilter, pods: dict[str, Endpoint]) -> dict[str, Endpoint]:
    return dict(flt.filter(CycleState(), LLMRequest(request_id="r"), pods))


def test_drops_saturated_keeps_healthy():
    flt = SaturationFilter(kv_threshold=0.9, waiting_threshold=16)
    pods = {
        "hot_kv": _ep("hot_kv", kv=0.95),
        "deep_queue": _ep("deep_queue", waiting=40),
        "ok": _ep("ok", kv=0.4, waiting=3),
    }
    assert set(_run(flt, pods)) == {"ok"}


def test_thresholds_are_inclusive():
    flt = SaturationFilter(kv_threshold=0.9, waiting_threshold=16)
    pods = {
        "at_kv": _ep("at_kv", kv=0.9),
        "at_waiting": _ep("at_waiting", waiting=16),
        "under": _ep("under", kv=0.89, waiting=15),
    }
    assert set(_run(flt, pods)) == {"under"}


def test_all_saturated_returns_full_set():
    """A filter must never leave the ballot empty."""
    flt = SaturationFilter(kv_threshold=0.9, waiting_threshold=16)
    pods = {name: _ep(name, kv=0.99, waiting=99) for name in ("a", "b", "c")}
    assert set(_run(flt, pods)) == {"a", "b", "c"}


def test_missing_stats_counts_as_healthy():
    flt = SaturationFilter()
    pods = {"fresh": Endpoint(name="fresh"), "hot": _ep("hot", kv=0.99)}
    assert set(_run(flt, pods)) == {"fresh"}


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="kv_threshold"):
        SaturationFilter(kv_threshold=0.0)
    with pytest.raises(ValueError, match="kv_threshold"):
        SaturationFilter(kv_threshold=1.5)
    with pytest.raises(ValueError, match="waiting_threshold"):
        SaturationFilter(waiting_threshold=0)


def test_drop_log_includes_reason():
    flt = SaturationFilter(kv_threshold=0.9, waiting_threshold=16)
    assert flt._saturation_reason(kv=0.95, waiting=2) == "kv=0.95"
    assert flt._saturation_reason(kv=0.1, waiting=40) == "waiting=40"
    assert flt._saturation_reason(kv=0.95, waiting=40) == "kv=0.95,waiting=40"
    assert flt._saturation_reason(kv=0.5, waiting=3) is None
