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

from py_inference_scheduler.framework import Endpoint
from py_inference_scheduler.plugins.common import (
    endpoint_load,
    saturation_reason,
    validate_saturation_thresholds,
)


def test_reason_names_each_fired_threshold():
    kw = {"kv_threshold": 0.9, "waiting_threshold": 16}
    assert saturation_reason(kv=0.95, waiting=2, **kw) == "kv=0.95"
    assert saturation_reason(kv=0.1, waiting=40, **kw) == "waiting=40"
    assert saturation_reason(kv=0.95, waiting=40, **kw) == "kv=0.95,waiting=40"
    assert saturation_reason(kv=0.5, waiting=3, **kw) is None


def test_thresholds_are_inclusive():
    kw = {"kv_threshold": 0.9, "waiting_threshold": 16}
    assert saturation_reason(kv=0.9, waiting=0, **kw) == "kv=0.90"
    assert saturation_reason(kv=0.0, waiting=16, **kw) == "waiting=16"
    assert saturation_reason(kv=0.89, waiting=15, **kw) is None


def test_endpoint_load_reads_routing_stats():
    ep = Endpoint(name="ep", attributes={"routing_stats": {"kv": 0.7, "num_waiting_reqs": 5}})
    assert endpoint_load(ep) == (0.7, 5)


def test_missing_or_malformed_stats_read_as_unloaded():
    assert endpoint_load(Endpoint(name="fresh")) == (0.0, 0)
    assert endpoint_load(Endpoint(name="empty", attributes={"routing_stats": {}})) == (0.0, 0)
    assert endpoint_load(Endpoint(name="bad", attributes={"routing_stats": "oops"})) == (0.0, 0)


def test_validation_rejects_vacuous_thresholds():
    with pytest.raises(ValueError, match="kv_threshold"):
        validate_saturation_thresholds(0.0, 16)
    with pytest.raises(ValueError, match="kv_threshold"):
        validate_saturation_thresholds(1.5, 16)
    with pytest.raises(ValueError, match="waiting_threshold"):
        validate_saturation_thresholds(0.9, 0)
    validate_saturation_thresholds(1.0, 1)
