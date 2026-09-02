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

from py_inference_scheduler.framework import Endpoint


def validate_saturation_thresholds(kv_threshold: float, waiting_threshold: int) -> None:
    """Reject threshold configs that would make the saturation check vacuous."""
    if not 0.0 < kv_threshold <= 1.0:
        raise ValueError("kv_threshold must be in (0, 1].")
    if waiting_threshold <= 0:
        raise ValueError("waiting_threshold must be positive.")


def endpoint_load(ep: Endpoint) -> tuple[float, int]:
    """KV utilization and waiting-request count; missing stats read as unloaded."""
    stats = ep.attributes.get("routing_stats", {})
    if not isinstance(stats, dict):
        stats = {}
    return float(stats.get("kv", 0.0)), int(stats.get("num_waiting_reqs", 0))


def saturation_reason(
    *, kv: float, waiting: int, kv_threshold: float, waiting_threshold: int
) -> str | None:
    """Which threshold(s) fired, or None if the endpoint is healthy."""
    reasons = []
    if kv >= kv_threshold:
        reasons.append(f"kv={kv:.2f}")
    if waiting >= waiting_threshold:
        reasons.append(f"waiting={waiting}")
    return ",".join(reasons) if reasons else None
