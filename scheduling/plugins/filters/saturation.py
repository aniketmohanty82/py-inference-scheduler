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

import logging
from typing import Mapping

from scheduling.framework import (
    CycleState,
    Endpoint,
    FilterPlugin,
    LLMRequest,
    register_filter,
)

logger = logging.getLogger(__name__)


@register_filter("saturation")
class SaturationFilter(FilterPlugin):
    """Drops saturated endpoints from candidacy so affinity cannot elect them."""

    def __init__(self, kv_threshold: float = 0.95, waiting_threshold: int = 16) -> None:
        if not 0.0 < kv_threshold <= 1.0:
            raise ValueError("kv_threshold must be in (0, 1].")
        if waiting_threshold <= 0:
            raise ValueError("waiting_threshold must be positive.")
        self.kv_threshold = kv_threshold
        self.waiting_threshold = waiting_threshold

    def filter(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        pods: Mapping[str, Endpoint],
    ) -> Mapping[str, Endpoint]:
        eligible: dict[str, Endpoint] = {}
        dropped: list[str] = []
        for name, ep in pods.items():
            stats = ep.attributes.get("routing_stats", {})
            if not isinstance(stats, dict):
                stats = {}
            reason = self._saturation_reason(
                kv=float(stats.get("kv", 0.0)),
                waiting=int(stats.get("num_waiting_reqs", 0)),
            )
            if reason:
                dropped.append(f"{name}({reason})")
            else:
                eligible[name] = ep

        # A filter must never leave the ballot empty.
        if not eligible:
            logger.warning(
                "all %d endpoints saturated: filter disabled for this decision", len(pods)
            )
            return pods
        if dropped:
            logger.info("saturation filter dropped %s", dropped)
        return eligible

    def _saturation_reason(self, *, kv: float, waiting: int) -> str | None:
        """Which threshold(s) fired, or None if the endpoint is healthy."""
        reasons = []
        if kv >= self.kv_threshold:
            reasons.append(f"kv={kv:.2f}")
        if waiting >= self.waiting_threshold:
            reasons.append(f"waiting={waiting}")
        return ",".join(reasons) if reasons else None
