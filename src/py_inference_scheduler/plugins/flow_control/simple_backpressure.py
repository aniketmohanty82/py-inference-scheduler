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
from typing import Sequence

from py_inference_scheduler.framework import (
    Endpoint,
    FlowControlPlugin,
    LLMRequest,
    register_flow_control,
)
from py_inference_scheduler.plugins.common import (
    endpoint_load,
    saturation_reason,
    validate_saturation_thresholds,
)

logger = logging.getLogger(__name__)


@register_flow_control("simple_backpressure")
class SimpleBackpressurePlugin(FlowControlPlugin):
    """Admission gate that only passes endpoints below the saturation thresholds.

    Unlike SaturationFilter, an all-saturated fleet yields an EMPTY result:
    that is the signal for the integration's FlowControlManager to park the
    request until live metrics show capacity again. Stateless on purpose --
    queue state belongs to the manager, so config hot-reload can swap this
    plugin without stranding parked requests.
    """

    def __init__(self, kv_threshold: float = 0.95, waiting_threshold: int = 6) -> None:
        validate_saturation_thresholds(kv_threshold, waiting_threshold)
        self.kv_threshold = kv_threshold
        self.waiting_threshold = waiting_threshold

    def get_allowed_candidates(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[Endpoint]:
        allowed: list[Endpoint] = []
        dropped: list[str] = []
        for ep in candidates:
            kv, waiting = endpoint_load(ep)
            reason = saturation_reason(
                kv=kv,
                waiting=waiting,
                kv_threshold=self.kv_threshold,
                waiting_threshold=self.waiting_threshold,
            )
            if reason:
                dropped.append(f"{ep.name}({reason})")
            else:
                allowed.append(ep)

        if not allowed and candidates:
            logger.warning(
                "all %d endpoints saturated: request will queue %s", len(dropped), dropped
            )
        elif dropped:
            logger.info("backpressure gate dropped %s", dropped)
        return allowed

    def reserve(self, request: LLMRequest, selected: Endpoint) -> None:
        """No bookkeeping: admission is re-derived from live metrics."""
        return

    def release(self, request: LLMRequest, endpoint_name: str) -> None:
        """No bookkeeping: completions do not drive re-admission."""
        return
