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

import os

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

logger = init_logger(__name__)


class RLPullPolicyConnector(MooncakeStoreConnector):
    """Upstream mooncake store plus the only admission rule still ours.

    Supersedes DecodeKVSavingConnector, which carried three things vLLM
    0.29.0 now ships itself:

    - the per-weight-update flush. Our ``reset_cache`` override existed
      because ``MooncakeStoreConnector`` inherited the base no-op, so verl's
      documented "clears both local and Mooncake KV caches at every weight
      update" silently did nothing for the Mooncake half. 0.29.0 implements
      it, with ``remove_all(force=True)`` on worker rank 0 - the same two
      details (force for access-granted leases, rank 0 for idempotence) we
      had arrived at independently.
    - decode-KV saving, now the ``save_decode_cache`` extra_config flag. Our
      version reached into six scheduler internals (``_request_trackers``,
      ``prefill_end_tokens``, ``ReqMeta.from_request_tracker``,
      ``original_block_size`` ...), showed no measured tier-volume benefit
      over stock, and is the prime suspect for a 2.2x step-1 entropy
      deviation from the no-offload baseline that stock does not exhibit.
    - the worker-side flush handshake that existed only to carry the above.

    What is left is the pull-admission policy, which upstream has no
    equivalent for. Keeping the subclass this thin is deliberate: it makes
    the A/B against stock a single-variable test.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        # RLS_MIN_PULL_TOKENS: below this many externally-matched tokens,
        # recompute locally instead of parking the request behind an async
        # store pull (0 disables). A small pull costs a scheduler round-trip
        # for KV that local prefill regenerates faster. Measured against
        # stock at 1024: 18% fewer load_get operations for the same
        # recompute avoided, because stock's marginal fetches yield ~1,227
        # tokens against its ~3,002-token average.
        self.min_pull_tokens = int(os.getenv("RLS_MIN_PULL_TOKENS", "0"))
        # RLS_MAX_INFLIGHT_LOADS: cap concurrent async pulls per engine
        # (0 disables, and 0 is the right default). A load-waiter
        # pre-allocates its full context, so unbounded admission deadlocked
        # all four engines at a 34k-token pool. At gmu 0.38 the pool is
        # ~106k and one max-length waiter holds 27% of it rather than 85%,
        # where the cap instead serialised fetches and cost more than half
        # the external tier's share (43.8% vs 73.5%). Size the pool; do not
        # cap admission.
        self.max_inflight_loads = int(os.getenv("RLS_MAX_INFLIGHT_LOADS", "0"))
        self._inflight_loads: set[str] = set()

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        matched = super().get_num_new_matched_tokens(request, num_computed_tokens)
        # 0.29 widened the first element to int | None; a None match means
        # "nothing external", so there is nothing for either rule to decline.
        num_matched, load_async = matched
        if not num_matched:
            return matched
        if 0 < num_matched < self.min_pull_tokens:
            # Upstream registers a LoadSpec before returning a match; declining
            # afterwards orphans it (a pairing upstream never produces itself),
            # and an engine crashed on it in _apply_current_save_block_ids.
            self.connector_scheduler.load_specs.pop(request.request_id, None)
            return 0, False
        # Only async pulls park a request on pre-allocated blocks, so only
        # those are capped. Declining is always safe: the request recomputes.
        if load_async and self.max_inflight_loads:
            if len(self._inflight_loads) >= self.max_inflight_loads:
                self.connector_scheduler.load_specs.pop(request.request_id, None)
                return 0, False
            self._inflight_loads.add(request.request_id)
        return matched

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        # A request that reaches the scheduled lists is no longer parked on
        # its load, so it stops counting against the in-flight cap. Requests
        # the scheduler dropped are cleared against its unfinished set,
        # which is the only lifetime view available on this side.
        if self._inflight_loads:
            for req in scheduler_output.scheduled_new_reqs:
                self._inflight_loads.discard(req.req_id)
            self._inflight_loads.difference_update(
                scheduler_output.scheduled_cached_reqs.req_ids
            )
            sched = self.connector_scheduler
            if sched is not None:
                self._inflight_loads.intersection_update(
                    sched._unfinished_requests.keys()
                )
        return meta
