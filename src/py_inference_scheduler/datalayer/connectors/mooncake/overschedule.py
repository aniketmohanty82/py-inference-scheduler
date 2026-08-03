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

from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

logger = init_logger(__name__)

# Set per request via vllm_xargs; only tagged requests are offloaded.
OFFLOAD_TAG_KEY = "rls_offload"


def wants_offload(request: Request) -> bool:
    """True when the client tagged this request for offload-on-finish."""
    sampling_params = request.sampling_params
    if sampling_params is None:  # pooling requests carry no sampling params
        return False
    extra_args = sampling_params.extra_args or {}
    return extra_args.get(OFFLOAD_TAG_KEY) in {"1", 1}


def preserve_prefix_blocks_from_config(vllm_config: VllmConfig) -> int:
    """
    Number of leading blocks never evicted, from kv_connector_extra_config.

    Required and validated at engine boot so a bad deploy fails loudly instead
    of silently evicting the shared system-prompt blocks.
    """
    kv_transfer_config = vllm_config.kv_transfer_config
    extra_config = (
        kv_transfer_config.kv_connector_extra_config if kv_transfer_config else {}
    )
    value = extra_config.get("preserve_prefix_blocks")
    # extra_config values arrive as JSON ints or strings.
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(
            "OverschedulingScheduler requires preserve_prefix_blocks (a non-negative"
            " integer) in kv_connector_extra_config"
        )
    return int(str(value))


class OverschedulingScheduler(Scheduler):
    """
    Evicts a tagged request's private KV blocks from HBM when it finishes.

    For multi-turn trajectories the KV is cold until the tool call or user
    reply returns; the store connector already saved it, so the next turn
    pulls it back on any replica instead of recomputing. Only worth enabling
    when HBM is saturated - below saturation a local cache hit beats a pull.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.preserve_prefix_blocks = preserve_prefix_blocks_from_config(self.vllm_config)
        self.num_offloaded_blocks = 0

    def _free_request(
        self,
        request: Request,
        delay_free_blocks: bool = False,  # noqa: FBT001, FBT002 - mirrors the base signature
    ) -> dict[str, Any] | None:
        # A request's block list includes blocks shared with other requests
        # (the common system-prompt prefix), and evict_blocks de-hashes even
        # blocks still in use elsewhere. Both filters below are load-bearing:
        # ref_cnt == 1 before the free means only this request holds the block
        # (also true while the connector delays the free for pending store
        # saves), and preserve_prefix_blocks shields the shared prefix at
        # moments no other request happens to reference it.
        evict_ids: set[int] = set()
        if wants_offload(request):
            blocks_per_group = self.kv_cache_manager.coordinator.get_blocks(
                request.request_id
            )
            for group_blocks in blocks_per_group:
                for block in group_blocks[self.preserve_prefix_blocks :]:
                    if block.ref_cnt == 1 and not block.is_null:
                        evict_ids.add(block.block_id)

        result = super()._free_request(request, delay_free_blocks)

        if evict_ids:
            self.kv_cache_manager.evict_blocks(evict_ids)
            self.num_offloaded_blocks += len(evict_ids)
            # The pilot reads this to confirm offload fired and at what volume.
            logger.info(
                "overschedule: evicted %d blocks for %s (total %d)",
                len(evict_ids),
                request.request_id,
                self.num_offloaded_blocks,
            )
        return result
