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
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

# Must live under the vllm logger hierarchy: the EngineCore only configures
# handlers for "vllm.*", so INFO from any other namespace is silently dropped.
logger = init_logger("vllm.rls.overschedule")

# Set per request via vllm_xargs; only tagged requests are offloaded.
OFFLOAD_TAG_KEY = "rls_offload"

# When set, every generation request is treated as tagged. For RL training,
# where the client (an agent harness behind a gateway) cannot inject
# vllm_xargs on its own requests.
ENV_OFFLOAD_ALL = "MOONCAKE_OVERSCHEDULING_OFFLOAD_ALL"


def offload_all_from_env() -> bool:
    return os.environ.get(ENV_OFFLOAD_ALL, "0") == "1"


def offload_all_from_config(vllm_config: VllmConfig) -> bool:
    """offload_all from kv_connector_extra_config, falling back to the env flag.

    Extra config is the reliable channel: it provably reaches the EngineCore
    process (preserve_prefix_blocks arrives the same way), while env vars
    depend on how the engine process was spawned.
    """
    value = _extra_config(vllm_config).get("offload_all")
    if value is None:
        return offload_all_from_env()
    return value in {True, "true", "True", "1"}


def wants_offload(request: Request, *, offload_all: bool = False) -> bool:
    """True when this request should be offloaded on finish.

    Pooling requests never offload; otherwise a request qualifies when the
    engine runs offload-all or the client tagged it via vllm_xargs.
    """
    sampling_params = request.sampling_params
    if sampling_params is None:  # pooling requests carry no sampling params
        return False
    if offload_all:
        return True
    extra_args = sampling_params.extra_args or {}
    return extra_args.get(OFFLOAD_TAG_KEY) in {"1", 1}


def _extra_config(vllm_config: VllmConfig) -> dict:
    kv_transfer_config = vllm_config.kv_transfer_config
    return kv_transfer_config.kv_connector_extra_config if kv_transfer_config else {}


def preserve_prefix_blocks_from_config(vllm_config: VllmConfig) -> int:
    """
    Number of leading blocks never evicted, from kv_connector_extra_config.

    Required and validated at engine boot so a bad deploy fails loudly instead
    of silently evicting the shared system-prompt blocks.
    """
    value = _extra_config(vllm_config).get("preserve_prefix_blocks")
    # extra_config values arrive as JSON ints or strings.
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(
            "OverschedulingScheduler requires preserve_prefix_blocks (a non-negative"
            " integer) in kv_connector_extra_config"
        )
    return int(str(value))


def min_kv_usage_from_config(vllm_config: VllmConfig) -> float:
    """KV usage (0.0-1.0) below which finished turns keep their blocks."""
    value = _extra_config(vllm_config).get("min_kv_usage")
    try:
        usage = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        usage = -1.0
    if not 0.0 <= usage <= 1.0:
        raise ValueError(
            "OverschedulingScheduler requires min_kv_usage (a float in [0, 1])"
            " in kv_connector_extra_config"
        )
    return usage


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
        self.min_kv_usage = min_kv_usage_from_config(self.vllm_config)
        self.offload_all = offload_all_from_config(self.vllm_config)
        if self.offload_all:
            logger.info("overschedule: %s set - offloading every finished request", ENV_OFFLOAD_ALL)
        self.num_offloaded_blocks = 0
        self.num_skipped_idle = 0

    def _under_pressure(self) -> bool:
        """True when HBM is contended enough for offloading to pay for itself."""
        if self.min_kv_usage <= 0.0:
            return True
        if self.kv_cache_manager.usage >= self.min_kv_usage:
            return True
        self.num_skipped_idle += 1
        return False

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
        if wants_offload(request, offload_all=self.offload_all) and self._under_pressure():
            blocks_per_group = self.kv_cache_manager.coordinator.get_blocks(
                request.request_id
            )
            for group_blocks in blocks_per_group:
                for block in group_blocks[self.preserve_prefix_blocks :]:
                    if block.ref_cnt == 1 and not block.is_null:
                        evict_ids.add(block.block_id)

        # vllm is absent from the typecheck env, so the base call is Any; the
        # annotation is where its contract gets stated.
        result: dict[str, Any] | None = super()._free_request(request, delay_free_blocks)

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
