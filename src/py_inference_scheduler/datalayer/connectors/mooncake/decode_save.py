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
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

logger = init_logger(__name__)


def should_save_decode_request(
    *,
    is_resumed: bool,
    num_computed_tokens: int,
    prefill_end_tokens: int,
) -> bool:
    """True only for decode since vllm already saves prefill."""
    # is_resumed means vllm's own preemption-resume step (bulk block realloc,
    # skipped for one step), not a relocated request - those save normally.
    if is_resumed:
        return False
    return num_computed_tokens >= prefill_end_tokens


def should_flush_decode_save(
    *,
    token_len: int,
    num_saved_tokens: int,
    block_size: int,
    min_blocks: int,
) -> bool:
    """Aggregate decode saves into batches of at least min_blocks full blocks.

    Per-block emission produces one store put per 2MB block, each paying the
    full RPC overhead (~3.9ms against ~40us of wire time on RDMA); measured
    at 256-wide rollouts this made the save queue an admission governor.
    Blocks that never reach a full batch before the request finishes are
    simply not saved - the store is a cache, and losing the newest few
    blocks costs at most one future partial hit.
    """
    aligned = token_len // block_size * block_size
    return aligned - num_saved_tokens >= min_blocks * block_size


class DecodeKVSavingConnector(MooncakeStoreConnector):
    """
    vllm only saves prompt KV to the store; this also saves decode KV.

    So other replicas can reuse generated tokens. Enabled by save_decode_kv.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        # extra_config values arrive as JSON bools or strings.
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = (
            kv_transfer_config.kv_connector_extra_config if kv_transfer_config else {}
        )
        value = extra_config.get("save_decode_kv", False)
        self.save_decode_kv = value is True or str(value).lower() == "true"
        # RLS_DECODE_SAVE_MIN_BLOCKS: emit a decode-save only once this many
        # unsaved full blocks have accumulated (1 = legacy per-block saves).
        self.save_min_blocks = max(1, int(os.getenv("RLS_DECODE_SAVE_MIN_BLOCKS", "1")))
        # RLS_MIN_PULL_TOKENS: below this many externally-matched tokens,
        # recompute locally instead of parking the request behind an async
        # store pull (0 disables). Small pulls cost a scheduler round-trip
        # plus ~45ms/op for KV that local prefill regenerates faster.
        self.min_pull_tokens = int(os.getenv("RLS_MIN_PULL_TOKENS", "0"))
        # RLS_MAX_INFLIGHT_LOADS: cap concurrent async pulls per engine
        # (0 disables). A request awaiting an async load pre-allocates its
        # full context and holds it until the load lands, so unbounded
        # admission lets waiters own the entire KV pool - measured at
        # 256-wide: all four engines at running=0, waiting~52, KV 91-97%,
        # deadlocked because running requires blocks that only running can
        # release. Declining a pull is always safe: the request recomputes.
        self.max_inflight_loads = int(os.getenv("RLS_MAX_INFLIGHT_LOADS", "0"))
        self._inflight_loads: set[str] = set()
        # Implements the reset_cache() contract vLLM/verl already call after
        # every weight sync; see reset_cache below for why the default is on.
        self.flush_on_reset = os.getenv("RLS_FLUSH_STORE_ON_RESET", "1") != "0"
        self._flush_store = None

    def reset_cache(self) -> bool:
        """Wipe the external tier when vLLM signals a weight-update reset.

        verl passes reset_connector=True to reset_prefix_cache() after every
        weight sync (vllm_async_server.py), intending to "drop any attached
        external KV store whose entries were computed against the previous
        weights". The base connector never implements reset_cache(), so it
        returns None, which the scheduler reads as success while the remote
        tier keeps serving KV written under earlier weights - measured as a
        store-only entropy climb 0.198 -> 0.775 over four LoRA steps against
        a flat recompute control. Store keys are content hashes with no
        weight version, so wiping is the only invalidation available.

        Set RLS_FLUSH_STORE_ON_RESET=0 to restore the un-flushed behaviour
        that benchmark-results/verl-swe-ab/pressure-pair-v2 was recorded on.
        """
        if not self.flush_on_reset:
            return True
        store = self._flush_client()
        if store is None:
            return False
        rc = store.remove_all()
        if rc < 0:
            logger.warning("Mooncake store flush failed on reset (rc=%s)", rc)
            return False
        logger.info("Flushed external KV store on weight-update reset")
        return True

    def _flush_client(self):
        """A segment-less store client: mounts no memory, only issues RPCs."""
        if self._flush_store is not None:
            return self._flush_store
        try:
            import json
            import socket

            from mooncake.store import MooncakeDistributedStore

            cfg = json.load(open(os.environ["MOONCAKE_CONFIG_PATH"]))
            store = MooncakeDistributedStore()
            host = socket.gethostbyname(socket.gethostname())
            # 0 global segment + 0 local buffer: setup skips mounting and
            # registration, leaving a control-plane-only client.
            rc = store.setup(
                f"{host}:0",
                cfg["metadata_server"],
                0,
                0,
                cfg["protocol"],
                cfg.get("device_name", ""),
                cfg["master_server_address"],
            )
            if rc != 0:
                logger.warning("Mooncake flush client setup failed (rc=%s)", rc)
                return None
            self._flush_store = store
        except Exception:
            logger.exception("Could not build the Mooncake flush client")
            return None
        return self._flush_store

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Skip the store lookup when local KV already covers the prompt.

        The base lookup is a blocking ZMQ hop plus a master RPC carrying one
        key per block per rank, and it runs inside the scheduler loop - so it
        stalls the engine for every request, including ones the local prefix
        cache already satisfies. Upstream only compares against the local hit
        after paying for the lookup; comparing first costs nothing.
        """
        sched = self.connector_scheduler
        if sched is not None:
            block_size = sched._block_size
            token_len = request.num_tokens // block_size * block_size
            if num_computed_tokens >= token_len:
                return 0, False
        # vllm is absent from the typecheck env, so the base call is Any; the
        # annotation is where its contract gets stated.
        matched: tuple[int, bool] = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if 0 < matched[0] < self.min_pull_tokens:
            return 0, False
        # matched[1] is load_kv_async: those are the pulls that park a
        # request on pre-allocated blocks, so only those are capped.
        if matched[0] > 0 and matched[1] and self.max_inflight_loads:
            if len(self._inflight_loads) >= self.max_inflight_loads:
                return 0, False
            self._inflight_loads.add(request.request_id)
        return matched

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
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
            sched_now = self.connector_scheduler
            if sched_now is not None:
                self._inflight_loads.intersection_update(
                    sched_now._unfinished_requests.keys()
                )
        if not isinstance(meta, MooncakeStoreConnectorMetadata):
            return meta  # upstream returned a different metadata type
        if not self.save_decode_kv or self.kv_role == "kv_consumer":
            return meta

        sched = self.connector_scheduler
        if sched is None:  # set for the scheduler role only
            return meta
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            new_block_ids = cached.new_block_ids[i]
            if not new_block_ids:
                continue
            tracker = sched._request_trackers.get(req_id)
            if tracker is None:
                continue
            if not should_save_decode_request(
                is_resumed=req_id in cached.resumed_req_ids,
                num_computed_tokens=cached.num_computed_tokens[i],
                prefill_end_tokens=tracker.prefill_end_tokens,
            ):
                continue
            req_tuple = sched._unfinished_requests.get(req_id)
            if not req_tuple:
                continue
            unfinished_req = req_tuple[0]

            # The stock scheduler only advances tracker.token_len on
            # block-allocating steps, so in decode it lags the real token count;
            true_token_len = (
                cached.num_computed_tokens[i]
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            tracker.token_len = max(tracker.token_len, true_token_len)

            tracker.update(new_block_ids)

            if not should_flush_decode_save(
                token_len=tracker.token_len,
                num_saved_tokens=tracker.num_saved_tokens,
                block_size=sched._block_size,
                min_blocks=self.save_min_blocks,
            ):
                continue

            # returns None until a new full block has completed.
            req_meta = ReqMeta.from_request_tracker(
                tracker,
                sched._block_size,
                load_spec=None,
                skip_save=False,
                block_hashes=unfinished_req.block_hashes,
                is_last_chunk=False,
                original_block_size=sched.original_block_size,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
        return meta
