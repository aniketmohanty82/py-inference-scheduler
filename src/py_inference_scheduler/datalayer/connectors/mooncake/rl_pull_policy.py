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

import contextlib
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
        # Upstream derives the lookup-RPC socket path from the host, the DP
        # rank and an optional lookup_rpc_port only, so co-located engines on
        # one host share one IPC path and race each other's unlink-then-bind
        # at start-up (an engine died with EADDRINUSE on the third 12-step
        # launch). Key the path on the engine's instance id instead; both the
        # scheduler-side and the worker-side halves see the same config.
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        extra.setdefault("lookup_rpc_port", f"i{vllm_config.instance_id}")
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
        # RLS_LOG_PULL_SOURCE=1: name the segment every pull came from, which
        # is the evidence for cross-node KV sharing. Upstream already fetches
        # replica descriptors per load batch when VLLM_MOONCAKE_STORE_TIER_LOG
        # is set (for a memory/disk tier line at INFO, which verl's WARN
        # filter hides); the wrapper reuses that one lookup and prints the
        # owning transport endpoints against this engine's own host.
        # Installed in register_kv_caches: upstream creates its KV receive
        # threads there, not in the constructor, and the proxy wraps them.
        self._log_pull_source = (
            self.connector_worker is not None and os.getenv("RLS_LOG_PULL_SOURCE", "0") == "1"
        )

    def register_kv_caches(self, kv_caches: dict) -> None:  # type: ignore[override]
        super().register_kv_caches(kv_caches)
        if self._log_pull_source:
            n = _install_pull_source_log(self.connector_worker)
            print(f"PULLSRC instrument installed pid={os.getpid()} recv_threads={n}", flush=True)

    def start_load_kv(self, forward_context: object, **kwargs: object) -> None:  # type: ignore[override]
        super().start_load_kv(forward_context, **kwargs)
        # vLLM skips wait_for_save() on a step that schedules no tokens
        # (kv_connector_no_forward), but upstream queues its store jobs from
        # wait_for_save() and nowhere else, while the scheduler side has
        # already pinned every block those jobs cover until each rank reports
        # the job done. A job emitted on such a step - a new turn parked on an
        # async pull while the engine is otherwise idle - is never run, never
        # reported, and its blocks stay pinned for the rest of the run; four
        # max-length turns fill the pool and the engine starves with nothing
        # running. Queue the jobs here on exactly those steps, flagged by our
        # scheduler side, so every emitted job is retired and none is queued
        # twice (a double report trips upstream's "too many ranks" assert).
        if getattr(self._get_connector_metadata(), "rls_no_forward_step", False):
            super().wait_for_save()

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        matched: tuple[int | None, bool] = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
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
        # Read by start_load_kv on the worker side; see there.
        meta.rls_no_forward_step = scheduler_output.total_num_scheduled_tokens == 0  # type: ignore[attr-defined]
        # A request that reaches the scheduled lists is no longer parked on
        # its load, so it stops counting against the in-flight cap. Requests
        # the scheduler dropped are cleared against its unfinished set,
        # which is the only lifetime view available on this side.
        if self._inflight_loads:
            for req in scheduler_output.scheduled_new_reqs:
                self._inflight_loads.discard(req.req_id)
            self._inflight_loads.difference_update(scheduler_output.scheduled_cached_reqs.req_ids)
            sched = self.connector_scheduler
            if sched is not None:
                self._inflight_loads.intersection_update(sched._unfinished_requests.keys())
        return meta


# PULLSRC line cadence: every load batch at first, then a sample with cumulative counts.
_PULLSRC_VERBOSE_BATCHES = 100
_PULLSRC_EVERY = 50


def _endpoint_host(endpoint: str | None) -> str:
    return endpoint.rsplit(":", 1)[0] if endpoint else "?"


def _replica_host(descs: object) -> str:
    """Host of a key's first replica; "?" when it is not a memory replica."""
    with contextlib.suppress(Exception):  # descriptor shapes vary by mooncake build
        desc = descs[0]  # type: ignore[index]
        if desc.is_memory_replica():
            return str(
                _endpoint_host(desc.get_memory_descriptor().buffer_descriptor.transport_endpoint)
            )
    return "?"


class _TracedStore:
    """Store proxy that records where each load batch's keys live.

    Wraps the worker's MooncakeDistributedStore: the load call records its
    replica hosts before delegating; everything else forwards unchanged.
    """

    def __init__(self, store: object, counts: dict[str, int]) -> None:
        self._store = store
        self._counts = counts

    def __getattr__(self, name: str) -> object:
        return getattr(self._store, name)

    def batch_get_into_multi_buffers(self, keys: list[str], addrs: object, sizes: object) -> object:
        _record_pull_sources(self._store, keys, self._counts)
        return self._store.batch_get_into_multi_buffers(keys, addrs, sizes)  # type: ignore[attr-defined]


def _record_pull_sources(store: object, keys: list[str], counts: dict[str, int]) -> None:
    """One PULLSRC line per sampled load batch, with cumulative key counts.

    Every batch is sampled for the first _PULLSRC_VERBOSE_BATCHES, then one in
    _PULLSRC_EVERY (each sample costs one replica-descriptor RPC). ``cum_cross``
    counts sampled keys whose replica lives on another host. Printed, not
    logged: the engine worker's stdout reaches the driver log at any vLLM log
    level.
    """
    counts["batches"] += 1
    if counts["batches"] > _PULLSRC_VERBOSE_BATCHES and counts["batches"] % _PULLSRC_EVERY:
        return
    try:
        descs_by_key = store.batch_get_replica_desc(keys)  # type: ignore[attr-defined]
        local = _endpoint_host(store.get_hostname())  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        logger.warning("pull-source lookup failed for %d keys: %s", len(keys), e)
        return
    by_host: dict[str, int] = {}
    for key in keys:
        descs = descs_by_key.get(key) if hasattr(descs_by_key, "get") else None
        host = _replica_host(descs)
        by_host[host] = by_host.get(host, 0) + 1
    counts["keys"] += len(keys)
    counts["cross_keys"] += sum(n for h, n in by_host.items() if h not in {"?", local})
    counts["unknown_keys"] += by_host.get("?", 0)
    print(
        f"PULLSRC local={local} keys={len(keys)} src={by_host} "
        f"cum_batches={counts['batches']} cum_keys={counts['keys']} "
        f"cum_cross={counts['cross_keys']} cum_unknown={counts['unknown_keys']}",
        flush=True,
    )


def _install_pull_source_log(worker: object) -> int:
    """Wrap the store handle of every KV receive thread; returns how many."""
    counts = {"batches": 0, "keys": 0, "cross_keys": 0, "unknown_keys": 0}
    threads = list(getattr(worker, "kv_recv_threads", []) or [])
    for thread in threads:
        if not isinstance(getattr(thread, "store", None), _TracedStore):
            thread.store = _TracedStore(thread.store, counts)
    return len(threads)
