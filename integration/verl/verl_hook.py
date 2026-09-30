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
"""verl integration hook: delegate rollout routing to py-inference-scheduler.

Supports two verl layouts, auto-detected at import time:

- **legacy** (v0.7.1): ``AsyncLLMServerManager`` lives in
  ``verl.experimental.agent_loop.agent_loop`` and owns the server list.
- **modern** (v0.9.x): ``LLMServerClient`` lives in
  ``verl.workers.rollout.llm_server``; a ``GlobalRequestLoadBalancer`` Ray
  actor owns the server registry and does atomic acquire. The scheduler client
  bootstraps its endpoint set by draining the balancer once at first use
  (acquire every server with unique request ids, record the handles, release).

Both layouts expose the same entrypoint for the trainer flag:
``+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager``
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import OrderedDict
from typing import Sequence

import ray
from omegaconf import DictConfig  # type: ignore[import-not-found]

try:  # legacy layout (verl v0.7.x)
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
        AgentLoopManager,
        AgentLoopWorker,
    )
    from verl.experimental.agent_loop.agent_loop import (
        AsyncLLMServerManager as _LegacyServerManager,
    )

    _VERL_LAYOUT = "legacy"
except ImportError:  # modern layout (verl v0.9.x)
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
        AgentLoopManager,
        AgentLoopWorker,
    )
    from verl.workers.rollout.llm_server import (  # type: ignore[import-not-found]
        LLMServerClient as _ModernServerClient,
    )

    _VERL_LAYOUT = "modern"

from backends.verl.sglang import SglangEnginePatch
from backends.verl.vllm import VllmEnginePatch
from integration.verl.shared_inflight import METRICS_INTERVAL_MS, SharedInflightLedger
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.flow_control import FlowControlClosedError, FlowControlManager
from py_inference_scheduler.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)

# Seconds between FLEET lines per worker; the line is the measurement
# instrument for per-engine balance, so it must not be so frequent that
# it dominates the driver log.
_FLEET_LOG_INTERVAL_S = 15.0
# request ids remembered for the AFFINITY line; a rollout has 512.
_CONTINUITY_CAPACITY = 20000
# Mirror reads that may race the poller's first tick, and the snapshot age, in
# poll intervals, past which a worker reports that it is routing on stale metrics.
_WARMUP_APPLIES = 2
_STALE_INTERVALS = 5
# How long the first decision waits for the fleet poller's first snapshot before
# scoring on empty stats (every engine would look idle).
_FIRST_POLL_WAIT_S = 2.0
logger.info("py-inference-scheduler verl hook: %s layout detected", _VERL_LAYOUT)

# Must apply at module level to patch classes before use across distributed
# Ray workers without modifying verl.
VllmEnginePatch.apply()
SglangEnginePatch.apply()


def _rollout_config(config: DictConfig):
    if config.get("actor_rollout_ref"):
        return config.actor_rollout_ref.rollout
    return config.rollout


class _SchedulerCore:
    """Layout-independent scheduling state: engine, inflight tracking, metrics."""

    def __init__(self) -> None:
        self.scheduler = Scheduler()
        self._mirror: asyncio.Task[None] | None = None
        self._applies = 0
        self._fleet_staleness = float("inf")
        self._last_stale_warn = 0.0
        self.shared_inflight = SharedInflightLedger()
        self.endpoints: list[Endpoint] = []
        self.lb_acquired_requests: set[str] = set()
        self.lock = asyncio.Lock()
        self._last_fleet_log = 0.0
        # Routing continuity for the AFFINITY line: did this turn land where
        # the previous turn of the same request_id ran? "moved" is the count
        # of full-context re-prefills the router chose to pay.
        self._last_endpoint: OrderedDict[str, str] = OrderedDict()
        self.affinity = {"kept": 0, "moved": 0, "fresh": 0}
        # Admission-only: gate (and park) but do NOT choose the server, leaving
        # placement to verl's own balancer. Isolates flow control in an A/B.
        self.admission_only = os.environ.get("RLS_ADMISSION_ONLY", "0") == "1"
        # Parks a request while every engine is over the profile's flow-control
        # thresholds; its watcher polls _refresh_endpoints (which takes the
        # lock itself) and re-admits at an AIMD-paced rate.
        self.flow_control = FlowControlManager(
            self.scheduler.get_flow_control_plugins,
            self._refresh_endpoints,
            poll_interval_s=float(os.environ.get("FLOW_CONTROL_POLL_S", "0.1")),
        )
        # print(): only actor stdout reaches the Ray driver log. One line per
        # AgentLoopWorker is live proof of the shipped code version and of the
        # worker fan-out behind this core; the pid keeps it unique per actor so
        # Ray's cross-actor stdout dedup cannot collapse it.
        print(
            f"RLS[{os.getpid()}] core init: shared inflight ledger enabled; "
            f"flow control plugins={len(self.scheduler.get_flow_control_plugins())} "
            f"admission_only={int(self.admission_only)}"
        )

    async def _refresh_endpoints(self) -> Sequence[Endpoint]:
        """The fleet view the mirror keeps current; no scrape on this path.

        Also the flow-control watcher's poll source while a task is parked.
        """
        if self._mirror is None or self._mirror.done():
            self._mirror = asyncio.create_task(self._mirror_fleet())
            await self._apply_fleet(wait_for_first_poll=True)
        return self.endpoints

    async def _mirror_fleet(self) -> None:
        while True:
            await asyncio.sleep(METRICS_INTERVAL_MS / 1000.0)
            try:
                await self._apply_fleet()
            except Exception as exc:  # noqa: BLE001 - keep mirroring; the next tick may succeed
                self._warn_stale(f"fleet read failed: {exc!r}")

    async def _apply_fleet(self, *, wait_for_first_poll: bool = False) -> None:
        async with self.lock:
            fleet = await self.shared_inflight.fleet()
            deadline = time.monotonic() + _FIRST_POLL_WAIT_S
            while (
                wait_for_first_poll
                and fleet["staleness"] == float("inf")
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
                fleet = await self.shared_inflight.fleet()
            stats: dict[str, dict[str, object]] = fleet["stats"]  # type: ignore[assignment]
            inflight: dict[str, int] = fleet["inflight"]  # type: ignore[assignment]
            for ep in self.endpoints:
                ep.attributes["routing_stats"] = stats.get(ep.name, {})
                ep.attributes["queue_len"] = inflight.get(ep.name, 0)
        self._fleet_staleness = float(fleet["staleness"])  # type: ignore[arg-type]
        self._applies += 1
        self._log_fleet()

    def _warn_stale(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_stale_warn > _FLEET_LOG_INTERVAL_S:
            self._last_stale_warn = now
            print(f"RLS[{os.getpid()}] {message}")

    def _log_fleet(self) -> None:
        """Periodic snapshot of EVERY engine, the per-engine balance instrument.

        Per-decision lines describe only the selected endpoint, so they cannot
        show an engine nobody is selecting. Emitted with print(): under Ray
        only an actor's stdout reaches the driver log. pid-tagged so identical
        lines from different workers are not deduplicated away, and so the log
        proves every worker sees the whole fleet with the same q counts.
        """
        now = time.monotonic()
        if now - self._last_fleet_log < _FLEET_LOG_INTERVAL_S:
            return
        self._last_fleet_log = now
        fleet = [
            "{}=kv{:.2f}/w{}/r{}/p{}/q{}".format(
                ep.name.rsplit(":", 1)[-1],
                float(ep.attributes.get("routing_stats", {}).get("kv", 0.0)),
                ep.attributes.get("routing_stats", {}).get("num_waiting_reqs", 0),
                ep.attributes.get("routing_stats", {}).get("num_running_reqs", 0),
                ep.attributes.get("routing_stats", {}).get("num_preempted", 0),
                ep.attributes.get("queue_len", 0),
            )
            for ep in self.endpoints
        ]
        print(f"FLEET[{os.getpid()}] " + " ".join(fleet))
        a = self.affinity
        print(f"AFFINITY[{os.getpid()}] kept={a['kept']} moved={a['moved']} fresh={a['fresh']}")

    async def schedule(self, request_id: str, prompt_ids: list[int] | None) -> Endpoint | None:
        """Gate and pick an endpoint from the mirrored fleet view; None means verl's LB.

        Parking happens OUTSIDE the lock so the mirror and the watcher keep
        running while a task waits.
        """
        request = LLMRequest(request_id=request_id, body=prompt_ids)
        candidates: Sequence[Endpoint] = await self._refresh_endpoints()
        # Staleness only matters when a decision uses the snapshot: an old poll
        # during the weight update, with nothing to route, is not worth a line.
        if (
            self._applies > _WARMUP_APPLIES
            and self._fleet_staleness > _STALE_INTERVALS * METRICS_INTERVAL_MS / 1000.0
        ):
            self._warn_stale(f"routing on stale metrics: snapshot {self._fleet_staleness:.2f}s old")
        if candidates and self.flow_control.has_plugins():
            try:
                candidates = await self.flow_control.admit(request, candidates)
            except FlowControlClosedError:
                return None
            if not candidates:
                return None
        if self.admission_only:
            # Admission was gated; placement stays with verl's balancer.
            return None
        async with self.lock:
            # Exact fleet counts for this decision, read under the lock so no
            # dispatch counted by another task is overwritten in between (the
            # mirror's copy is up to one interval old, and a parked request may
            # have waited far longer).
            counts = await self.shared_inflight.counts()
            for ep in self.endpoints:
                ep.attributes["queue_len"] = counts.get(ep.name, 0)
            selected = self.scheduler.run(request, candidates=candidates)
            if not selected:
                return None
            winner: Endpoint = selected[0].endpoint
            self._note_continuity(request_id, winner.name)
            self.note_dispatch(winner.name)
            self.flow_control.commit(request, winner)
            return winner

    def _note_continuity(self, request_id: str, endpoint_name: str) -> None:
        prev = self._last_endpoint.get(request_id)
        key = "fresh" if prev is None else "kept" if prev == endpoint_name else "moved"
        self.affinity[key] += 1
        if key == "moved" and prev is not None:
            # Engine names are host:port, so a host change is a node change:
            # the next turn's context can only come from the KV tier.
            cross = int(prev.rsplit(":", 1)[0] != endpoint_name.rsplit(":", 1)[0])
            print(
                f"MOVE[{os.getpid()}] rid={request_id} from={prev} to={endpoint_name}"
                f" cross_node={cross}"
            )
        self._last_endpoint[request_id] = endpoint_name
        self._last_endpoint.move_to_end(request_id)
        if len(self._last_endpoint) > _CONTINUITY_CAPACITY:
            self._last_endpoint.popitem(last=False)

    def note_dispatch(self, endpoint_name: str) -> None:
        """Count a dispatch in the fleet ledger and on the endpoint itself."""
        self.shared_inflight.increment(endpoint_name)
        self._bump_queue_len(endpoint_name, +1)

    def _bump_queue_len(self, endpoint_name: str, delta: int) -> None:
        # The mirror rewrites queue_len once per interval; without counting
        # here, a burst's decisions all score one snapshot and the engine it
        # ranks lowest takes the whole batch (118 vs 34 first turns, 09-26).
        for ep in self.endpoints:
            if ep.name == endpoint_name:
                ep.attributes["queue_len"] = max(0, int(ep.attributes.get("queue_len", 0)) + delta)
                return

    def release(self, server_id: str, request_id: str | None = None) -> None:
        self.shared_inflight.decrement(server_id)
        self._bump_queue_len(server_id, -1)
        self.flow_control.release(LLMRequest(request_id=request_id or "", body=None), server_id)


if _VERL_LAYOUT == "legacy":

    class InferenceSchedulerServerManager(_LegacyServerManager):  # type: ignore[misc]
        """Delegate routing to py-inference-scheduler. Compatible with verl v0.7.1."""

        def __init__(
            self,
            config: DictConfig,
            servers: list[tuple[str, ray.actor.ActorHandle]],
            load_balancer_handle: ray.actor.ActorHandle,
            *args: object,
            **kwargs: object,
        ) -> None:
            super().__init__(config, servers, load_balancer_handle, *args, **kwargs)
            self.rollout_config = _rollout_config(config)
            self.core = _SchedulerCore()
            self.core.endpoints = [
                Endpoint(name=server_id, attributes={"replica_obj": handle, "routing_stats": {}})
                for server_id, handle in servers
            ]

        async def _acquire_server(
            self,
            request_id: str,
            prompt_ids: list[int] | None = None,
        ) -> tuple[str, ray.actor.ActorHandle]:
            winner = await self.core.schedule(request_id, prompt_ids)
            if winner is None:
                if not self.core.admission_only:
                    logger.warning(
                        "py-inference-scheduler returned no endpoints, falling back to verl LB."
                    )
                self.core.lb_acquired_requests.add(request_id)
                server_id, handle = await super()._acquire_server(request_id)  # type: ignore[no-any-return]
                self.core.note_dispatch(server_id)
                return server_id, handle
            return winner.name, winner.attributes["replica_obj"]

        def _release_server(self, server_id: str, request_id: str | None = None) -> None:
            self.core.release(server_id, request_id)
            if request_id and request_id in self.core.lb_acquired_requests:
                super()._release_server(server_id)
                self.core.lb_acquired_requests.remove(request_id)

        async def generate(
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, object],
            image_data: list[object] | None = None,
            video_data: list[object] | None = None,
        ) -> object:
            # Yield CPU so queued metric/scheduling tasks can interleave.
            await asyncio.sleep(0)
            server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)

            # vLLMAsyncServer ignores ignore_eos from config, so pass it explicitly.
            # A fresh request_id per generation avoids vLLM KV-cache collisions
            # with verl's sticky multi-turn request ids.
            ignore_eos = self.rollout_config.get("ignore_eos", False)
            if isinstance(sampling_params, dict):
                sampling_params["ignore_eos"] = ignore_eos
            elif hasattr(sampling_params, "ignore_eos"):
                sampling_params.ignore_eos = ignore_eos

            try:
                return await server.generate.remote(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                )
            finally:
                self._release_server(server_id, request_id)

    class PyInferenceAgentLoopWorker(AgentLoopWorker):  # type: ignore[misc]
        """Inject the custom ServerManager before calling super().__init__."""

        def __init__(
            self,
            config: DictConfig,
            servers: list[tuple[str, ray.actor.ActorHandle]],
            load_balancer_handle: ray.actor.ActorHandle,
            reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
        ) -> None:
            self.server_manager = InferenceSchedulerServerManager(
                config, servers, load_balancer_handle
            )
            super().__init__(config, servers, load_balancer_handle, reward_loop_worker_handles)

else:  # modern layout

    class InferenceSchedulerServerClient(_ModernServerClient):  # type: ignore[misc]
        """Delegate routing to py-inference-scheduler. Compatible with verl v0.9.x.

        The GlobalRequestLoadBalancer actor owns the (server_id -> handle)
        registry but enumerates ids only; the shared ledger actor recovers
        the handles once for the whole fleet (see shared_inflight.py).
        """

        def __init__(
            self,
            config: DictConfig,
            load_balancer_handle: ray.actor.ActorHandle = None,
            **kwargs: object,
        ) -> None:
            super().__init__(config, load_balancer_handle, **kwargs)
            self.rollout_config = _rollout_config(config)
            self.core = _SchedulerCore()

        async def _ensure_endpoints(self) -> None:
            server_ids = await self._load_balancer.get_all_servers.remote()
            known = {ep.name: ep for ep in self.core.endpoints}
            if known and len(known) >= len(server_ids):
                return
            # Engines register during startup, so a short view is re-checked
            # on every request until it covers the balancer's server list.
            handles = await self.core.shared_inflight.discover_servers(
                self._load_balancer, len(server_ids)
            )
            # A step's first decisions all get here together; re-read so only
            # the one that actually changes the view says so.
            known = {ep.name: ep for ep in self.core.endpoints}
            if set(handles) == set(known):
                return
            self.core.endpoints = [
                known.get(server_id)
                or Endpoint(name=server_id, attributes={"replica_obj": handle, "routing_stats": {}})
                for server_id, handle in handles.items()
            ]
            print(f"RLS[{os.getpid()}] endpoint view: {len(handles)}/{len(server_ids)} servers")

        async def _acquire_server(
            self,
            request_id: str,
            prompt_ids: list[int] | None = None,
        ) -> tuple[str, ray.actor.ActorHandle]:
            await self._ensure_endpoints()
            winner = await self.core.schedule(request_id, prompt_ids)
            if winner is None:
                if not self.core.admission_only:
                    logger.warning(
                        "py-inference-scheduler returned no endpoints, falling back to verl LB."
                    )
                self.core.lb_acquired_requests.add(request_id)
                server_id, handle = await super()._acquire_server(request_id)
                self.core.note_dispatch(server_id)
                return server_id, handle
            return winner.name, winner.attributes["replica_obj"]

        def _release_server(self, server_id: str, request_id: str | None = None) -> None:
            self.core.release(server_id, request_id)
            if request_id and request_id in self.core.lb_acquired_requests:
                super()._release_server(server_id)
                self.core.lb_acquired_requests.remove(request_id)

        async def generate(  # noqa: PLR0913
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, object],
            image_data: list[object] | None = None,
            video_data: list[object] | None = None,
            audio_data: list[object] | None = None,
            mm_processor_kwargs: dict[str, object] | None = None,
            **kwargs: object,
        ) -> object:
            await asyncio.sleep(0)
            server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)

            ignore_eos = self.rollout_config.get("ignore_eos", False)
            if isinstance(sampling_params, dict):
                sampling_params["ignore_eos"] = ignore_eos

            multimodal_kwargs: dict[str, object] = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            try:
                return await server.generate.remote(
                    request_id=uuid.uuid4().hex,  # fresh id per turn, mirrors upstream
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                    **multimodal_kwargs,
                    **kwargs,
                )
            finally:
                self._release_server(server_id, request_id)

    class PyInferenceAgentLoopWorker(AgentLoopWorker):  # type: ignore[misc,no-redef]
        """Swap the incoming LLMServerClient for the scheduler-backed client."""

        def __init__(
            self,
            config: DictConfig,
            llm_client: object,
            teacher_client: dict | None = None,
            reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
        ) -> None:
            scheduler_client = InferenceSchedulerServerClient(
                config, load_balancer_handle=llm_client._load_balancer
            )
            super().__init__(config, scheduler_client, teacher_client, reward_loop_worker_handles)


class PyInferenceAgentLoopManager(AgentLoopManager):
    """Main hook entrypoint loaded by ray_trainer.py.

    Overrides the worker actor class that verl spawns across the cluster.
    Works on both supported verl layouts (the worker class above is selected
    at import time).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.agent_loop_workers_class = ray.remote(PyInferenceAgentLoopWorker)
        super().__init__(*args, **kwargs)
