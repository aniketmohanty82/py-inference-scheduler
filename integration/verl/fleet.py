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

import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import cast

import aiohttp
import ray

from integration.verl.admission import Admission
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.poller import MetricsPoller
from py_inference_scheduler.datalayer.metrics.verl.fetch_metrics import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint, LLMRequest

_ACTOR_NAME = "rls_fleet"
_METRICS_INTERVAL_MS = int(os.environ.get("RLS_METRICS_INTERVAL_MS", "100"))
# A queued admit holds a call slot until placed, and finish must always find one to free it.
_MAX_CALLS = 1_000_000


@dataclass
class FleetSnapshot:
    """Engine metrics and fleet-wide in-flight counts, keyed by engine name."""

    stats: dict[str, dict[str, object]]
    inflight: dict[str, int]


class Fleet(InflightStore):
    """State shared by every AgentLoopWorker, held in one named Ray actor.

    - Counts in-flight requests per engine across all workers.
    - Recovers the engine handles from verl's balancer once for the whole fleet.
    - Polls engine metrics in the background, so no decision scrapes an engine.
    - With flow control in the profile, places every request and queues those that fit nowhere.
    """

    def __init__(self, interval_ms: int) -> None:
        super().__init__()
        self._endpoints: dict[str, Endpoint] = {}
        self._poller = MetricsPoller(
            lambda: list(self._endpoints.values()), self, _fetch, interval_ms=interval_ms
        )
        self._poller.start()
        self._interval_s = interval_ms / 1000
        self._discovery = asyncio.Lock()
        self._admission: Admission | None = None
        # VERIFY-ONLY instrument (scratch branch, never in a PR).
        self._dispatched: dict[str, int] = {}
        threading.Thread(target=self._report, daemon=True).start()

    def increment(self, endpoint_name: str) -> None:
        super().increment(endpoint_name)
        self._dispatched[endpoint_name] = self._dispatched.get(endpoint_name, 0) + 1

    def _report(self) -> None:
        peak_kv: dict[str, float] = {}
        peak_res: dict[str, int] = {}
        peak_waiting = 0
        tick = 0
        while True:
            time.sleep(1)
            tick += 1
            admission = self._admission
            reserved = dict(getattr(admission._plugin, '_reserved', {})) if admission else {}
            waiting = len(admission._waiting) if admission else 0
            peak_waiting = max(peak_waiting, waiting)
            for name, ep in list(self._endpoints.items()):
                s = ep.attributes.get('routing_stats', {})
                peak_kv[name] = max(peak_kv.get(name, 0.0), float(s.get('kv', 0.0)))
                peak_res[name] = max(peak_res.get(name, 0), reserved.get(name, 0))
            if tick % 15 or not self._endpoints:
                continue
            inflight = self.get_all()
            parts = []
            for name, ep in sorted(self._endpoints.items()):
                s = ep.attributes.get('routing_stats', {})
                parts.append(
                    f"{name}=cap{ep.attributes.get('kv_cache_size', 0)}/res{reserved.get(name, 0)}"
                    f"/resmax{peak_res.get(name, 0)}/kvmax{peak_kv.get(name, 0.0):.2f}"
                    f"/r{s.get('num_running_reqs', 0)}/w{s.get('num_waiting_reqs', 0)}"
                    f"/q{inflight.get(name, 0)}/d{self._dispatched.get(name, 0)}/pre{s.get('preempt', 0)}"
                )
            v = dict(admission.v) if admission else {}
            print(
                f'VERIFY_FLEET t={time.time():.0f} waiting={waiting} waitmax_n={peak_waiting} '
                f'adm={v} ' + ' '.join(parts),
                flush=True,
            )
            peak_kv.clear()
            peak_res.clear()
            peak_waiting = 0

    def watch(self, handles: dict[str, ray.actor.ActorHandle]) -> None:
        """Add these engines to the background poll."""
        for name, handle in handles.items():
            if name not in self._endpoints:
                self._endpoints[name] = Endpoint(
                    name=name, attributes={"replica_obj": handle, "routing_stats": {}}
                )

    async def discover(
        self, balancer: ray.actor.ActorHandle, expected: int
    ) -> dict[str, ray.actor.ActorHandle]:
        """Recover the engine handles from verl's balancer, which enumerates only ids.

        - Acquires with unique request ids until every engine is visited, then releases them.
        - Runs one drain at a time for the whole fleet, so workers never drain it concurrently.
        """
        async with self._discovery:
            handles = {name: ep.attributes["replica_obj"] for name, ep in self._endpoints.items()}
            acquired: list[str] = []
            # Live traffic skews the balancer's counters, so an engine can be visited twice.
            for _ in range(expected * 3):
                if len(handles) >= expected:
                    break
                server_id, handle = await balancer.acquire_server.remote(
                    request_id=f"rls-discover-{uuid.uuid4().hex}"
                )
                acquired.append(server_id)
                handles[server_id] = handle
            for server_id in acquired:
                balancer.release_server.remote(server_id=server_id)
            self.watch(handles)
            return handles

    def snapshot(self) -> FleetSnapshot:
        stats = {
            name: ep.attributes.get("routing_stats", {}) for name, ep in self._endpoints.items()
        }
        return FleetSnapshot(cast("dict[str, dict[str, object]]", stats), self.get_all())

    async def admit(self, request: LLMRequest) -> tuple[str, ray.actor.ActorHandle]:
        """Place a request on an engine with room, waiting in line until one has it."""
        try:  # VERIFY-ONLY
            winner = await self._admission_or_load().admit(request)
        except Exception as e:
            print(f'VERIFY_ADMIT_ERROR {e!r}', flush=True)
            raise
        return winner.name, winner.attributes["replica_obj"]

    def finish(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        """Free an admitted request's place and retry the requests waiting for one."""
        self._admission_or_load().finish(request_id, endpoint_name, output_tokens)

    def _admission_or_load(self) -> Admission:
        if self._admission is None:
            # Loaded once: a hot reload would replace the plugin and drop its reservations.
            config = SchedulerConfig.from_file(os.environ["ROUTER_CONFIG_PATH"])
            scheduler = Scheduler.new_with_config(config)
            plugins = scheduler.get_flow_control_plugins()
            if len(plugins) != 1:
                raise ValueError("verl admission needs exactly one flow_control plugin.")
            self._admission = Admission(
                scheduler,
                plugins[0],
                self,
                lambda: list(self._endpoints.values()),
                self._interval_s,
            )
        return self._admission


async def _fetch(ep: Endpoint, inflight: InflightStore, session: aiohttp.ClientSession) -> None:
    # verl serves engine stats over a Ray call to the server actor, so the HTTP session is unused.
    await fetch_worker_metrics(ep, inflight)


_FleetActor = ray.remote(Fleet)


def fleet_actor() -> ray.actor.ActorHandle:
    """The fleet actor, created by the first caller and shared with the rest by name."""
    # num_cpus=0: an actor pending behind verl's CPU reservations would stall every worker.
    return _FleetActor.options(
        name=_ACTOR_NAME, get_if_exists=True, num_cpus=0, max_concurrency=_MAX_CALLS
    ).remote(_METRICS_INTERVAL_MS)


async def place(
    fleet: ray.actor.ActorHandle, request: LLMRequest
) -> tuple[str, ray.actor.ActorHandle]:
    """Ask the fleet actor to admit a request, from an agent-loop worker.

    - Hands back a place made after the caller was cancelled, since Ray keeps the call running.
    """
    placing: asyncio.Future[tuple[str, ray.actor.ActorHandle]] = asyncio.ensure_future(
        fleet.admit.remote(request)
    )
    try:
        return await asyncio.shield(placing)
    except asyncio.CancelledError:
        placing.add_done_callback(lambda done: _hand_back(fleet, request.request_id, done))
        raise


def _hand_back(
    fleet: ray.actor.ActorHandle,
    request_id: str,
    placing: asyncio.Future[tuple[str, ray.actor.ActorHandle]],
) -> None:
    if not placing.cancelled() and placing.exception() is None:
        fleet.finish.remote(request_id, placing.result()[0], None)
