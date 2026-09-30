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

"""Fleet state shared by every verl AgentLoopWorker.

verl fans a batch over several AgentLoopWorker actors (8 by default), each
holding its own _SchedulerCore. One named Ray actor therefore keeps what all
of them need to agree on: the fleet-wide in-flight count per engine, the
engine handles (verl's balancer registers them but only enumerates ids), and
the engines' routing metrics, polled in the background at a fixed interval
by the datalayer's MetricsPoller so that no scheduling decision scrapes an
engine. Workers mirror `fleet()` into their endpoint attributes on the same
interval; a worker's own dispatches keep its counts exact in between.
"""

from __future__ import annotations

import os
import uuid

import aiohttp
import ray

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.poller import MetricsPoller
from py_inference_scheduler.datalayer.metrics.verl.fetch_metrics import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint

_ACTOR_NAME = "rls_shared_inflight"
METRICS_INTERVAL_MS = int(os.environ.get("RLS_METRICS_INTERVAL_MS", "100"))


class _Ledger(InflightStore):
    def __init__(self, interval_ms: int) -> None:
        super().__init__()
        self._endpoints: list[Endpoint] = []
        self._poller = MetricsPoller(lambda: self._endpoints, self, _fetch, interval_ms=interval_ms)
        self._polling = False

    def discover_servers(
        self, balancer: ray.actor.ActorHandle, expected: int
    ) -> dict[str, ray.actor.ActorHandle]:
        """Recover the engine handles from the balancer, once for the fleet.

        Acquiring with unique ids and releasing only at the end visits every
        server in turn. 64 concurrent drains per worker skewed the balancer's
        counters into partial views (09-26); here it runs once, on an idle
        balancer, and re-runs only while engines are still registering.
        """
        known = {ep.name: ep.attributes["replica_obj"] for ep in self._endpoints}
        if len(known) < expected:
            acquired: list[str] = []
            for _ in range(max(1, expected) * 3):
                server_id, handle = ray.get(
                    balancer.acquire_server.remote(request_id=f"rls-discover-{uuid.uuid4().hex}")
                )
                acquired.append(server_id)
                known[server_id] = handle
                if len(known) >= expected:
                    break
            for server_id in acquired:
                balancer.release_server.remote(server_id=server_id)
            current = {ep.name: ep for ep in self._endpoints}
            self._endpoints = [
                current.get(sid)
                or Endpoint(name=sid, attributes={"replica_obj": handle, "routing_stats": {}})
                for sid, handle in known.items()
            ]
        if not self._polling:
            self._poller.start()
            self._polling = True
        return known

    def fleet(self) -> dict[str, object]:
        """Latest polled metrics, fleet in-flight counts and the poll's age."""
        return {
            "stats": {
                ep.name: dict(ep.attributes.get("routing_stats", {})) for ep in self._endpoints
            },
            "inflight": self.get_all(),
            "staleness": self._poller.staleness(),
        }


async def _fetch(ep: Endpoint, inflight: InflightStore, session: aiohttp.ClientSession) -> None:
    # verl's stats come over a Ray call to the server actor, so the poller's
    # HTTP session is unused here.
    await fetch_worker_metrics(ep, inflight)


_LedgerActor = ray.remote(_Ledger)


class SharedInflightLedger:
    """Client for the fleet actor, attached lazily and only under Ray.

    Constructing it outside a cluster (unit tests) is a no-op; every method
    then behaves as if the fleet were empty.
    """

    def __init__(self) -> None:
        self._actor: ray.actor.ActorHandle | None = None

    def _handle(self) -> ray.actor.ActorHandle | None:
        if self._actor is None and ray.is_initialized():
            # num_cpus=0: the ledger must never pend behind verl's CPU
            # reservations; a pending actor would stall every worker.
            self._actor = _LedgerActor.options(
                name=_ACTOR_NAME, get_if_exists=True, num_cpus=0
            ).remote(METRICS_INTERVAL_MS)
        return self._actor

    def increment(self, endpoint_name: str) -> None:
        actor = self._handle()
        if actor is not None:
            actor.increment.remote(endpoint_name)

    def decrement(self, endpoint_name: str) -> None:
        actor = self._handle()
        if actor is not None:
            actor.decrement.remote(endpoint_name)

    async def discover_servers(
        self, balancer: ray.actor.ActorHandle, expected: int
    ) -> dict[str, ray.actor.ActorHandle]:
        actor = self._handle()
        if actor is None:
            raise RuntimeError("shared ledger unavailable: cannot discover the engine set")
        return await actor.discover_servers.remote(balancer, expected)  # type: ignore[no-any-return]

    async def counts(self) -> dict[str, int]:
        """Fleet in-flight per engine, read at decision time (one small RPC)."""
        actor = self._handle()
        if actor is None:
            return {}
        return await actor.get_all.remote()  # type: ignore[no-any-return]

    async def fleet(self) -> dict[str, object]:
        actor = self._handle()
        if actor is None:
            return {"stats": {}, "inflight": {}, "staleness": float("inf")}
        return await actor.fleet.remote()  # type: ignore[no-any-return]
