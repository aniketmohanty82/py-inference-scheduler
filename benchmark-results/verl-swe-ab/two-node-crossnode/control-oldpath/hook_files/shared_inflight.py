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
"""Fleet-wide inflight counts shared by every verl AgentLoopWorker.

verl fans the batch over several AgentLoopWorker actors (8 by default) and
the hook builds an independent _SchedulerCore in each, so a per-worker
InflightStore sees only that worker's slice of the requests the fleet has
admitted. queue_len published from the local store therefore understates
engine load by the worker fan-out, which misleads every consumer of the
attribute -- least_queue scoring most of all. This ledger mirrors the counts
into one named Ray actor so every worker reads the same fleet totals.

Increments and decrements are fire-and-forget actor calls; Ray executes tasks
from one caller in submission order, so a worker's own snapshot always
reflects its earlier increments. Cross-worker skew is one RPC (~1 ms).
"""

from __future__ import annotations

import uuid

import ray

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore

_ACTOR_NAME = "rls_shared_inflight"


class _Ledger(InflightStore):
    """The fleet ledger plus one-time engine discovery.

    verl's balancer holds the server handles but enumerates ids only, so the
    handles are recovered by acquiring with unique ids until every server has
    answered. Done here, once for the fleet: 64 concurrent drains per worker
    skewed the balancer's counters into partial views (09-26).
    """

    def __init__(self) -> None:
        super().__init__()
        self._servers: dict[str, ray.actor.ActorHandle] = {}

    def discover_servers(
        self, balancer: ray.actor.ActorHandle, expected: int
    ) -> dict[str, ray.actor.ActorHandle]:
        if len(self._servers) >= expected:
            return dict(self._servers)
        acquired: list[str] = []
        for _ in range(max(1, expected) * 3):
            server_id, handle = ray.get(
                balancer.acquire_server.remote(request_id=f"rls-discover-{uuid.uuid4().hex}")
            )
            acquired.append(server_id)
            self._servers[server_id] = handle
            if len(self._servers) >= expected:
                break
        for server_id in acquired:
            balancer.release_server.remote(server_id=server_id)
        return dict(self._servers)


_LedgerActor = ray.remote(_Ledger)


class SharedInflightLedger:
    """InflightStore mirror backed by a named, job-scoped Ray actor.

    Lazy: the actor is attached on first use and only when Ray is
    initialized, so constructing the ledger outside a cluster (unit tests)
    is a no-op. If the actor becomes unreachable the ledger degrades to
    snapshot() -> None and callers fall back to their local counts.
    """

    def __init__(self) -> None:
        self._actor: ray.actor.ActorHandle | None = None
        self._warned = False

    def _handle(self) -> ray.actor.ActorHandle | None:
        if self._actor is None and ray.is_initialized():
            # num_cpus=0: the ledger must never pend behind verl's CPU
            # reservations -- a pending actor would hang every metric refresh.
            self._actor = _LedgerActor.options(
                name=_ACTOR_NAME, get_if_exists=True, num_cpus=0
            ).remote()
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

    async def snapshot(self) -> dict[str, int] | None:
        """Fleet-wide counts, or None when no shared ledger is reachable."""
        actor = self._handle()
        if actor is None:
            return None
        try:
            return await actor.get_all.remote()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            if not self._warned:
                self._warned = True
                # print(): only actor stdout reaches the Ray driver log.
                print("FLOWCONTROL shared inflight snapshot failed; using local counts")
            return None
