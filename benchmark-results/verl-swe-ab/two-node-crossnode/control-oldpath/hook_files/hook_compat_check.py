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
"""Hook compat check for the modern (verl v0.9.x) layout, GPU-free.

Runs on the Ray head pod: spins up fake rollout-server actors plus a REAL
verl GlobalRequestLoadBalancer, then drives InferenceSchedulerServerClient
through bootstrap -> scheduler routing -> generate -> release, with TWO
clients standing in for two AgentLoopWorkers. Verifies: full endpoint view
in both (re-drain), load-aware routing away from a saturated fake engine,
turn-to-turn affinity with the saturation filter as the only way off the
holder, local AND shared-ledger inflight returning to zero, LB counters
untouched.

Usage:
    ROUTER_CONFIG_PATH=configs/swe-backpressure.yaml \
    PYTHONPATH=/tmp/swe_repo:/tmp/swe_repo/src \
        python3 -m integration.verl.hook_compat_check
"""

from __future__ import annotations

import asyncio
import collections

import ray
from omegaconf import OmegaConf

# srv-0 looks loaded, srv-1 idle, srv-2 middling: a load-aware profile must
# route away from srv-0; a blind one would not.
LOADS = {"srv-0": (12, 20, 0.97), "srv-1": (0, 0, 0.05), "srv-2": (6, 3, 0.55)}


@ray.remote
class FakeServer:
    def __init__(self, name: str):  # noqa: ANN204
        self.name = name
        self.calls = 0
        self.load = LOADS[name]

    async def generate(self, **kwargs):  # noqa: ANN201
        self.calls += 1
        return {"token_ids": [1, 2, 3], "server": self.name}

    def set_load(self, running: int, waiting: int, kv: float) -> None:
        self.load = (running, waiting, kv)

    def get_routing_stats(self):  # noqa: ANN201
        return {
            "num_running_reqs": self.load[0],
            "num_waiting_reqs": self.load[1],
            "kv": self.load[2],
            "num_preempted": 0,
        }

    def get_calls(self):  # noqa: ANN201
        return self.calls


async def _parking_scenario(client, servers, turn) -> bool:
    """Park under fleet-wide saturation, admit when one engine recovers.

    Only when the profile has flow control: with every engine over the
    thresholds a request must PARK, not route; when one engine recovers the
    watcher must admit it there.
    """
    if not client.core.flow_control.has_plugins():
        print("flow control: no plugins in this profile, parking scenario skipped")
        return True
    for server in servers.values():
        await server.set_load.remote(12, 20, 0.99)
    parked = asyncio.ensure_future(turn("traj-park"))
    await asyncio.sleep(1.0)
    was_parked = not parked.done()
    await servers["srv-2"].set_load.remote(*LOADS["srv-2"])
    landed = await asyncio.wait_for(parked, timeout=15)
    for name, server in servers.items():
        await server.set_load.remote(*LOADS[name])
    print(f"flow control: parked={was_parked} then admitted to {landed}")
    return bool(was_parked and landed == "srv-2")


async def main() -> int:  # noqa: PLR0914
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)

    from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

    from integration.verl import verl_hook

    print("layout:", verl_hook._VERL_LAYOUT)
    assert verl_hook._VERL_LAYOUT == "modern", "expected modern layout on this verl build"  # noqa: S101

    servers = {f"srv-{i}": FakeServer.remote(f"srv-{i}") for i in range(3)}
    # verl 0.8 shipped the balancer pre-decorated; 0.9 ships a plain class that
    # verl wraps with ray.remote at instantiation. Handle both.
    lb_cls = (
        GlobalRequestLoadBalancer
        if hasattr(GlobalRequestLoadBalancer, "remote")
        else ray.remote(GlobalRequestLoadBalancer)
    )
    lb = lb_cls.remote(servers)

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"ignore_eos": False}}})
    client = verl_hook.InferenceSchedulerServerClient(config, load_balancer_handle=lb)

    # Two clients = two AgentLoopWorkers. The shared ledger must make both see
    # the same fleet q counts, and the re-drain must give both a full view even
    # though the second client bootstraps after the first has traffic in flight.
    client2 = verl_hook.InferenceSchedulerServerClient(config, load_balancer_handle=lb)

    shared_prefix = list(range(400))
    routed = collections.defaultdict(int)
    for i in range(12):
        c = client if i % 2 == 0 else client2
        out = await c.generate(
            request_id=f"traj-{i}",
            prompt_ids=shared_prefix + list(range(1000 + i * 50, 1000 + (i + 1) * 50)),
            sampling_params={"temperature": 1.0},
        )
        routed[out["server"]] += 1

    # Affinity: turn 2 of a request_id must return to the engine that served
    # turn 1; once that engine saturates the filter must move it; and after
    # the old holder recovers the NEXT turn must stay on the new engine (a
    # rendezvous "home" scorer would bounce it back).
    async def turn(rid: str) -> str:
        out = await client.generate(
            request_id=rid, prompt_ids=[*shared_prefix, 7], sampling_params={"temperature": 1.0}
        )
        return str(out["server"])

    first = await turn("traj-aff")
    second = await turn("traj-aff")
    await servers[first].set_load.remote(12, 20, 0.99)
    third = await turn("traj-aff")
    await servers[first].set_load.remote(*LOADS[first])
    fourth = await turn("traj-aff")
    affinity_ok = second == first and third != first and fourth == third
    print(
        f"affinity turns: {first} -> {second} -> (holder saturated) {third}"
        f" -> (holder recovered) {fourth}"
    )
    print(f"affinity counters: {client.core.affinity}")

    parked_ok = await _parking_scenario(client, servers, turn)

    n_endpoints = len(client.core.endpoints)
    n_endpoints2 = len(client2.core.endpoints)
    residual_inflight = sum(client.core.inflight_store.get(ep.name) for ep in client.core.endpoints)
    residual_inflight += sum(
        client2.core.inflight_store.get(ep.name) for ep in client2.core.endpoints
    )
    shared_snapshot = await client.core.shared_inflight.snapshot()
    shared_residual = sum((shared_snapshot or {}).values())
    lb_status = await lb.get_status.remote()

    print(f"endpoints bootstrapped: client1={n_endpoints} client2={n_endpoints2}")
    print(f"routing distribution (loaded srv-0 / idle srv-1 / mid srv-2): {dict(routed)}")
    print(f"scheduler inflight residual (local): {residual_inflight}")
    print(f"shared ledger residual: {shared_residual}  snapshot={shared_snapshot}")
    print(f"LB total_inflight after run: {lb_status['total_inflight']}")

    ok = (
        n_endpoints == len(servers)
        and n_endpoints2 == len(servers)  # both workers see the whole fleet
        and routed.get("srv-0", 0) == 0  # saturated engine (kv .97 / waiting 20) never chosen
        and routed.get("srv-1", 0) >= routed.get("srv-2", 0)  # idle engine preferred
        and residual_inflight == 0
        and shared_residual == 0  # fleet ledger returns to zero: every dispatch released
        and shared_snapshot is not None  # the ledger actor was reachable
        and lb_status["total_inflight"] == 0  # LB counters untouched by scheduler-routed traffic
        and affinity_ok  # stay with the holder; leave only when filtered; do not bounce back
        and client.core.affinity["kept"] == 2  # noqa: PLR2004 - traj-aff turns 2 and 4
        and client.core.affinity["moved"] == 1  # traj-aff turn 3
        and parked_ok  # flow control parks when saturated, admits on recovery
    )
    print("HOOK COMPAT CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
