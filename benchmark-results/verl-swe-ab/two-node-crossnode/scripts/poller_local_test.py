"""Local Ray check of the fleet actor: discovery through a fake balancer, the
background poller thread calling fake server actors, fleet() staleness and stats."""
import asyncio, sys, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import ray

@ray.remote
class FakeServer:
    def __init__(self, kv): self.kv = kv; self.calls = 0
    def get_routing_stats(self):
        self.calls += 1
        return {"num_waiting_reqs": 1, "num_running_reqs": 2, "kv": self.kv, "num_preempted": 0, "error": None}
    def set_kv(self, kv): self.kv = kv
    def n_calls(self): return self.calls

@ray.remote
class FakeBalancer:
    def __init__(self, servers): self._servers = servers; self._inflight = dict.fromkeys(servers, 0)
    def get_all_servers(self): return list(self._servers)
    def acquire_server(self, request_id):
        sid = min(self._inflight, key=lambda k: (self._inflight[k], list(self._servers).index(k)))
        self._inflight[sid] += 1
        return sid, self._servers[sid]
    def release_server(self, server_id): self._inflight[server_id] -= 1
    def loads(self): return dict(self._inflight)

def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

async def main():
    step("ray.init")
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=True)
    step("ray up")
    from integration.verl.shared_inflight import SharedInflightLedger, METRICS_INTERVAL_MS
    servers = {f"srv-{i}": FakeServer.remote(0.1 * i) for i in range(8)}
    lb = FakeBalancer.remote(servers)
    ledger = SharedInflightLedger()
    step("discover")
    t = time.perf_counter()
    handles = await ledger.discover_servers(lb, 8)
    print(f"discovered {len(handles)} servers in {1e3*(time.perf_counter()-t):.0f} ms; balancer loads after release: {ray.get(lb.loads.remote())}")
    step("sleep 350 ms then fleet()")
    await asyncio.sleep(0.35)
    f = await ledger.fleet()
    step("fleet returned")
    print(f"fleet after 350 ms: staleness={f['staleness']:.3f}s stats for srv-3={f['stats'].get('srv-3')} inflight={f['inflight']}")
    ray.get(servers["srv-3"].set_kv.remote(0.95))
    await asyncio.sleep(0.25)
    f2 = await ledger.fleet()
    ledger.increment("srv-3"); ledger.increment("srv-3"); ledger.decrement("srv-3")
    f3 = await ledger.fleet()
    calls = ray.get(servers["srv-3"].n_calls.remote())
    print(f"after set_kv 0.95 + 250 ms: srv-3 kv={f2['stats']['srv-3']['kv']} ; after +2-1 inflight: {f3['inflight']} ; srv-3 scraped {calls} times in ~0.7 s (interval {METRICS_INTERVAL_MS} ms)")
    ok = len(handles) == 8 and f["staleness"] < 0.3 and f2["stats"]["srv-3"]["kv"] == 0.95 and f3["inflight"].get("srv-3") == 1 and 4 <= calls <= 12 and all(v == 0 for v in ray.get(lb.loads.remote()).values())
    print("LOCAL POLLER TEST:", "PASS" if ok else "FAIL")
    ray.shutdown()
    return 0 if ok else 1

sys.exit(asyncio.run(main()))
