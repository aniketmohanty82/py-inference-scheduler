import collections, contextlib, io, sys
sys.path.insert(0, "src")
from py_inference_scheduler.core.scheduler import Scheduler
from py_inference_scheduler.framework import Endpoint, LLMRequest
s = Scheduler()
def run(label, qs, bump):
    eps = [Endpoint(name=f"e{i}", attributes={"routing_stats": {"kv": 0.0, "num_waiting_reqs": 0, "num_running_reqs": 0, "num_preempted": 0}, "queue_len": q}) for i, q in enumerate(qs)]
    wins = collections.Counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for k in range(64):
            sel = s.run(LLMRequest(request_id=f"{label}-{k}", body=[1]), candidates=eps)
            wins[sel[0].endpoint.name] += 1
            if bump:
                sel[0].endpoint.attributes["queue_len"] += 1
    print(f"{label:52s} -> " + " ".join(f"{e.name}:{wins[e.name]:2d}" for e in eps))
run("static snapshot, all idle (exact tie, jitter decides)", [0]*8, False)
run("static snapshot, one engine 1 lower than the rest", [0,1,1,1,1,1,1,1], False)
run("static snapshot, other cores' herds visible (24,32)", [24,32,0,1,0,0,1,0], False)
run("queue_len bumped on dispatch (the fix), all idle", [0]*8, True)
run("queue_len bumped on dispatch (the fix), herds visible", [24,32,0,1,0,0,1,0], True)
