"""Routing instruments of the scheduler arm, from the verl driver log alone.

The hook prints three kinds of lines from every AgentLoopWorker:
  FLEET[pid]    one line per 15 s: every engine's kv / waiting / running /
                preemptions / fleet-wide in-flight, read together
  AFFINITY[pid] cumulative kept / moved / fresh turns for that worker
  "Selected endpoint ..." one line per routing decision (framework print)

    python3 analyze_routing.py sched_driver.log.gz [--engines 4]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--engines", type=int, default=4)
ap.add_argument("--busy", type=int, default=20, help="min fleet running+waiting for busy")
ap.add_argument("--kv", type=float, default=0.98, help="saturation filter kv threshold")
ap.add_argument("--waiting", type=int, default=8, help="saturation filter waiting threshold")
a = ap.parse_args()
KV_HIGH = 0.95  # the baseline profile's per-engine p75, reported for comparability

opener = gzip.open if a.log.endswith(".gz") else open
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", opener(a.log, "rt", errors="replace").read())

aff: dict[str, tuple[int, int, int]] = {}
for pid, k, m, f in re.findall(r"AFFINITY\[(\d+)\] kept=(\d+) moved=(\d+) fresh=(\d+)", text):
    aff[pid] = (int(k), int(m), int(f))
kept = sum(v[0] for v in aff.values())
moved = sum(v[1] for v in aff.values())
fresh = sum(v[2] for v in aff.values())
print(
    f"affinity: kept {kept} moved {moved} fresh {fresh} "
    f"-> kept share {kept / max(1, kept + moved):.3f}, moved share {moved / max(1, kept + moved):.3f}"
)

sel = collections.Counter(
    re.findall(r"Selected endpoint \[ScoredEndpoint\(endpoint=Endpoint\(name='[\d.]+:(\d+)'", text)
)
tot = sum(sel.values())
print("decisions per engine:", {k: f"{v} ({100 * v / tot:.1f}%)" for k, v in sorted(sel.items())})

rows = []
for _pid, body in re.findall(r"FLEET\[(\d+)\] (.+)", text):
    parts = re.findall(r"(\d+)=kv([\d.]+)/w(\d+)/r(\d+)/p(\d+)/q(\d+)", body)
    if len(parts) == a.engines:
        rows.append({
            p[0]: (float(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])) for p in parts
        })
busy = [r for r in rows if sum(v[1] + v[2] for v in r.values()) >= a.busy]
print(f"FLEET snapshots: {len(rows)} busy: {len(busy)}")

idle = sum(
    1 for r in busy if min(v[2] for v in r.values()) == 0 and max(v[1] for v in r.values()) > 0
)
print(f"idle-while-queued: {idle} / {len(busy)}")

spread = sorted(
    (max(v[2] for v in r.values()) - min(v[2] for v in r.values()))
    / max(1e-9, sum(v[2] for v in r.values()) / a.engines)
    for r in busy
)
kv = sorted(v[0] for r in busy for v in r.values())
print(
    f"running spread/mean: median {spread[len(spread) // 2]:.2f} p90 {spread[int(0.9 * len(spread))]:.2f} | "
    f"kv median {kv[len(kv) // 2]:.2f} p75 {kv[int(0.75 * len(kv))]:.2f} share>={KV_HIGH} {sum(x >= KV_HIGH for x in kv) / len(kv):.2f}"
)

over = lambda v: v[1] >= a.waiting or v[0] >= a.kv  # noqa: E731
pin = sum(1 for r in busy if all(over(v) for v in r.values()))
mixed = sum(
    1 for r in busy if any(over(v) for v in r.values()) and not all(over(v) for v in r.values())
)
print(
    f"filter state over busy snapshots: all-over (pin) {pin} mixed {mixed} none-over {len(busy) - pin - mixed}"
)

ident = sum(1 for r in busy if len({(v[1], v[2]) for v in r.values()}) == 1)
ratio = statistics.median(
    sum(v[4] for v in r.values()) / sum(v[1] + v[2] for v in r.values()) for r in busy
)
print(
    f"per-engine attribution: identical (waiting,running) on all engines in {ident}/{len(busy)} busy lines | "
    f"ledger q/(running+waiting) median {ratio:.2f}"
)
if rows:
    last = rows[-1]
    print(
        "preemptions per engine (cumulative):",
        {k: v[3] for k, v in last.items()},
        "total",
        sum(v[3] for v in last.values()),
    )
print(
    "fallbacks to verl LB:",
    len(re.findall(r"falling back to verl global LB", text)),
    "| engine fatal errors:",
    len(re.findall(r"EngineCore encountered a fatal error", text)),
)
