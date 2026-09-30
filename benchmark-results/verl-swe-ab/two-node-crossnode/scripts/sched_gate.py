"""Gate for the scheduler arm: are the routing instruments live and honest?

Reads a driver log and checks the lines the hook prints (RLS[pid] / FLEET[pid]
/ AFFINITY[pid] / "Selected endpoint"). Exit 0 = every check passed. The point
is to fail a run that "works" while routing blind. Failure modes this has
caught so far: 1/8th inflight views (private ledgers), partial endpoint views
pinning a worker to one engine, and a shared PROMETHEUS_MULTIPROC_DIR making
all four engines report the same metrics (smoke 1, 2026-09-23).

    python3 sched_gate.py <driver.log> [--engines 4] [--max-share 0.45]
"""

import argparse
import collections
import re
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--engines", type=int, default=4)
ap.add_argument("--max-share", type=float, default=0.45)
ap.add_argument("--min-kept", type=float, default=0.6)
a = ap.parse_args()

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
text = ANSI.sub("", open(a.log, errors="replace").read())

fails: list[str] = []
def check(ok: bool, msg: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        fails.append(msg)

# 1. every AgentLoopWorker built a core WITH the shared ledger
cores = set(re.findall(r"RLS\[(\d+)\] core init: shared inflight ledger enabled", text))
check(len(cores) >= 1, f"cores initialised with shared ledger: {len(cores)} worker pid(s)")

# 2. every worker's endpoint view covers the whole fleet (re-drain worked)
views = re.findall(r"RLS\[(\d+)\] endpoint view: (\d+)/(\d+) servers", text)
final_view: dict[str, tuple[int, int]] = {}
for pid, seen, total in views:
    final_view[pid] = (int(seen), int(total))
full = [pid for pid, (s, t) in final_view.items() if s == t == a.engines]
check(bool(final_view) and len(full) == len(final_view),
      f"endpoint view {a.engines}/{a.engines} in every worker: {len(full)}/{len(final_view)} pids")

# 3. FLEET lines parsed into per-engine tuples
fleet = re.findall(r"FLEET\[(\d+)\] (.+)", text)
check(len(fleet) >= 5, f"FLEET snapshots: {len(fleet)}")
rows = []  # (pid, [(port, kv, w, r, p, q)])
for pid, body in fleet:
    parts = re.findall(r"(\d+)=kv([\d.]+)/w(\d+)/r(\d+)/p(\d+)/q(\d+)", body)
    if len(parts) == a.engines:
        rows.append((pid, [(p[0], float(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])) for p in parts]))
busy = [(pid, e) for pid, e in rows if sum(x[2] + x[3] for x in e) >= 20]
kv_max = max((x[1] for _, e in rows for x in e), default=0.0)
r_max = max((x[3] for _, e in rows for x in e), default=0)
w_max = max((x[2] for _, e in rows for x in e), default=0)
check(kv_max > 0.0 and r_max > 0, f"scorer inputs populated: kv max {kv_max:.2f}, running max {r_max}, waiting max {w_max}")

# 4. per-engine attribution. With one shared multiprocess metrics dir every
#    /metrics port serves the same aggregate, and the hook's simultaneous
#    scrape then shows identical (waiting, running) on all engines: 94% of
#    busy lines in smoke 1. Real per-engine registries almost never tie.
ident = sum(1 for _, e in busy if len({(x[2], x[3]) for x in e}) == 1)
share_ident = ident / len(busy) if busy else 1.0
check(bool(busy) and share_ident <= 0.2,
      f"per-engine attribution: identical (waiting,running) on all engines in {ident}/{len(busy)} busy lines ({share_ident:.0%})")

# 5. the shared ledger. q is dispatches in flight from acquire to release;
#    summed over engines it must match the engines' own running+waiting.
#    A private per-worker ledger shows ~1/N of that (8 workers -> 0.13).
ratios = [sum(x[5] for x in e) / sum(x[2] + x[3] for x in e) for _, e in busy]
med = statistics.median(ratios) if ratios else 0.0
pids_seen = {pid for pid, _ in busy}
check(bool(ratios) and 0.8 <= med <= 1.25 and len(pids_seen) >= 2,
      f"shared ledger: median q_sum/(running+waiting) {med:.2f} over {len(ratios)} busy lines from {len(pids_seen)} pids")

# 6. routing decisions land on every engine and none dominates
sel = re.findall(r"Selected endpoint \[ScoredEndpoint\(endpoint=Endpoint\(name='([^']+)'", text)
if not sel:
    sel = re.findall(r"Selected endpoint.*?name='([^']+)'", text)
dist = collections.Counter(sel)
share = (max(dist.values()) / sum(dist.values())) if dist else 1.0
check(len(dist) == a.engines, f"decisions reached {len(dist)}/{a.engines} engines: {dict(dist)}")
check(share <= a.max_share, f"top engine share {share:.0%} <= {a.max_share:.0%}")

# 7. affinity: multi-turn continuity. kept = turn landed on the engine that
#    ran the previous turn; moved = the filter took the holder off the
#    ballot and the turn paid a full-context prefill elsewhere.
aff_last: dict[str, tuple[int, int, int]] = {}
for pid, k, m, f in re.findall(r"AFFINITY\[(\d+)\] kept=(\d+) moved=(\d+) fresh=(\d+)", text):
    aff_last[pid] = (int(k), int(m), int(f))
kept = sum(v[0] for v in aff_last.values()); moved = sum(v[1] for v in aff_last.values()); fresh = sum(v[2] for v in aff_last.values())
kept_share = kept / (kept + moved) if kept + moved else 0.0
check(kept + moved > 0 and kept_share >= a.min_kept,
      f"affinity: kept {kept} moved {moved} fresh {fresh} -> kept share {kept_share:.0%} >= {a.min_kept:.0%} ({len(aff_last)} pids)")

# 8. no silent fallback to verl's balancer
fb = len(re.findall(r"returned no endpoints, falling back to verl global LB", text))
check(fb == 0, f"fallbacks to verl LB: {fb}")

print("\nSCHED GATE:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
