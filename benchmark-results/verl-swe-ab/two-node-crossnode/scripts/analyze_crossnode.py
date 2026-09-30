"""Cross-node run analysis: routing instruments + pull sources + flow control.

    python3 analyze_crossnode.py <harvest dir>   (sched2.sh output: fcstore.log,
        pullsrc_w*.log, fcstore_scrape_w*.log, workers.txt)
"""

import collections
import glob
import os
import re
import statistics
import sys

out = sys.argv[1]
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", open(os.path.join(out, "fcstore.log"), errors="replace").read())

# --- per-step verl metrics
KEYS = ["timing_s/gen", "timing_s/agent_loop/generate_sequences/mean", "timing_s/agent_loop/generate_sequences/max",
        "timing_s/agent_loop/tool_calls/mean", "num_turns/mean", "perf/throughput", "actor/entropy", "critic/score/mean"]
for i, line in enumerate(l for l in text.splitlines() if re.search(r"step:\d+ -", l)):
    vals = {k: re.search(re.escape(k) + r":(?:np\.\w+\()?([\d.]+)", line) for k in KEYS}
    print(f"step {i + 1}: " + " ".join(f"{k.split('/')[-1] if 'agent_loop' not in k else k.split('/')[2]}={float(m.group(1)):.1f}" for k, m in vals.items() if m))

# --- affinity / moves
aff = {}
for pid, k, m, f in re.findall(r"AFFINITY\[(\d+)\] kept=(\d+) moved=(\d+) fresh=(\d+)", text):
    aff[pid] = (int(k), int(m), int(f))
kept = sum(v[0] for v in aff.values()); moved = sum(v[1] for v in aff.values()); fresh = sum(v[2] for v in aff.values())
moves = re.findall(r"MOVE\[\d+\] rid=(\S+) from=([\d.]+):(\d+) to=([\d.]+):(\d+) cross_node=(\d)", text)
cross = [m for m in moves if m[5] == "1"]
print(f"\naffinity: kept {kept} moved {moved} fresh {fresh} -> kept share {kept / max(1, kept + moved):.3f}")
print(f"moves: {len(moves)} total, cross-node {len(cross)} ({100 * len(cross) / max(1, len(moves)):.0f}%), "
      f"trajectories that ever crossed: {len({m[0] for m in cross})}")
flows = collections.Counter((m[1], m[3]) for m in cross)
print("cross-node flows host->host:", dict(flows))

# --- flow control
parks = re.findall(r"FLOWCONTROL park: all (\d+) endpoints saturated", text)
drops = re.findall(r"FLOWCONTROL drop (\d+)/(\d+)", text)
print(f"\nflow control: parks {len(parks)}, partial drops {len(drops)}"
      + (f", median engines dropped per decision {statistics.median(int(d[0]) for d in drops)}" if drops else ""))

# --- decisions per engine and per host
sel = re.findall(r"Selected endpoint \[ScoredEndpoint\(endpoint=Endpoint\(name='([\d.]+):(\d+)'", text)
by_host = collections.Counter(h for h, _ in sel); by_eng = collections.Counter(f"{h}:{p}" for h, p in sel)
print(f"\ndecisions: {len(sel)} | per host {dict(by_host)} | per engine min/max {min(by_eng.values()) if by_eng else 0}/{max(by_eng.values()) if by_eng else 0}")

# --- FLEET: 8-engine snapshots
rows = []
for _pid, body in re.findall(r"FLEET\[(\d+)\] (.+)", text):
    parts = re.findall(r"(\d+)=kv([\d.]+)/w(\d+)/r(\d+)/p(\d+)/q(\d+)", body)
    if len(parts) >= 8:
        rows.append([(p[0], float(p[1]), int(p[2]), int(p[3]), int(p[4])) for p in parts])
busy = [r for r in rows if sum(v[2] + v[3] for v in r) >= 40]
if busy:
    kv = sorted(v[1] for r in busy for v in r)
    idle = sum(1 for r in busy if min(v[3] for v in r) == 0 and max(v[2] for v in r) > 0)
    print(f"\nFLEET 8-engine snapshots {len(rows)} busy {len(busy)} | kv median {kv[len(kv) // 2]:.2f} p75 {kv[int(.75 * len(kv))]:.2f} | idle-while-queued {idle}/{len(busy)} | preemptions {sum(v[4] for v in rows[-1])}")

# --- PULLSRC per worker
for path in sorted(glob.glob(os.path.join(out, "pullsrc_w*.log"))):
    lines = open(path, errors="replace").read().splitlines()
    last = [ln for ln in lines if "cum_cross=" in ln][-1:]
    if last:
        m = re.search(r"local=(\S+) .*cum_batches=(\d+) cum_keys=(\d+) cum_cross=(\d+) cum_unknown=(\d+)", last[0])
        if m:
            print(f"{os.path.basename(path)}: host {m.group(1)}: {m.group(2)} load batches, {m.group(3)} keys, "
                  f"{m.group(4)} from the other host ({100 * int(m.group(4)) / max(1, int(m.group(3))):.0f}%), {m.group(5)} unknown")

# --- engine by_source per worker
def by_source(path: str) -> dict[str, float]:
    tot: dict[str, float] = collections.defaultdict(float); port = None; last: dict[str, dict[str, float]] = {}
    for ln in open(path, errors="replace"):
        if ln.startswith("=== PORT"):
            port = ln.split()[2]; last.setdefault(port, {})
        elif port and ln.startswith("vllm:prompt_tokens_by_source_total"):
            src = re.search(r'source="([a-z_]+)"', ln).group(1); last[port][src] = float(ln.rsplit(" ", 1)[1])
    for d in last.values():
        for s, v in d.items(): tot[s] += v
    return dict(tot)
for path in sorted(glob.glob(os.path.join(out, "fcstore_scrape_w*.log"))):
    t = by_source(path); s = sum(t.values()) or 1
    print(f"{os.path.basename(path)}: " + " ".join(f"{k}={v:,.0f} ({100 * v / s:.1f}%)" for k, v in sorted(t.items())))
