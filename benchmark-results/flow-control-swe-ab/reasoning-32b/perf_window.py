"""Engine-side performance for one arm window, from the scraper tsv(s).

Counters (generation_tokens, prompt_tokens, queue_time sum/count, tpot
sum/count, preemptions) are POD-WIDE prometheus aggregates: every engine on
a pod reports the same value, so each pod contributes ONE delta, taken from
its lowest port. Gauges (kv, running) are per engine, so busy engine-seconds
sum over engines. Any column may be 'na' (e.g. a metric this vLLM version
does not export); a missing column never discards the row, it is simply
absent from that row's contribution.

usage: perf_window.py START END tsv [tsv...]   (window may wrap midnight)
"""

import sys

start, end = sys.argv[1], sys.argv[2]
files = sys.argv[3:]
INTERVAL = 15.0
COUNTERS = {
    "gen": "gen_tokens",
    "prompt": "prompt_tokens",
    "qsum": "q_sum",
    "qcnt": "q_count",
    "tsum": "tpot_sum",
    "tcnt": "tpot_count",
    "pre": "preempted",
}


def in_window(ts):
    if start <= end:
        return start <= ts <= end
    return ts >= start or ts <= end


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


totals = {k: 0.0 for k in COUNTERS}
have = {k: False for k in COUNTERS}
busy_s = 0.0
sat = 0
n = 0
kvs = []
for path in files:
    per_port = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header) or not in_window(f[0]):
                continue
            kv, run = num(f[idx["kv"]]), num(f[idx["running"]])
            if kv is None or run is None:
                continue  # engine not answering yet
            row = {"kv": kv, "run": run}
            for k, col in COUNTERS.items():
                row[k] = num(f[idx[col]]) if col in idx else None
            per_port.setdefault(f[idx["port"]], []).append(row)
    if not per_port:
        continue
    ref = per_port[min(per_port, key=int)]
    for k in COUNTERS:
        vals = [r[k] for r in ref if r[k] is not None]
        if len(vals) >= 2:
            totals[k] += vals[-1] - vals[0]
            have[k] = True
    for rows in per_port.values():
        for r in rows:
            n += 1
            kvs.append(r["kv"])
            if r["run"] > 0:
                busy_s += INTERVAL
            if r["kv"] >= 0.9:
                sat += 1

kvs.sort()
p = lambda q: kvs[min(int(len(kvs) * q), len(kvs) - 1)] if kvs else float("nan")
print(f"window {start}-{end}  samples={n}  busy engine-seconds={busy_s:.0f}")
print(f"  gen_tokens={totals['gen']:.0f}  prompt_tokens={totals['prompt']:.0f}  preemptions={totals['pre']:.0f}")
if busy_s and have["gen"]:
    print(f"  decode throughput = {totals['gen']/busy_s:.1f} tok per busy-engine-second")
if have["qcnt"] and totals["qcnt"]:
    print(f"  mean queue wait  = {totals['qsum']/totals['qcnt']:.3f} s over {totals['qcnt']:.0f} requests")
if have["tcnt"] and totals["tcnt"]:
    print(f"  mean tpot        = {1000*totals['tsum']/totals['tcnt']:.1f} ms")
else:
    print("  tpot: not exported by this vLLM (column na)")
if have["gen"] and totals["gen"]:
    print(f"  preempt per Mtok(gen) = {1e6*totals['pre']/totals['gen']:.1f}   prompt/gen ratio = {totals['prompt']/totals['gen']:.2f}")
print(f"  kv p50={p(0.5):.2f} p90={p(0.9):.2f} max={kvs[-1] if kvs else float('nan'):.2f}  sat>=0.9: {sat} ({100*sat/max(n,1):.1f}%)")
