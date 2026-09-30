"""Per-step tables for a two-node fcstore run (sched2.sh harvest).

    python3 harvest_x2.py <harvest dir> <driver ticks file>

Reads fcstore.log (verl step lines + hook prints), pullsrc_w*.log and
snap13_full_w*.log (the full per-minute engine series fetched from each
worker's /tmp/snap13.log), plus the driver's tick file for the wall-clock
time at which each step finished. Prints markdown: verl metrics per step,
routing and pull activity per step, engine token sources per step and node.
"""

import collections
import glob
import os
import re
import statistics
import sys
import time

out, ticks_path = sys.argv[1], sys.argv[2]
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", open(os.path.join(out, "fcstore.log"), errors="replace").read())
lines = text.splitlines()


def num(s: str) -> float:
    return float(s)


# ---- step boundaries in the log (line index of each "step:N - global_seqlen" line)
step_lines: dict[int, int] = {}
metrics: dict[int, dict[str, float]] = {}
for i, ln in enumerate(lines):
    m = re.search(r"step:(\d+) - (global_seqlen.*)", ln)
    if m:
        s = int(m.group(1))
        step_lines[s] = i
        metrics[s] = {k: num(v) for k, v in re.findall(r"([\w/.@-]+):(?:np\.\w+\()?(-?[\d.]+(?:e-?\d+)?)\)?", m.group(2))}
steps = sorted(step_lines)
if not steps:
    sys.exit("no step lines")

# ---- wall-clock end of each step from the driver ticks ("[HH:MM:SS] ... steps=N/")
tick_end: dict[int, str] = {}
launch_ts = None
for ln in open(ticks_path, errors="replace"):
    m = re.match(r"\[(\d\d:\d\d:\d\d)\].*steps=(\d+)(?:/|\s|$)", ln)
    if m:
        s = int(m.group(2))
        if s >= 1 and s not in tick_end:
            tick_end[s] = m.group(1)
    m2 = re.match(r"\[(\d\d:\d\d:\d\d)\] launched", ln)
    if m2:
        launch_ts = m2.group(1)

# ---- verl metrics table
KEYS = [
    ("timing_s/gen", "rollout time/step (timing_s/gen)", "{:,.0f}"),
    ("timing_s/agent_loop/generate_sequences/mean", "sampling per trajectory, mean (s)", "{:.1f}"),
    ("timing_s/agent_loop/generate_sequences/max", "sampling per trajectory, max (s)", "{:.1f}"),
    ("timing_s/agent_loop/tool_calls/mean", "tool time per trajectory, mean (s)", "{:.1f}"),
    ("timing_s/agent_loop/tool_calls/max", "tool time per trajectory, max (s)", "{:.1f}"),
    ("num_turns/mean", "num_turns/mean", "{:.2f}"),
    ("response_length/mean", "response_length/mean", "{:,.0f}"),
    ("critic/score/mean", "critic/score/mean", "{:.4f}"),
    ("actor/entropy", "actor/entropy", "{:.4f}"),
    ("actor/grad_norm", "actor/grad_norm", "{:.4f}"),
    ("training/rollout_probs_diff_mean", "rollout vs actor logprob diff, mean", "{:.4f}"),
    ("timing_s/old_log_prob", "timing_s/old_log_prob", "{:.0f}"),
    ("timing_s/update_actor", "timing_s/update_actor", "{:.0f}"),
    ("timing_s/step", "timing_s/step", "{:,.0f}"),
    ("perf/throughput", "perf/throughput (tok/s/GPU)", "{:.0f}"),
]
print("## verl metrics per step\n")
print("| metric | " + " | ".join(str(s) for s in steps) + " | mean |")
print("|---|" + "---|" * (len(steps) + 1))
for key, label, fmt in KEYS:
    vals = [metrics[s].get(key) for s in steps]
    if all(v is None for v in vals):
        continue
    cells = [fmt.format(v) if v is not None else "-" for v in vals]
    have = [v for v in vals if v is not None]
    print(f"| `{label}` | " + " | ".join(cells) + f" | {fmt.format(statistics.mean(have))} |")
print(f"| `step finished (UTC)` | " + " | ".join(tick_end.get(s, "-") for s in steps) + " | |")

# ---- per-step routing / flow control / pull counts from the hook prints
def seg(s: int) -> list[str]:
    lo = step_lines[steps[steps.index(s) - 1]] + 1 if steps.index(s) > 0 else 0
    return lines[lo : step_lines[s] + 1]


def pull_cum(seg_lines: list[str], series: dict[tuple[str, str], tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Advance the per-rank PULLSRC counters seen in this segment; return per-host sums.

    Every TP rank of every engine keeps its own cumulative (keys, cross) pair and
    prints it on sampled batches, so a per-host figure has to sum the latest
    value of each rank's series, carried forward across steps, rather than take
    a max over whichever ranks happened to print in the segment.
    """
    for ln in seg_lines:
        m = re.search(r"pid=(\d+)\) PULLSRC local=(\S+) .*cum_keys=(\d+) cum_cross=(\d+)", ln)
        if m:
            series[(m.group(2), m.group(1))] = (int(m.group(3)), int(m.group(4)))
    per_host: dict[str, tuple[int, int]] = {}
    for (host, _pid), (keys, cross) in series.items():
        k0, x0 = per_host.get(host, (0, 0))
        per_host[host] = (k0 + keys, x0 + cross)
    return per_host


print("\n## routing, flow control and tier pulls per step\n")
hosts = sorted({m for m in re.findall(r"PULLSRC local=(\S+)", text)})
hdr = ["decisions", "turns kept home", "moves", "cross-node moves", "trajectories that crossed", "flow-control parks", "flow-control partial drops", "peak running (8 engines)", "busy snapshots kv median"]
hdr += [f"keys pulled by {h} (rank sum / TP)" for h in hosts] + [f"of which from the other node ({h})" for h in hosts]
print("| step | " + " | ".join(hdr) + " |")
print("|---|" + "---|" * len(hdr))
prev_pull: dict[str, tuple[int, int]] = {}
pull_series: dict[tuple[str, str], tuple[int, int]] = {}  # (host, rank pid) -> last cumulative (keys, cross)
TP = 2  # both TP ranks of an engine pull the same keys (their own shards), so a per-node key count is the rank sum / TP
last_kept: dict[str, int] = {}  # AFFINITY kept= is cumulative per hook pid
for s in steps:
    sl = seg(s)
    body = "\n".join(sl)
    decisions = len(re.findall(r"Selected endpoint", body))
    moves = re.findall(r"MOVE\[\d+\] rid=(\S+) from=([\d.]+):\d+ to=([\d.]+):\d+ cross_node=(\d)", body)
    cross = [m for m in moves if m[3] == "1"]
    parks = len(re.findall(r"FLOWCONTROL park", body))
    drops = len(re.findall(r"FLOWCONTROL drop", body))
    before = sum(last_kept.values())
    for pid, k in re.findall(r"AFFINITY\[(\d+)\] kept=(\d+)", body):
        last_kept[pid] = int(k)
    kept = sum(last_kept.values()) - before
    fleet = []
    for m in re.finditer(r"FLEET\[\d+\] (.+)", body):
        parts = re.findall(r"=kv([\d.]+)/w(\d+)/r(\d+)/p(\d+)/q(\d+)", m.group(1))
        if len(parts) >= 8:
            fleet.append([(float(p[0]), int(p[1]), int(p[2])) for p in parts])
    peak_run = max((sum(v[2] for v in row) for row in fleet), default=0)
    busy = [row for row in fleet if sum(v[1] + v[2] for v in row) >= 40]
    kv_med = statistics.median(v[0] for row in busy for v in row) if busy else float("nan")
    pull = pull_cum(sl, pull_series)
    cells = [decisions, kept, len(moves), len(cross), len({m[0] for m in cross}), parks, drops, peak_run, f"{kv_med:.2f} ({len(busy)}/{len(fleet)})"]
    for h in hosts:
        now_k = pull.get(h, prev_pull.get(h, (0, 0)))[0]
        cells.append(f"{(now_k - prev_pull.get(h, (0, 0))[0]) // TP:,}")
    for h in hosts:
        now_k, now_x = pull.get(h, prev_pull.get(h, (0, 0)))
        dk = now_k - prev_pull.get(h, (0, 0))[0]
        dx = now_x - prev_pull.get(h, (0, 0))[1]
        cells.append(f"{dx // TP:,} ({100 * dx / dk:.0f}%)" if dk else "0")
    for h, v in pull.items():
        prev_pull[h] = v
    print(f"| {s} | " + " | ".join(str(c) for c in cells) + " |")

# ---- engine token sources per step from the per-minute series
def parse_series(path: str) -> list[tuple[int, dict[str, dict[str, float]]]]:
    """[(ts, {port: {metric-key: value}})] for snapshots that carry engine blocks."""
    series = []
    ts, port, blk = None, None, {}
    for ln in open(path, errors="replace"):
        if ln.startswith("SNAP "):
            if ts is not None and blk:
                series.append((ts, blk))
            ts, port, blk = int(ln.split()[1]), None, {}
        elif ln.startswith("=== PORT"):
            port = ln.split()[2]
            blk.setdefault(port, {})
        elif port and ln.startswith("vllm:"):
            name, _, val = ln.rpartition(" ")
            if name.startswith("vllm:prompt_tokens_by_source_total"):
                src = re.search(r'source="([a-z_]+)"', name)
                if src:
                    blk[port]["src:" + src.group(1)] = float(val)
            elif name.startswith(("vllm:num_preemptions_total", "vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")):
                blk[port][name.split("{")[0]] = float(val)
    if ts is not None and blk:
        series.append((ts, blk))
    return series


def hms_to_epoch(hms: str, ref_epoch: int) -> int:
    """Driver ticks are UTC HH:MM:SS of the run day; anchor to the series' day."""
    day = time.strftime("%Y-%m-%d", time.gmtime(ref_epoch))
    return int(time.mktime(time.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S")) - time.timezone)


series_files = sorted(glob.glob(os.path.join(out, "snap13_full_w*.log")))
if series_files and launch_ts:
    print("\n## engine token sources per step and node (deltas of the per-minute series)\n")
    print("| step | node | tokens from tier | local prefix hits | recomputed | tier share | preemptions |")
    print("|---|---|---|---|---|---|---|")
    for path in series_files:
        series = parse_series(path)
        if not series:
            continue
        ref = series[-1][0]
        t_launch = hms_to_epoch(launch_ts, ref)
        run_series = [(t, b) for t, b in series if t >= t_launch]
        if not run_series:
            continue

        def at(t_end: int):
            cands = [(t, b) for t, b in run_series if t <= t_end]
            return cands[-1][1] if cands else None

        def totals(b):
            tot = collections.Counter()
            for port, d in b.items():
                for k, v in d.items():
                    if k.startswith("src:") or k == "vllm:num_preemptions_total":
                        tot[k] += v
            return tot

        prev = collections.Counter()
        node = os.path.basename(path).replace("snap13_full_", "").replace(".log", "")
        for s in steps:
            if s not in tick_end:
                continue
            b = at(hms_to_epoch(tick_end[s], ref) + 90)
            if b is None:
                continue
            cur = totals(b)
            d = cur - prev
            tier, hit, comp = d["src:external_kv_transfer"], d["src:local_cache_hit"], d["src:local_compute"]
            tot = tier + hit + comp
            print(f"| {s} | {node} | {tier:,.0f} | {hit:,.0f} | {comp:,.0f} | {100 * tier / tot if tot else 0:.1f}% | {d['vllm:num_preemptions_total']:.0f} |")
            prev = cur
