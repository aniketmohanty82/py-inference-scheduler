"""Per-engine load split over a window: prompt/generation token and request deltas per port.

    python3 per_port_load.py <scrape log> <HH:MM:SS start> <HH:MM:SS end>

Works with the counter-only (v2) and full (v3) scrape formats; request
counts need v3 (e2e histogram _count).
"""
import sys, time
path, t_start, t_end = sys.argv[1:4]
snaps = []; ts = None; port = None; blk = {}
for ln in open(path, errors="replace"):
    if ln.startswith("SNAP "):
        if ts is not None and blk: snaps.append((ts, blk))
        ts, port, blk = int(ln.split()[1]), None, {}
    elif ln.startswith("=== PORT"):
        port = ln.split()[2]; blk.setdefault(port, {})
    elif port and ln.startswith(("vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:e2e_request_latency_seconds_count", "vllm:num_preemptions_total")):
        name, _, val = ln.rpartition(" ")
        blk[port][name.split("{")[0]] = float(val)
if ts is not None and blk: snaps.append((ts, blk))
day = time.strftime("%Y-%m-%d", time.gmtime(snaps[-1][0]))
ep = lambda hms: int(time.mktime(time.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S")) - time.timezone)
at = lambda t: [s for s in snaps if s[0] <= t][-1]
a, b = at(ep(t_start)), at(ep(t_end))
tot_p = tot_r = 0.0; rows = []
for p in sorted(set(a[1]) & set(b[1])):
    d = lambda k: b[1][p].get(k, 0.0) - a[1][p].get(k, 0.0)
    rows.append((p, d("vllm:prompt_tokens_total"), d("vllm:generation_tokens_total"), d("vllm:e2e_request_latency_seconds_count"), d("vllm:num_preemptions_total")))
    tot_p += rows[-1][1]; tot_r += rows[-1][3]
print(f"{path} {time.strftime('%H:%M:%S', time.gmtime(a[0]))}->{time.strftime('%H:%M:%S', time.gmtime(b[0]))}: " + "  ".join(f"{p}: {pt/1e6:.1f}M prompt ({100*pt/max(tot_p,1):.0f}%) {gt/1e3:.0f}k gen {int(rq)} req {int(pe)} pre" for p, pt, gt, rq, pe in rows) + f"  | node total {tot_p/1e6:.1f}M prompt, {int(tot_r)} req")
