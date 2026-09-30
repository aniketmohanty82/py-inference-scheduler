"""Per-request latency means over a time window, from a per-minute engine series.

    python3 hist_delta.py <scrape log with full vllm: series> <HH:MM:SS start> <HH:MM:SS end> [label]

Takes the snapshot at or before `start` and the one at or before `end` (UTC,
same day as the series), sums each engine's histogram `_sum` and `_count`
deltas, and prints mean seconds per request for the request-level
histograms plus iterations and tokens in the window. Engines whose block is
missing from either snapshot are skipped.
"""

import re
import sys
import time

path, t_start, t_end = sys.argv[1], sys.argv[2], sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else path

HISTS = [
    ("vllm:time_to_first_token_seconds", "time to first token"),
    ("vllm:request_queue_time_seconds", "queue wait"),
    ("vllm:request_prefill_time_seconds", "prefill"),
    ("vllm:request_decode_time_seconds", "decode"),
    ("vllm:time_per_output_token_seconds", "time per output token"),
    ("vllm:request_inference_time_seconds", "inference (prefill+decode)"),
    ("vllm:e2e_request_latency_seconds", "end to end"),
    ("vllm:request_prompt_tokens", "prompt tokens per request"),
    ("vllm:request_generation_tokens", "generated tokens per request"),
    ("vllm:iteration_tokens_total", "tokens per engine iteration"),
]
COUNTERS = [
    ("vllm:num_preemptions_total", "preemptions"),
    ("vllm:generation_tokens_total", "generated tokens"),
    ("vllm:prompt_tokens_total", "prompt tokens"),
]

snapshots: list[tuple[int, dict[str, dict[str, float]]]] = []
ts, port, blk = None, None, {}
for ln in open(path, errors="replace"):
    if ln.startswith("SNAP "):
        if ts is not None and blk:
            snapshots.append((ts, blk))
        ts, port, blk = int(ln.split()[1]), None, {}
    elif ln.startswith("=== PORT"):
        port = ln.split()[2]
        blk.setdefault(port, {})
    elif port and ln.startswith("vllm:"):
        name, _, val = ln.rpartition(" ")
        base = name.split("{")[0]
        if base.endswith(("_sum", "_count")) or any(base == c for c, _ in COUNTERS):
            try:
                blk[port][base] = blk[port].get(base, 0.0) + float(val)
            except ValueError:
                pass
if ts is not None and blk:
    snapshots.append((ts, blk))
if not snapshots:
    sys.exit("no snapshots")

day = time.strftime("%Y-%m-%d", time.gmtime(snapshots[-1][0]))


def epoch(hms: str) -> int:
    return int(time.mktime(time.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S")) - time.timezone)


def at(t: int):
    cands = [s for s in snapshots if s[0] <= t]
    return cands[-1] if cands else None


a, b = at(epoch(t_start)), at(epoch(t_end))
if a is None or b is None:
    sys.exit("window outside the series")
print(f"== {label}: {time.strftime('%H:%M:%S', time.gmtime(a[0]))} -> {time.strftime('%H:%M:%S', time.gmtime(b[0]))} UTC, engines {sorted(set(a[1]) & set(b[1]))}")
for base, name in HISTS:
    s = c = 0.0
    for p in set(a[1]) & set(b[1]):
        # An engine whose start snapshot lacks this histogram (older scraper
        # format) is skipped: treating the missing value as 0 would turn the
        # delta into the engine's whole history.
        if base + "_count" not in a[1][p] or base + "_count" not in b[1][p]:
            continue
        s += b[1][p][base + "_sum"] - a[1][p][base + "_sum"]
        c += b[1][p][base + "_count"] - a[1][p][base + "_count"]
    if c > 0:
        unit = "" if "tokens" in base else " s"
        print(f"  {name:28s} n={c:8.0f} mean={s / c:10.3f}{unit}")
for base, name in COUNTERS:
    d = sum(b[1][p].get(base, 0.0) - a[1][p].get(base, 0.0) for p in set(a[1]) & set(b[1]))
    print(f"  {name:28s} {d:,.0f}")
