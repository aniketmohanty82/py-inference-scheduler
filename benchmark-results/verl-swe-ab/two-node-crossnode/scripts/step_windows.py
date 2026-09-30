"""Print rollout windows (HH:MM:SS start end) found in a per-minute engine scrape log.

A window starts at the last idle snapshot before engines report running or
waiting requests, and ends at the first snapshot after which they stay idle
for 3 minutes. Feed the pairs to hist_delta.py.
"""
import sys, time
snaps = []; ts = None; load = 0.0; seen = False
for ln in open(sys.argv[1], errors="replace"):
    if ln.startswith("SNAP "):
        if ts is not None: snaps.append((ts, load, seen))
        ts = int(ln.split()[1]); load = 0.0; seen = False
    elif ln.startswith("=== PORT"): seen = True
    elif ln.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting{")):
        load += float(ln.rsplit(" ", 1)[1])
if ts is not None: snaps.append((ts, load, seen))
snaps = [s for s in snaps if s[2]]
i = 0
while i < len(snaps):
    if snaps[i][1] > 0:
        start = snaps[max(i - 1, 0)][0]
        j = i
        while j < len(snaps):
            if snaps[j][1] == 0 and all(s[1] == 0 for s in snaps[j:j + 3]):
                break
            j += 1
        end = snaps[min(j, len(snaps) - 1)][0]
        f = lambda t: time.strftime("%H:%M:%S", time.gmtime(t))
        print(f(start), f(end + 30))
        i = j + 3
    else:
        i += 1
