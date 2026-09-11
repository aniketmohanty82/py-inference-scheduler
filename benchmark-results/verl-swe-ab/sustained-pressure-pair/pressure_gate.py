"""Judge whether a run held the KV pool oversubscribed, not merely touched it.

pressure-pair-v2 passed a "preemptions > 0" gate and still spent half its
steps in a low-occupancy state where the local prefix cache served 91% of
prefill - the store had nothing left to rescue and the comparison collapsed to
a wash. The distinguishing signal is not peak occupancy (every step of both
arms touched kv_cache_usage_perc 0.99 at some point) but how much of the
engine-busy window was spent there:

    pair-v2 recompute step 1   6/12 active snapshots above 0.85   collapsed
    pair-v2 recompute step 4   0/5  active snapshots above 0.85   healthy

So the gate is a FRACTION over the busy window. Snapshots with few running
requests are excluded: a draining rollout sits near-idle at low occupancy
through no fault of the regime, and counting those buries the signal.

Usage: pressure_gate.py <scrape.log> [min_hot_frac] [hot_threshold]
Exit 0 if the run was sustainedly pressured, 1 otherwise.
"""

import gzip
import sys

RUN_FLOOR = 5  # below this the engines are draining, not serving
DEFAULT_MIN_HOT = 0.60
HOT = 0.85


def snapshots(path):
    """Yield (total_running, mean_kv, total_preemptions) per SNAP block."""
    opener = gzip.open if path.endswith(".gz") else open
    cur, run, kvs, pre = None, 0.0, [], 0.0
    with opener(path, "rt", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("SNAP "):
                if cur is not None:
                    yield run, (sum(kvs) / len(kvs) if kvs else 0.0), pre
                cur, run, kvs, pre = line, 0.0, [], 0.0
                continue
            # engine_scrape.py emits only "SNAP <ts>", "=== PORT n ===" and
            # metric lines, and every metric it selects starts with "vllm:".
            if cur is None or not line.startswith("vllm:"):
                continue
            name, _, val = line.rpartition(" ")
            try:
                v = float(val)
            except ValueError:
                continue
            base = name.split("{")[0]
            if base == "vllm:num_requests_running":
                run += v
            elif base == "vllm:kv_cache_usage_perc":
                kvs.append(v)
            elif base == "vllm:num_preemptions_total":
                pre += v
    if cur is not None:
        yield run, (sum(kvs) / len(kvs) if kvs else 0.0), pre


def main() -> int:
    path = sys.argv[1]
    min_hot = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MIN_HOT
    # The occupancy threshold is the load-bearing number here, so it is tunable
    # alongside the fraction rather than buried as a constant.
    hot = float(sys.argv[3]) if len(sys.argv) > 3 else HOT

    active = [(kv, pre) for run, kv, pre in snapshots(path) if run > RUN_FLOOR]
    if not active:
        print("pressure=FAIL reason=no_active_snapshots")
        return 1

    n_hot = sum(1 for kv, _ in active if kv > hot)
    hot_frac = n_hot / len(active)
    mean_kv = sum(kv for kv, _ in active) / len(active)
    # num_preemptions_total is a counter, so take the span across the window
    # rather than its last value - on a sliced log the raw maximum reports
    # everything the engines ever preempted, not what this window did.
    pres = [pre for _, pre in active]
    preempt = max(pres) - min(pres)

    ok = hot_frac >= min_hot and preempt > 0
    print(
        f"pressure={'PASS' if ok else 'FAIL'} "
        f"hot_frac={hot_frac:.2f} (need {min_hot:.2f}) "
        f"active_snaps={len(active)} hot_snaps={n_hot} "
        f"mean_kv={mean_kv:.3f} preemptions={preempt:.0f}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
