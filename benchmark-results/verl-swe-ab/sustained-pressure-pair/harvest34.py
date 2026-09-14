"""Harvest the sustained-pressure pair: verl step lines + engine by_source.

Parser note: `perf/throughput` is the LAST field on a verl step line, and Ray
interleaves other actors' output onto it, gluing text straight onto the number
(`200.086(TaskRunner pid=...`). A parser that does float(field) drops the
metric for that step and silently shrinks the table - which is exactly what
happened to pressure-pair-v2. So: strip ANSI, then take the LEADING float of
each field rather than requiring the whole field to parse, and report which
steps are missing which keys instead of intersecting them away.
"""

import csv
import gzip
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LEADING_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _open(path):
    """Driver logs live raw on the pod and gzipped in the results dir."""
    return gzip.open(path, "rt", errors="replace") if str(path).endswith(".gz") \
        else open(path, errors="replace")


def parse_steps(path):
    rows = {}
    for raw in _open(path):
        line = ANSI.sub("", raw)
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        d = {}
        for field in m.group(2).split(" - "):
            if ":" not in field:
                continue
            # partition, not rpartition: metric keys never contain a colon, but
            # interleaved junk does ("...316.13(vLLMHttpServer...) WARNING:...:
            # Flushed..."), and splitting on the LAST colon hands back
            # " removed 0 keys" as the value and drops the metric.
            key, _, val = field.partition(":")
            hit = LEADING_NUM.match(val.strip())
            if hit:
                d[key.strip()] = float(hit.group())
        if len(d) > 20:
            rows[int(m.group(1))] = d
    return rows


def parse_scrape(path):
    """Per-SNAP totals across engines: running/waiting/kv/preempt/by_source."""
    snaps, cur = [], None
    for raw in _open(path):
        line = raw.strip()
        if line.startswith("SNAP "):
            cur = {"ts": int(line.split()[1]), "run": 0.0, "wait": 0.0, "kv": [],
                   "pre": 0.0, "compute": 0.0, "hit": 0.0, "ext": 0.0, "decode": 0.0}
            snaps.append(cur)
            continue
        if cur is None or not line.startswith("vllm:"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            v = float(val)
        except ValueError:
            continue
        base = name.split("{")[0]
        if base == "vllm:num_requests_running":
            cur["run"] += v
        elif base == "vllm:num_requests_waiting":
            cur["wait"] += v
        elif base == "vllm:kv_cache_usage_perc":
            cur["kv"].append(v)
        elif base == "vllm:num_preemptions_total":
            cur["pre"] += v
        elif base == "vllm:generation_tokens_total":
            cur["decode"] += v
        elif base == "vllm:prompt_tokens_by_source_total":
            for tag, k in (("local_compute", "compute"),
                           ("local_cache_hit", "hit"),
                           ("external_kv_transfer", "ext")):
                if f'source="{tag}"' in name:
                    cur[k] += v
    return snaps


def totals(snaps):
    """Final cumulative counters. Engines keep their ports for a whole arm."""
    last = {k: 0.0 for k in ("compute", "hit", "ext", "decode", "pre")}
    for s in snaps:
        for k in last:
            if s[k] > last[k]:
                last[k] = s[k]
    return last


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "supervised34"
    arms = {a: parse_steps(f"{base}/{a}_try1.log") for a in ("recompute", "store")}
    scr = {a: parse_scrape(f"{base}/{a}_try1_scrape.log") for a in ("recompute", "store")}

    # Coverage audit: which keys are missing where (never intersect silently).
    allkeys = set().union(*(set(d) for r in arms.values() for d in r.values()))
    for arm, rows in arms.items():
        for s, d in sorted(rows.items()):
            missing = allkeys - set(d)
            if missing:
                print(f"COVERAGE {arm} step{s}: missing {sorted(missing)}")
    print(f"keys per step: {len(allkeys)}\n")

    KEYS = ["timing_s/step", "timing_s/gen", "timing_s/agent_loop/generate_sequences/mean",
            "timing_s/agent_loop/tool_calls/mean", "timing_s/agent_loop/slowest/tool_calls",
            "timing_s/agent_loop/slowest/generate_sequences", "perf/throughput",
            "num_turns/mean", "response_length/mean", "prompt_length/mean",
            "perf/total_num_tokens", "actor/entropy", "actor/ppo_kl", "actor/grad_norm",
            "critic/score/mean", "actor/perf/cpu_memory_used_gb"]
    steps = sorted(set(arms["recompute"]) & set(arms["store"]))
    w = max(len(k) for k in KEYS) + 2
    hdr = f"{'metric':<{w}}" + "".join(f"{'rc s'+str(s):>12}" for s in steps)
    hdr += "  |" + "".join(f"{'st s'+str(s):>12}" for s in steps) + f"{'delta%':>10}{'st<rc':>7}"
    print(hdr)
    for k in KEYS:
        rv = [arms["recompute"][s].get(k) for s in steps]
        sv = [arms["store"][s].get(k) for s in steps]
        line = f"{k:<{w}}"
        for v in rv:
            line += f"{v:>12,.3f}" if v is not None else f"{'-':>12}"
        line += "  |"
        for v in sv:
            line += f"{v:>12,.3f}" if v is not None else f"{'-':>12}"
        pair = [(a, b) for a, b in zip(rv, sv) if a is not None and b is not None]
        if pair:
            ra = sum(a for a, _ in pair) / len(pair)
            sa = sum(b for _, b in pair) / len(pair)
            line += f"{(sa - ra) / ra * 100:>+10.1f}"
            line += f"{sum(1 for a, b in pair if b < a):>4}/{len(pair)}"
        print(line)

    print()
    for arm in ("recompute", "store"):
        t = totals(scr[arm])
        tot = t["compute"] + t["hit"] + t["ext"]
        print(f"{arm:>10}: local_compute {t['compute']:>13,.0f} ({t['compute']/tot:6.2%})  "
              f"local_cache_hit {t['hit']:>13,.0f} ({t['hit']/tot:6.2%})  "
              f"external {t['ext']:>12,.0f} ({t['ext']/tot:6.2%})  "
              f"total {tot:>13,.0f}  preempt {t['pre']:>5,.0f}  decode {t['decode']:>11,.0f}")


def write_csv(arms, scr, path="metrics.csv"):
    """Every recorded number for this pair in one flat file.

    verl step metrics carry their own names; engine-scrape metrics are derived
    per rollout window and prefixed `engine/` so the two sources stay
    distinguishable. Wide layout (one row per metric) because the common use is
    reading it, not plotting it.
    """
    steps = sorted(set(arms["recompute"]) & set(arms["store"]))
    rows = {}
    for arm in ("recompute", "store"):
        for k in set().union(*(set(d) for d in arms[arm].values())):
            rows.setdefault(k, {})[arm] = [arms[arm][s].get(k) for s in steps]
        for k, vals in engine_per_step(scr[arm], arms[arm], steps).items():
            rows.setdefault(f"engine/{k}", {})[arm] = vals

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric"] + [f"recompute_s{s}" for s in steps]
                   + [f"store_s{s}" for s in steps]
                   + ["recompute_mean", "store_mean", "delta_pct"])
        for k in sorted(rows):
            rc = rows[k].get("recompute", [None] * len(steps))
            st = rows[k].get("store", [None] * len(steps))
            pair = [(a, b) for a, b in zip(rc, st) if a is not None and b is not None]
            if pair:
                ra = sum(a for a, _ in pair) / len(pair)
                sa = sum(b for _, b in pair) / len(pair)
                d = f"{(sa - ra) / ra * 100:+.2f}" if ra else ""
            else:
                ra = sa = d = ""
            # repr(), not %g: repr is the shortest string that round-trips a
            # float exactly, so the CSV is a faithful record rather than a
            # 6-significant-figure approximation of it. %g silently turned
            # 66,551,848 recorded tokens into 66,551,800.
            w.writerow([k] + ["" if v is None else repr(v) for v in rc + st]
                       + [repr(ra) if ra != "" else "", repr(sa) if sa != "" else "", d])
    return path, len(rows)


def engine_per_step(snaps, rows, steps):
    """Per-rollout engine counters, keyed the same way as the verl metrics."""
    rolls, cur = [], []
    for i, s in enumerate(snaps):
        if s["run"] > 5:
            cur.append(i)
        elif cur:
            rolls.append(cur)
            cur = []
    if cur:
        rolls.append(cur)
    rolls = [r for r in rolls if len(r) >= 3]
    out = {k: [] for k in ("kv_usage_avg", "kv_usage_max", "hot_frac", "waiting_peak",
                           "running_peak", "preemptions", "local_compute", "local_cache_hit",
                           "external_kv_transfer", "decode_tokens", "busy_minutes")}
    for n, r in enumerate(rolls, 1):
        if n > len(steps):
            break
        a, b = r[0], r[-1]
        w = snaps[a:b + 2]
        kv = [sum(s["kv"]) / len(s["kv"]) for s in w if s["kv"]]
        d = {k: snaps[min(b + 1, len(snaps) - 1)][k] - snaps[max(a - 1, 0)][k]
             for k in ("compute", "hit", "ext", "pre", "decode")}
        out["kv_usage_avg"].append(sum(kv) / len(kv))
        out["kv_usage_max"].append(max(kv))
        out["hot_frac"].append(sum(1 for x in kv if x > 0.85) / len(kv))
        out["waiting_peak"].append(max(s["wait"] for s in w))
        out["running_peak"].append(max(s["run"] for s in w))
        out["preemptions"].append(d["pre"])
        out["local_compute"].append(d["compute"])
        out["local_cache_hit"].append(d["hit"])
        out["external_kv_transfer"].append(d["ext"])
        out["decode_tokens"].append(d["decode"])
        out["busy_minutes"].append(len(r))
    return out


def write_timeseries(path="engine_timeseries.csv", arms=("recompute", "store")):
    """Long-format Prometheus dump with a step and phase stamped on every row.

    The raw scrape is wall-clock only. Correlating a counter with the training
    step it belongs to is the thing anyone actually wants, so rollout windows
    (contiguous snapshots with requests running) are numbered and the gaps
    between them are labelled as the weight-update phase.
    """
    rows = []
    for arm in arms:
        snaps = parse_scrape(f"{arm}_scrape.log.gz")
        t0 = snaps[0]["ts"]
        # number the rollout windows; everything between them is training
        step_of = {}
        cur, n = [], 0
        for i, s in enumerate(snaps):
            if s["run"] > 5:
                cur.append(i)
            elif cur:
                if len(cur) >= 3:
                    n += 1
                    for j in cur:
                        step_of[j] = (n, "rollout")
                cur = []
        if cur and len(cur) >= 3:
            n += 1
            for j in cur:
                step_of[j] = (n, "rollout")
        seen = 0
        for i in range(len(snaps)):
            if i in step_of:
                seen = step_of[i][0]
            else:
                step_of[i] = (seen if seen else 1, "train" if seen else "startup")

        for i, raw in enumerate(_raw_snapshots(f"{arm}_scrape.log.gz")):
            ts, port_lines = raw
            step, phase = step_of.get(i, (0, "?"))
            for port, lines in port_lines.items():
                for name, value in lines:
                    base, _, lab = name.partition("{")
                    rows.append((ts, ts - t0, arm, step, phase, port, base,
                                 lab.rstrip("}"), value))

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unix_ts", "t_rel_s", "arm", "step", "phase", "engine_port",
                    "metric", "labels", "value"])
        w.writerows(rows)
    return path, len(rows)


def _raw_snapshots(path):
    """Yield (ts, {port: [(metric_name, value), ...]}) preserving all labels."""
    ts, ports, cur_port = None, None, None
    for raw in _open(path):
        line = raw.strip()
        if line.startswith("SNAP "):
            if ts is not None:
                yield ts, ports
            ts, ports, cur_port = int(line.split()[1]), {}, None
            continue
        if line.startswith("=== PORT"):
            cur_port = line.split()[2]
            ports.setdefault(cur_port, [])
            continue
        if ts is None or cur_port is None or not line.startswith("vllm:"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            ports[cur_port].append((name, float(val)))
        except ValueError:
            continue
    if ts is not None:
        yield ts, ports
