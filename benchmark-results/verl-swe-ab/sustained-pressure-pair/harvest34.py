"""Harvest the sustained-pressure pair: verl step lines + engine by_source.

Parser note: `perf/throughput` is the LAST field on a verl step line, and Ray
interleaves other actors' output onto it, gluing text straight onto the number
(`200.086(TaskRunner pid=...`). A parser that does float(field) drops the
metric for that step and silently shrinks the table - which is exactly what
happened to pressure-pair-v2. So: strip ANSI, then take the LEADING float of
each field rather than requiring the whole field to parse, and report which
steps are missing which keys instead of intersecting them away.
"""

import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LEADING_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def parse_steps(path):
    rows = {}
    for raw in open(path, errors="replace"):
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
    for raw in open(path, errors="replace"):
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
