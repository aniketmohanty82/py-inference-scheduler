"""pb2 analyzer: per-replica + aggregate windowed metrics from scraper logs.

Handles N replicas (the single-replica scripts assumed sorted(urls)[0]).
Window per replica: first sample with kv>=0.95 to last with kv>=0.5.
Aggregate window: earliest window start to latest window end across replicas.

Usage: python pb2_analyze.py <scraper_raw.log>
"""

import re
import sys
from collections import defaultdict

LINE = re.compile(r"(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.e+-]+)")


def load(path):
    snaps, cur = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith("TS "):
            cur = {"ts": int(line.split()[1]), "d": defaultdict(dict)}
            snaps.append(cur)
        elif line.startswith("URL ") and cur is not None:
            p = line.split(" ", 2)
            m = LINE.match(p[2])
            if not m:
                continue
            name, labels = m.group(1), m.group(2) or ""
            if name == "vllm:prompt_tokens_by_source_total":
                src = re.search(r'source="([a-z_]+)"', labels)
                name = f"{name}:{src.group(1)}" if src else name
            cur["d"][p[1]][name] = float(m.group(3))
    return [s for s in snaps if s["d"]]


def analyze(path):
    snaps = load(path)
    urls = sorted({u for s in snaps for u in s["d"]})
    print(f"{len(snaps)} snapshots, {len(urls)} replicas: {urls}")

    def v(s, url, k):
        return s["d"].get(url, {}).get(k, 0.0)

    agg = defaultdict(float)
    starts, ends = [], []
    for url in urls:
        kvs = [(i, v(s, url, "vllm:kv_cache_usage_perc")) for i, s in enumerate(snaps) if url in s["d"]]
        hot = [i for i, k in kvs if k >= 0.95]
        if not hot:
            print(f"\n== {url}: never reached kv>=0.95 (max {max(k for _, k in kvs):.2f}) - no window")
            continue
        i0 = hot[0]
        i1 = max(i for i, k in kvs if k >= 0.5)
        w0, w1 = snaps[i0], snaps[i1]
        dur = w1["ts"] - w0["ts"]
        starts.append(w0["ts"])
        ends.append(w1["ts"])

        def d(k):
            return v(w1, url, k) - v(w0, url, k)

        window_kv = [k for i, k in kvs if i0 <= i <= i1]
        pre = d("vllm:num_preemptions_total")
        cp = d("vllm:prompt_tokens_total")
        sp = d("vllm:request_prompt_tokens_sum")
        sg = d("vllm:request_generation_tokens_sum")
        lq = d("vllm:prefix_cache_queries_total")
        lh = d("vllm:prefix_cache_hits_total")
        eq = d("vllm:external_prefix_cache_queries_total")
        eh = d("vllm:external_prefix_cache_hits_total")
        ext = d("vllm:prompt_tokens_by_source_total:external_kv_transfer")
        print(f"\n== {url}")
        print(f"  window {dur/60:.1f} min ({i1-i0+1} samples), kv mean {sum(window_kv)/len(window_kv)*100:.1f}%")
        print(f"  preemptions {pre:.0f}")
        print(f"  computed prompt {cp:,.0f} tok ({cp/dur:,.0f}/s) | served prompt {sp:,.0f} | gen {sg:,.0f} ({sg/dur:,.0f}/s)")
        if sp:
            print(f"  computed as % of served: {cp/sp*100:.1f}%")
        if lq:
            print(f"  local hit rate {lh/lq*100:.1f}% ({lh:,.0f}/{lq:,.0f})")
        if eq:
            print(f"  store-tier hit rate {eh/eq*100:.1f}% ({eh:,.0f}/{eq:,.0f})")
        print(f"  tokens loaded FROM store (by_source external_kv_transfer): {ext:,.0f}")
        for k, val in [("pre", pre), ("cp", cp), ("sp", sp), ("sg", sg),
                       ("lq", lq), ("lh", lh), ("eq", eq), ("eh", eh), ("ext", ext)]:
            agg[k] += val

    if starts:
        dur = max(ends) - min(starts)
        print(f"\n== AGGREGATE (union window {dur/60:.1f} min)")
        print(f"  preemptions {agg['pre']:.0f}")
        print(f"  computed prompt {agg['cp']:,.0f} tok | served prompt {agg['sp']:,.0f} | gen {agg['sg']:,.0f}")
        if agg["sp"]:
            print(f"  computed as % of served: {agg['cp']/agg['sp']*100:.1f}%")
        if agg["lq"]:
            print(f"  local hit rate {agg['lh']/agg['lq']*100:.1f}%")
        if agg["eq"]:
            print(f"  store-tier hit rate {agg['eh']/agg['eq']*100:.1f}%")
        print(f"  tokens loaded FROM store: {agg['ext']:,.0f}")


if __name__ == "__main__":
    analyze(sys.argv[1])
