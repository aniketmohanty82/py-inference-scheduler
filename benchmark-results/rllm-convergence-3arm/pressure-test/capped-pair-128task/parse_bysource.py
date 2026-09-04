"""Final per-source counter values from a recovered scraper dump."""

import re
import sys

best = {}
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"(vllm:[a-z_]+_total)\{([^}]*)\}\s+([0-9.e+]+)", line)
    if not m:
        continue
    labels, raw = m.group(2), m.group(3)
    src = re.search(r'source="(\w+)"', labels)
    op = re.search(r'operation="(\w+)"', labels)
    status = re.search(r'status="(\w+)"', labels)
    key = m.group(1)
    if src:
        key += ":" + src.group(1)
    if op:
        key += ":" + op.group(1) + ("/" + status.group(1) if status else "")
    try:
        best[key] = max(best.get(key, 0.0), float(raw))
    except ValueError:
        pass

total = best.get("vllm:prompt_tokens_total", 0.0)
for k in sorted(best):
    share = f"  ({best[k] / total * 100:5.1f}% of processed)" if total and "by_source" in k else ""
    print(f"{k:<62} {best[k]:>14,.0f}{share}")

parts = [best.get(f"vllm:prompt_tokens_by_source_total:{s}", 0.0)
         for s in ("local_compute", "local_cache_hit", "external_kv_transfer")]
if total:
    print(f"\nidentity check: parts sum {sum(parts):,.0f} vs total {total:,.0f} "
          f"-> {'EXACT' if abs(sum(parts) - total) < 1 else 'MISMATCH'}")
