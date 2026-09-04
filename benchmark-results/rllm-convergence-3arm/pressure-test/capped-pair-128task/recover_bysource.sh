#!/bin/bash
# Recover the final by_source compute split for one arm from Cloud Logging.
# The sidecar's stdout outlives the pod there; pods from completed RayJobs are
# recreated, so the in-pod log is gone by harvest time.
POD=$1; FROM=$2; TO=$3; OUT=$4
gcloud logging read \
  "resource.type=\"k8s_container\" resource.labels.container_name=\"metrics-scraper\" resource.labels.pod_name=\"$POD\" timestamp>=\"$FROM\" timestamp<=\"$TO\" (textPayload:\"by_source_total\" OR textPayload:\"prompt_tokens_total{\" OR textPayload:\"num_preemptions_total\" OR textPayload:\"mooncake_store_operation_total\")" \
  --project=aniket-gke-dev --format='value(timestamp,textPayload)' --limit=4000 2>/dev/null | sort > "$OUT"
echo "lines: $(grep -c . "$OUT")"
python3 - "$OUT" <<'PY'
import re, sys
best = {}
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r'(vllm:[a-z_]+(?:_by_source)?_total)\{[^}]*?(?:source="(\w+)")?[^}]*\}\s+([0-9.e+]+)', line)
    if not m:
        continue
    key = m.group(1) + (":" + m.group(2) if m.group(2) else "")
    try:
        best[key] = max(best.get(key, 0.0), float(m.group(3)))
    except ValueError:
        pass
for k in sorted(best):
    print(f"  {k:<58} {best[k]:,.0f}")
PY
