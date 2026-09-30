#!/bin/bash
# Run the prefill probe in the first idle window of a rollout tail: the live
# engines (ports from the newest snapshot block) have had a busy phase in the
# last 10 minutes, the last two snapshots show nothing running or waiting,
# and the GPUs are quiet. Polls every 15 s for up to 90 min; v2 probe (fresh prompts, GPU-idle guard).
#   ./tail_probe.sh <kube ctx> <worker pod>
set -u
CTX=$1; POD=$2
for i in $(seq 1 360); do
  STATE=$(kubectl --context "$CTX" exec -i "$POD" -c ray-worker -- python3 - <<'EOF' 2>/dev/null
import time
snaps = []  # (ts, ports, running+waiting)
ts = None; ports = []; load = 0
for ln in open("/tmp/snap13.log", errors="replace"):
    if ln.startswith("SNAP "):
        if ts is not None: snaps.append((ts, ports, load))
        ts, ports, load = int(ln.split()[1]), [], 0
    elif ln.startswith("=== PORT"):
        ports.append(ln.split()[2])
    elif ln.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting{")):
        load += float(ln.rsplit(" ", 1)[1])
if ts is not None: snaps.append((ts, ports, load))
now = int(time.time())
recent = [s for s in snaps if now - s[0] <= 600 and s[1]]
last = [s for s in recent if now - s[0] <= 90]
busy_seen = any(s[2] >= 20 for s in recent)
idle_now = len(last) >= 1 and all(s[2] == 0 for s in recent[-2:])
port = recent[-1][1][0] if recent else ""
print(f"{port} {int(busy_seen)} {int(idle_now)} {len(recent)}")
EOF
)
  set -- $STATE; PORT=${1:-}; BUSY=${2:-0}; IDLE=${3:-0}; NREC=${4:-0}
  UTIL=$(kubectl --context "$CTX" exec "$POD" -c ray-worker -- nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -n 1)
  echo "[$(date -u +%T)] port=${PORT:-?} busy_seen=$BUSY idle_now=$IDLE recent_snaps=$NREC gpu_util_max=${UTIL:-?}"
  if [ -n "$PORT" ] && [ "$BUSY" = 1 ] && [ "$IDLE" = 1 ] && [ "${UTIL:-100}" -lt 15 ]; then
    echo "[$(date -u +%T)] idle window: probing engine $PORT"
    "$(dirname "$0")/prefill_probe2.sh" "$CTX" "$POD" "$PORT" && exit 0
    echo "[$(date -u +%T)] probe failed; keep polling"
  fi
  sleep 15
done
echo "no idle window found"
