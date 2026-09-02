#!/usr/bin/env bash
# Pre-launch stale-state sweep: run between a dead job and the next launch so
# every run starts from S3-clean conditions. Removes RUNTIME residue only -
# the dead job's RayCluster (surviving engine processes hold HBM, and a
# resubmit onto a live cluster reuses them silently), orphan sandbox pods,
# and the store registry (restart + verified-empty, per S3a). Never touches
# harvested artifacts, benchmark-results, driver logs, or mirrored images.
#
# Usage: prelaunch-cleanup.sh <rayjob-name> [namespace]
set -uo pipefail
JOB=${1:?usage: prelaunch-cleanup.sh <rayjob-name> [namespace]}
NS=${2:-default}
k() { kubectl -n "$NS" "$@"; }

echo "== 1/4 dead job teardown: $JOB =="
STATUS=$(k get rayjob "$JOB" -o jsonpath='{.status.jobStatus}' 2>/dev/null || true)
if [ "$STATUS" = "RUNNING" ]; then
  echo "REFUSING: $JOB is RUNNING - this sweep is for dead jobs only"
  exit 1
fi
k delete rayjob "$JOB" --ignore-not-found --wait=false
LEFT=-1
for _ in $(seq 1 60); do
  LEFT=$(k get pods --no-headers 2>/dev/null | grep -c "^${JOB}-")
  [ "$LEFT" -eq 0 ] && break
  sleep 5
done
if [ "$LEFT" -ne 0 ]; then
  echo "FAIL: $LEFT job pods still present after 5min - engines may hold HBM"
  exit 1
fi
echo "job pods gone"

echo "== 2/4 orphan sandboxes =="
ORPHANS=$(k get pods -l app=rllm-sandbox --no-headers 2>/dev/null | awk '{print $1}')
COUNT=$(printf '%s' "$ORPHANS" | grep -c . || true)
if [ "$COUNT" -gt 0 ]; then
  printf '%s\n' "$ORPHANS" | xargs -r kubectl -n "$NS" delete pod --wait=false > /dev/null
fi
echo "deleted $COUNT orphan sandbox pods"

echo "== 3/4 store registry reset (S3a) =="
k delete pod -l app=mooncake-master --wait=false
CAP=unknown
for _ in $(seq 1 30); do
  sleep 5
  CAP=$(k exec deploy/mooncake-master -- curl -s http://127.0.0.1:9003/metrics 2>/dev/null \
        | awk '/^master_total_capacity_bytes/ {print $2}')
  [ "$CAP" = "0" ] && break
done
if [ "$CAP" != "0" ]; then
  echo "FAIL: registry not verified empty (capacity=$CAP)"
  exit 1
fi
echo "registry empty"

echo "== 4/4 GPU-node residue report =="
# Report-only: anything non-system still scheduled on the GPU pool deserves
# eyes before the next launch (a second job's leftovers, a debug pod).
k get pods --all-namespaces -o wide --no-headers 2>/dev/null \
  | grep "rdma-gpu-pool" \
  | grep -vE "kube-system|gmp-system|Completed" || echo "gpu nodes clean"

echo "PRELAUNCH CLEAN"
