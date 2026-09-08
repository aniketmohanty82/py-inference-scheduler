#!/usr/bin/env bash
# Unattended store-vs-recompute A/B: recompute arm, store reset, store arm.
#
# Idempotent and resumable: each arm writes a .done marker, so re-running
# skips finished arms. Logs stream to $OUT_DIR for post-hoc analysis - the
# engines and their counters die with the job, so nothing may be left in the
# pod at the end.
#
# Recompute runs FIRST deliberately: it is the arm that cannot be corrupted
# by store state, so if the night dies halfway we still hold a clean baseline.
set -uo pipefail

CTX=${CTX:-gke_aniket-gke-dev_us-south1_gke-gpu-rdma-cluster}
OUT_DIR=${OUT_DIR:-/usr/local/google/home/aniketmohanty/.claude/jobs/68141fe0/tmp/swe_ab}
STEPS=${STEPS:-12}
HEAD_SELECTOR=${HEAD_SELECTOR:-ray.io/node-type=head}
mkdir -p "$OUT_DIR"

k() { kubectl --context "$CTX" "$@"; }

head_pod() {
  k get pods -l "$HEAD_SELECTOR,ray.io/cluster=swe-ab" --no-headers \
    -o custom-columns=':metadata.name' 2>/dev/null | head -1
}

reset_store() {
  # S3a between arms: fresh master, verified-empty registry. Never do this
  # while engines hold segments - only between jobs.
  echo "== store reset =="
  k delete pod -l app=mooncake-master --wait=false >/dev/null 2>&1 || true
  k rollout status deploy/mooncake-master --timeout=900s >/dev/null || {
    echo "FAIL: master not Ready"; return 1; }
  for _ in $(seq 1 30); do
    cap=$(k exec deploy/mooncake-master -- curl -s http://127.0.0.1:9003/metrics 2>/dev/null \
          | awk '/^master_total_capacity_bytes/ {print $2}')
    [ "$cap" = "0" ] && { echo "registry empty"; return 0; }
    sleep 5
  done
  echo "FAIL: registry did not drain"; return 1
}

sweep_sandboxes() {
  n=$(k get pods -l app=swe-sandbox --no-headers 2>/dev/null | grep -c . || true)
  [ "${n:-0}" -gt 0 ] && k delete pods -l app=swe-sandbox --wait=false >/dev/null 2>&1
  echo "swept ${n:-0} sandbox pods"
}

run_arm() {
  arm=$1
  marker="$OUT_DIR/$arm.done"
  if [ -f "$marker" ]; then echo "== $arm already done, skipping =="; return 0; fi
  pod=$(head_pod)
  [ -z "$pod" ] && { echo "FAIL: no head pod"; return 1; }

  echo "== $arm arm starting $(date -u +%H:%M:%S) on $pod =="
  sweep_sandboxes
  reset_store || return 1

  # Submit from inside the head pod so the job survives our shell exiting.
  k exec "$pod" -- bash -lc "
      cd /opt/py-inference-scheduler &&
      ARM=$arm STEPS=$STEPS SWE_DATA_DIR=/home/ray/data/swe \
      nohup bash integration/verl/examples/run_swe_ab.sh \
        > /tmp/${arm}_driver.log 2>&1 &
      echo submitted" >/dev/null 2>&1

  # Follow until the trainer prints its final step or the log goes quiet.
  last=0; still=0
  while true; do
    sleep 60
    lines=$(k exec "$pod" -- bash -lc "wc -l < /tmp/${arm}_driver.log" 2>/dev/null | tr -d ' \r')
    lines=${lines:-0}
    done_step=$(k exec "$pod" -- bash -lc \
      "grep -c 'step:${STEPS} ' /tmp/${arm}_driver.log" 2>/dev/null | tr -d ' \r')
    if [ "${done_step:-0}" -ge 1 ]; then echo "$arm reached step $STEPS"; break; fi
    if [ "$lines" = "$last" ]; then
      still=$((still + 1))
      if [ "$still" -ge 30 ]; then echo "$arm: log quiet 30min - stopping follow"; break; fi
    else
      still=0
    fi
    last=$lines
    echo "  $(date -u +%H:%M:%S) $arm lines=$lines"
  done

  echo "== $arm harvesting =="
  k exec "$pod" -- bash -lc "cat /tmp/${arm}_driver.log" > "$OUT_DIR/${arm}_driver.log" 2>/dev/null
  worker=$(k get pods -l ray.io/cluster=swe-ab --no-headers -o custom-columns=':metadata.name' \
           2>/dev/null | grep worker | head -1)
  [ -n "$worker" ] && k logs "$worker" -c metrics-scraper --tail=2000000 \
      > "$OUT_DIR/${arm}_scraper.log" 2>/dev/null
  k exec deploy/mooncake-master -- curl -s http://127.0.0.1:9003/metrics \
      > "$OUT_DIR/${arm}_master.txt" 2>/dev/null
  touch "$marker"
  echo "== $arm done $(date -u +%H:%M:%S), $(wc -l < "$OUT_DIR/${arm}_driver.log") log lines =="
}

run_arm recompute
run_arm store
sweep_sandboxes
echo "== A/B COMPLETE $(date -u +%H:%M:%S) =="
ls -la "$OUT_DIR"
