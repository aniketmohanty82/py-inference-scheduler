#!/bin/bash
# Scheduler arm driver for swe13: ARM=sched (our router, no KV offload).
#
#   MODE=smoke  -> 2 steps, then sched_gate.py on the driver log. Exit code is
#                  the gate's verdict: routing must be provably live, not just
#                  "the run finished".
#   MODE=run12  -> 12 steps, same supervision as run12fix.sh (crash detector,
#                  all-engine stall signature, in-pod headwatch, per-arm harvest).
#
# Redeploys the RayCluster first (pods must pick up the rebuilt image with
# the ledger/re-drain hook and the ROUTER_CONFIG_PATH / RAY_DEDUP_LOGS env),
# then re-establishes the engine snapshotter, which dies with the worker pod.
set -u
CTX=gke_aniket-gke-dev_us-south1_gke-gpu-rdma-cluster
JT=/usr/local/google/home/aniketmohanty/.claude/jobs/7c396abe/tmp/pressure
W=/usr/local/google/home/aniketmohanty/Projects/py-inference-scheduler-llmd/.claude/worktrees/verl-swe-store-ab
MODE=${MODE:-smoke}
ARM=sched
STEPS=$([ "$MODE" = smoke ] && echo 2 || echo 12)
TAG=sched_$MODE
OUT=$JT/$TAG
CAP_MIN=$([ "$MODE" = smoke ] && echo 100 || echo 420)
STALL_MIN=25
REDEPLOY=${REDEPLOY:-1}
mkdir -p "$OUT"
k(){ kubectl --context "$CTX" "$@"; }
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
num(){ tr -cd '0-9\n' | head -1; }
LOGF=/tmp/p40_${TAG}_${ARM}.log; DONEF=/tmp/p40_${TAG}_${ARM}.done

HEAD=$(k get pods --no-headers -o custom-columns=:metadata.name,:status.phase | awk '$1~/^swe-ab-head/&&$2=="Running"{print $1}'|head -1)
if [ -n "${HEAD:-}" ]; then
  LIVE=$(k exec "$HEAD" -- bash -c 'pgrep -cf "verl[.]trainer[.]main_ppo" || true' 2>/dev/null | num)
  [ "${LIVE:-0}" -eq 0 ] || { log "REFUSING: a trainer is running"; exit 1; }
fi

if [ "$REDEPLOY" = 1 ]; then
  log "redeploying RayCluster on rebuilt swe13"
  k delete raycluster swe-ab --ignore-not-found --wait=true >/dev/null 2>&1; sleep 20
  k apply -f "$W/integration/verl/k8s/swe-raycluster.yaml" >/dev/null
fi
for i in $(seq 1 60); do
  HEAD=$(k get pods --no-headers -o custom-columns=:metadata.name,:status.phase | awk '$1~/^swe-ab-head/&&$2=="Running"{print $1}'|head -1)
  WORKER=$(k get pods --no-headers | awk '$1~/^swe-ab-gpu-group-worker/ && $2=="2/2" && $3=="Running"{print $1}'|head -1)
  [ -n "${HEAD:-}" ] && [ -n "${WORKER:-}" ] && break; sleep 20
done
[ -n "${HEAD:-}" ] && [ -n "${WORKER:-}" ] || { log "pods never came up"; exit 1; }
log "head=$HEAD worker=$WORKER"

# Fail closed on the three things the arm depends on.
k exec "$HEAD" -- bash -c 'echo "  head env: ROUTER_CONFIG_PATH=$ROUTER_CONFIG_PATH RAY_DEDUP_LOGS=$RAY_DEDUP_LOGS"; test -f "$ROUTER_CONFIG_PATH"' 2>/dev/null \
  || { log "REFUSING: profile missing on head"; exit 1; }
LEDGER=$(k exec "$WORKER" -c ray-worker -- grep -c SharedInflightLedger /opt/py-inference-scheduler/integration/verl/verl_hook.py 2>/dev/null | num)
[ "${LEDGER:-0}" -ge 1 ] || { log "REFUSING: worker image lacks the ledger hook"; exit 1; }

# GPU-free compat check against the BAKED code and the live Ray cluster:
# routing away from a saturated fake engine, full endpoint view in two
# clients, shared ledger back to zero, and turn-to-turn affinity with the
# saturation filter as the only way off the holder.
k exec "$HEAD" -- bash -c 'cd /opt/py-inference-scheduler && PYTHONPATH=/opt/py-inference-scheduler:/opt/py-inference-scheduler/src python3 -m integration.verl.hook_compat_check 2>/dev/null' > "$OUT/compat.txt" 2>&1
grep -q "HOOK COMPAT CHECK: PASS" "$OUT/compat.txt" || { log "REFUSING: hook compat check failed"; tail -n 12 "$OUT/compat.txt"; exit 1; }
log "compat check PASS ($(grep -c . "$OUT/compat.txt") lines)"
k cp "$JT/p40_arm.sh" "$HEAD:/tmp/p40_arm.sh" >/dev/null 2>&1
k cp "$JT/headwatch.sh" "$HEAD:/tmp/headwatch.sh" >/dev/null 2>&1
k exec "$HEAD" -- chmod +x /tmp/p40_arm.sh /tmp/headwatch.sh >/dev/null 2>&1
k cp "$JT/snap13.py" "$WORKER:/tmp/snap13.py" -c ray-worker >/dev/null 2>&1
for attempt in 1 2 3 4 5; do
  RUNNING=$(k exec "$WORKER" -c ray-worker -- bash -c 'pgrep -cf "python3 /tmp/[s]nap13.py" || true' 2>/dev/null | num)
  [ "${RUNNING:-0}" -eq 0 ] && k exec "$WORKER" -c ray-worker -- bash -c 'setsid nohup python3 /tmp/snap13.py </dev/null >/tmp/snap13.err 2>&1 & disown' >/dev/null 2>&1
  sleep 20
  SNAPS=$(k exec "$WORKER" -c ray-worker -- bash -c 'grep -c "^SNAP " /tmp/snap13.log 2>/dev/null || true' 2>/dev/null | num)
  [ "${SNAPS:-0}" -ge 1 ] && break
done
[ "${SNAPS:-0}" -ge 1 ] || { log "REFUSING: snapshotter did not start"; exit 1; }

k exec "$WORKER" -c ray-worker -- bash -c 'pgrep -f "verl[.]trainer[.]main_ppo" >/dev/null || rm -f /tmp/lookup_rpc_port_*' >/dev/null 2>&1
k exec "$HEAD" -- bash -c "rm -f $DONEF $LOGF" >/dev/null 2>&1
SINCE=$(date -u +%s)
L=$(k exec "$HEAD" -- bash -c "setsid nohup env ARM=$ARM STEPS=$STEPS TAG=$TAG /tmp/p40_arm.sh </dev/null >/dev/null 2>&1 & disown; sleep 10; pgrep -cf '/tmp/p40_arm'" | num)
[ "${L:-0}" -ge 1 ] || { log "launch did not take"; exit 1; }
k exec "$HEAD" -- bash -c "setsid nohup env LOGF=$LOGF DONEF=$DONEF CAP_MIN=$CAP_MIN /tmp/headwatch.sh </dev/null >/tmp/headwatch_${TAG}.out 2>&1 & disown" >/dev/null 2>&1
log "launched $MODE ($STEPS steps); headwatch armed (cap ${CAP_MIN}min)"

SNAPCMD='tail -c 600000 /tmp/snap13.log | awk "/^SNAP /{n++} {b[n]=b[n] \"\n\" \$0} END{print b[n-1]}"'
START=$(date +%s); LAST=""; STALL=0; DONE=""
while [ $(( ($(date +%s)-START)/60 )) -lt "$CAP_MIN" ]; do
  sleep 60
  DONE=$(k exec "$HEAD" -- cat "$DONEF" 2>/dev/null); [ -n "$DONE" ] && break
  CRASH=$(k exec "$HEAD" -- bash -c "grep -c 'EngineCore encountered a fatal error' $LOGF || true" 2>/dev/null | num)
  if [ "${CRASH:-0}" -gt 0 ]; then log "ENGINE CRASH - stopping"; DONE="ENGINE_CRASH"
    k exec "$HEAD" -- bash -c 'for p in $(pgrep -f "p40_arm[.]sh") $(pgrep -f "verl[.]trainer[.]main_ppo"); do kill $p; done' >/dev/null 2>&1; break; fi
  S=$(k exec "$HEAD" -- bash -c "grep -acE 'step:[0-9]+ -' $LOGF || true" 2>/dev/null | num)
  VIEWS=$(k exec "$HEAD" -- bash -c "grep -c 'endpoint view: 4/4' $LOGF || true" 2>/dev/null | num)
  FLEETN=$(k exec "$HEAD" -- bash -c "grep -c '^.*FLEET\[' $LOGF || true" 2>/dev/null | num)
  AFF=$(k exec "$HEAD" -- bash -c "grep -a 'AFFINITY\[' $LOGF | tail -1 | grep -oE 'kept=.*'" 2>/dev/null)
  # Blind-metrics tripwire: the ledger says engines are busy (q_sum>=20) but
  # the scraped kv/running are zero on every engine in every such line. Run 1
  # (09-23) spent 53 min reaching the step-1 gate with exactly this signature.
  BLIND=$(k exec "$HEAD" -- bash -c "grep -a 'FLEET\[' $LOGF | tail -300" 2>/dev/null | python3 -c '
import re,sys
busy=live=0
for line in sys.stdin:
    parts=re.findall(r"(\d+)=kv([\d.]+)/w(\d+)/r(\d+)/p(\d+)/q(\d+)",line)
    if len(parts)<2 or sum(int(p[5]) for p in parts)<20: continue
    busy+=1; live+=any(float(p[1])>0 or int(p[3])>0 for p in parts)
print("BLIND" if busy>=40 and live==0 else f"ok busy={busy} live={live}")')
  if [ "${BLIND:-}" = BLIND ]; then log "METRICS BLIND (busy engines, zero kv/running everywhere) - stopping"; DONE="METRICS_BLIND"
    k exec "$HEAD" -- bash -c 'for p in $(pgrep -f "p40_arm[.]sh") $(pgrep -f "verl[.]trainer[.]main_ppo"); do kill $p; done' >/dev/null 2>&1; break; fi
  if [ "$MODE" = run12 ] && [ -z "${GATED:-}" ] && [ "${S:-0}" -ge 1 ]; then
    # The smoke folded into the run: gate on step 1's instruments and stop
    # here rather than spend eleven more steps routing blind.
    GATED=1
    k exec "$HEAD" -- cat "$LOGF" > "$OUT/${ARM}_step1.log" 2>/dev/null
    python3 "$JT/sched_gate.py" "$OUT/${ARM}_step1.log" > "$OUT/gate_step1.txt" 2>&1
    if grep -q "SCHED GATE: PASS" "$OUT/gate_step1.txt"; then log "step-1 gate PASS"; else
      log "step-1 gate FAIL - stopping"; cat "$OUT/gate_step1.txt"; DONE="GATE_FAIL"
      k exec "$HEAD" -- bash -c 'for p in $(pgrep -f "p40_arm[.]sh") $(pgrep -f "verl[.]trainer[.]main_ppo"); do kill $p; done' >/dev/null 2>&1; break; fi
  fi
  BLK=$(k exec "$WORKER" -c ray-worker -- bash -c "$SNAPCMD" 2>/dev/null)
  SIG=$(printf %s "$BLK" | grep -E 'num_requests_(running|waiting)\{|generation_tokens_total\{' | md5sum)
  RW=$(printf %s "$BLK" | grep -E 'num_requests_(running|waiting)\{' | awk '{s+=$2} END{print s+0}')
  if [ "$SIG" = "$LAST" ] && [ "${RW:-0}" -gt 0 ]; then STALL=$((STALL+1)); else STALL=0; LAST="$SIG"; fi
  log "  $TAG t=$(( ($(date +%s)-START)/60 ))min steps=${S:-0}/$STEPS views4/4=${VIEWS:-0} fleet_lines=${FLEETN:-0} ${AFF:-aff=?} metrics=${BLIND:-?} run+wait=${RW:-?} stall=${STALL}min"
  if [ "$STALL" -ge "$STALL_MIN" ]; then log "WEDGED"; DONE="WEDGED"
    k exec "$HEAD" -- bash -c 'for p in $(pgrep -f "p40_arm[.]sh") $(pgrep -f "verl[.]trainer[.]main_ppo"); do kill $p; done' >/dev/null 2>&1; break; fi
done
k exec "$HEAD" -- cat "$LOGF" > "$OUT/${ARM}.log" 2>/dev/null
k exec "$WORKER" -c ray-worker -- bash -c "awk -v t=$SINCE 'BEGIN{p=0} /^SNAP /{p=(\$2>=t)} p' /tmp/snap13.log" > "$OUT/${ARM}_scrape.log" 2>/dev/null
echo "${DONE:-TIMEOUT}" > "$OUT/${ARM}.done"
log "finished [${DONE:-TIMEOUT}] steps=$(grep -acE 'step:[0-9]+ -' "$OUT/${ARM}.log" 2>/dev/null) scrape=$(grep -c '^SNAP ' "$OUT/${ARM}_scrape.log" 2>/dev/null)"

if [ "$MODE" = smoke ]; then
  log "=== sched gate ==="
  python3 "$JT/sched_gate.py" "$OUT/${ARM}.log" | tee "$OUT/gate.txt"
  exit "${PIPESTATUS[0]}"
fi
