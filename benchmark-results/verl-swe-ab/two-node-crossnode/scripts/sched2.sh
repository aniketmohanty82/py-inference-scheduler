#!/bin/bash
# Two-node cross-node KV benchmark driver: ARM=fcstore (our store connector +
# the hook with flow control), NNODES=2, 8 engines at tp=2 over two a3-ultra
# nodes, RayCluster swe-ab2 from integration/verl/k8s/swe-raycluster-2node.yaml.
#
#   CTX=<kube context> MODE=smoke|run STEPS=<n> ./sched2.sh
#
# Same supervision as sched13.sh (fail-closed preflight, in-image compat check
# incl. the parking scenario, snapshotter on BOTH worker pods, blind-metrics
# tripwire, crash/stall detection, harvest), plus the cross-node instruments:
# MOVE cross_node=1 lines from the hook, PULLSRC lines from the engine worker
# logs on each node, and FLOWCONTROL park/drop lines. crossnode_gate.py is the
# verdict.
set -u
CTX=${CTX:?set CTX to the kube context of the cluster holding two H200 nodes}
JT=/usr/local/google/home/aniketmohanty/.claude/jobs/7c396abe/tmp/pressure
W=/usr/local/google/home/aniketmohanty/Projects/py-inference-scheduler-llmd/.claude/worktrees/verl-swe-store-ab
MODE=${MODE:-smoke}
# ARM=recompute gives the two-node control: verl's balancer, no tier, no hook.
ARM=${ARM:-fcstore}
NNODES=2
STEPS=${STEPS:-$([ "$MODE" = smoke ] && echo 2 || echo 12)}
# TAG_SUFFIX distinguishes variants of one arm (e.g. a save_decode_cache=false run).
TAG=x2_${MODE}$([ "$ARM" = fcstore ] || echo "_$ARM")${TAG_SUFFIX:-}
OUT=$JT/$TAG
CAP_MIN=${CAP_MIN:-$([ "$MODE" = smoke ] && echo 120 || echo 480)}
STALL_MIN=25
REDEPLOY=${REDEPLOY:-1}
mkdir -p "$OUT"
k(){ kubectl --context "$CTX" "$@"; }
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
num(){ tr -cd '0-9\n' | head -1; }
LOGF=/tmp/p40_${TAG}_${ARM}.log; DONEF=/tmp/p40_${TAG}_${ARM}.done

HEAD=$(k get pods --no-headers -o custom-columns=:metadata.name,:status.phase | awk '$1~/^swe-ab2-head/&&$2=="Running"{print $1}'|head -1)
if [ -n "${HEAD:-}" ]; then
  LIVE=$(k exec "$HEAD" -- bash -c 'pgrep -cf "verl[.]trainer[.]main_ppo" || true' 2>/dev/null | num)
  [ "${LIVE:-0}" -eq 0 ] || { log "REFUSING: a trainer is running"; exit 1; }
fi
GPU_NODES=$(k get nodes -l cloud.google.com/gke-accelerator=nvidia-h200-141gb --no-headers 2>/dev/null | grep -c " Ready")
[ "${GPU_NODES:-0}" -ge 2 ] || { log "REFUSING: need 2 Ready H200 nodes, have ${GPU_NODES:-0}"; exit 1; }

if [ "$REDEPLOY" = 1 ]; then
  log "deploying RayCluster swe-ab2 (2 GPU workers)"
  k delete raycluster swe-ab2 --ignore-not-found --wait=true >/dev/null 2>&1; sleep 20
  k apply -f "$W/integration/verl/k8s/swe-raycluster-2node.yaml" >/dev/null
fi
for i in $(seq 1 90); do
  HEAD=$(k get pods --no-headers -o custom-columns=:metadata.name,:status.phase | awk '$1~/^swe-ab2-head/&&$2=="Running"{print $1}'|head -1)
  WORKERS=$(k get pods --no-headers | awk '$1~/^swe-ab2-gpu-group-worker/ && $2=="2/2" && $3=="Running"{print $1}')
  NW=$(echo "$WORKERS" | grep -c .)
  [ -n "${HEAD:-}" ] && [ "$NW" -ge 2 ] && break; sleep 20
done
[ -n "${HEAD:-}" ] && [ "${NW:-0}" -ge 2 ] || { log "pods never came up (head=${HEAD:-none} workers=${NW:-0})"; exit 1; }
log "head=$HEAD workers: $(echo $WORKERS | tr '\n' ' ')"

# Fail closed on what the arm depends on: profile with flow control, hook with
# the manager, connector with the pull-source instrument, 2 nodes' RDMA claims.
[ "$ARM" != fcstore ] || k exec "$HEAD" -- bash -c 'echo "  head env: ROUTER_CONFIG_PATH=$ROUTER_CONFIG_PATH"; test -f "$ROUTER_CONFIG_PATH" && grep -q simple_backpressure "$ROUTER_CONFIG_PATH"' 2>/dev/null \
  || { log "REFUSING: cross-node profile (with flow control) missing on head"; exit 1; }
for wpod in $WORKERS; do
  FC=$(k exec "$wpod" -c ray-worker -- grep -c FlowControlManager /opt/py-inference-scheduler/integration/verl/verl_hook.py 2>/dev/null | num)
  PS=$(k exec "$wpod" -c ray-worker -- grep -c PULLSRC /opt/py-inference-scheduler/src/py_inference_scheduler/datalayer/connectors/mooncake/rl_pull_policy.py 2>/dev/null | num)
  if [ "$ARM" = fcstore ]; then [ "${FC:-0}" -ge 1 ] && [ "${PS:-0}" -ge 1 ] || { log "REFUSING: $wpod image lacks flow control / PULLSRC"; exit 1; }; fi
  IB=$(k exec "$wpod" -c ray-worker -- bash -c 'ls /dev/infiniband 2>/dev/null | grep -c uverbs' 2>/dev/null | num)
  log "  $wpod: hook+connector ok, RDMA devices=${IB:-0}, node=$(k get pod "$wpod" -o jsonpath='{.spec.nodeName}')"
done

# Cross-node NCCL must complete a collective through the pods' own env (the
# gIB plugin): the 09-24 smoke lost its step-1 update to a silent NCCL hang.
W1=$(echo "$WORKERS" | sed -n 1p); W2=$(echo "$WORKERS" | sed -n 2p)
PORT=29790 "$JT/nccl_pair_test.sh" "$CTX" "$W1" "$W2" > "$OUT/nccl_pair.txt" 2>&1
NET=$(grep -oE "Using network [A-Za-z]+" "$OUT/nccl_pair.txt" | head -1)
grep -q "RANK0: init" "$OUT/nccl_pair.txt" && grep -q "RANK1: init" "$OUT/nccl_pair.txt" || { log "REFUSING: cross-node NCCL did not complete ($NET)"; tail -n 6 "$OUT/nccl_pair.txt"; exit 1; }
log "cross-node NCCL ok: $NET; $(grep -oE 'RANK0: [^,]+, [^ ]+ [^ ]+ [^ ]+ [^ ]+ -> [0-9.]+ GB/s' "$OUT/nccl_pair.txt" | head -1)"

if [ "$ARM" = fcstore ]; then
k exec "$HEAD" -- bash -c 'cd /opt/py-inference-scheduler && PYTHONPATH=/opt/py-inference-scheduler:/opt/py-inference-scheduler/src python3 -m integration.verl.hook_compat_check 2>/dev/null' > "$OUT/compat.txt" 2>&1
grep -q "HOOK COMPAT CHECK: PASS" "$OUT/compat.txt" || { log "REFUSING: hook compat check failed"; tail -n 12 "$OUT/compat.txt"; exit 1; }
log "compat check PASS: $(grep -E 'flow control:|affinity turns' "$OUT/compat.txt" | tr '\n' ' ')"
fi

k cp "$JT/p40_arm.sh" "$HEAD:/tmp/p40_arm.sh" >/dev/null 2>&1
k cp "$JT/headwatch.sh" "$HEAD:/tmp/headwatch.sh" >/dev/null 2>&1
k exec "$HEAD" -- chmod +x /tmp/p40_arm.sh /tmp/headwatch.sh >/dev/null 2>&1
for wpod in $WORKERS; do
  k cp "$JT/snap13b.py" "$wpod:/tmp/snap13.py" -c ray-worker >/dev/null 2>&1
  for attempt in 1 2 3 4 5; do
    RUNNING=$(k exec "$wpod" -c ray-worker -- bash -c 'pgrep -cf "python3 /tmp/[s]nap13.py" || true' 2>/dev/null | num)
    [ "${RUNNING:-0}" -eq 0 ] && k exec "$wpod" -c ray-worker -- bash -c 'setsid nohup python3 /tmp/snap13.py </dev/null >/tmp/snap13.err 2>&1 & disown' >/dev/null 2>&1
    sleep 15
    SNAPS=$(k exec "$wpod" -c ray-worker -- bash -c 'grep -c "^SNAP " /tmp/snap13.log 2>/dev/null || true' 2>/dev/null | num)
    [ "${SNAPS:-0}" -ge 1 ] && break
  done
  [ "${SNAPS:-0}" -ge 1 ] || { log "REFUSING: snapshotter did not start on $wpod"; exit 1; }
  k exec "$wpod" -c ray-worker -- bash -c 'pgrep -f "verl[.]trainer[.]main_ppo" >/dev/null || rm -f /tmp/lookup_rpc_port_*' >/dev/null 2>&1
done

k exec "$HEAD" -- bash -c "rm -f $DONEF $LOGF" >/dev/null 2>&1
SINCE=$(date -u +%s)
L=$(k exec "$HEAD" -- bash -c "setsid nohup env ARM=$ARM STEPS=$STEPS NNODES=$NNODES TAG=$TAG SAVE_DECODE_CACHE=${SAVE_DECODE_CACHE:-true} CPU_OFFLOAD_GIB=${CPU_OFFLOAD_GIB:-128} P40_EXTRA=\"${P40_EXTRA:-}\" /tmp/p40_arm.sh </dev/null >/dev/null 2>&1 & disown; sleep 10; pgrep -cf '/tmp/p40_arm'" | num)
[ "${L:-0}" -ge 1 ] || { log "launch did not take"; exit 1; }
k exec "$HEAD" -- bash -c "setsid nohup env LOGF=$LOGF DONEF=$DONEF CAP_MIN=$CAP_MIN /tmp/headwatch.sh </dev/null >/tmp/headwatch_${TAG}.out 2>&1 & disown" >/dev/null 2>&1
log "launched $MODE ($STEPS steps, NNODES=$NNODES); headwatch armed (cap ${CAP_MIN}min)"

# PULLSRC lines come from the engine worker processes; Ray forwards their
# stdout to the driver log under the vLLMHttpServer actor's prefix, and keeps
# a copy in the worker pod's Ray logs. Read the driver copy, per node IP.
pullsrc(){ k exec "$HEAD" -- bash -c "grep -a 'PULLSRC local=$1' $LOGF | tail -1 | grep -oE 'cum_keys=[0-9]+ cum_cross=[0-9]+'" 2>/dev/null; }
IP1=$(k get pod "$(echo "$WORKERS" | sed -n 1p)" -o jsonpath='{.status.podIP}'); IP2=$(k get pod "$(echo "$WORKERS" | sed -n 2p)" -o jsonpath='{.status.podIP}')
SNAPCMD='tail -c 600000 /tmp/snap13.log | awk "/^SNAP /{n++} {b[n]=b[n] \"\n\" \$0} END{print b[n-1]}"'
START=$(date +%s); LAST=""; STALL=0; DONE=""
while [ $(( ($(date +%s)-START)/60 )) -lt "$CAP_MIN" ]; do
  sleep 60
  DONE=$(k exec "$HEAD" -- cat "$DONEF" 2>/dev/null); [ -n "$DONE" ] && break
  CRASH=$(k exec "$HEAD" -- bash -c "grep -c 'EngineCore encountered a fatal error' $LOGF || true" 2>/dev/null | num)
  if [ "${CRASH:-0}" -gt 0 ]; then log "ENGINE CRASH - stopping"; DONE="ENGINE_CRASH"
    k exec "$HEAD" -- bash -c 'for p in $(pgrep -f "p40_arm[.]sh") $(pgrep -f "verl[.]trainer[.]main_ppo"); do kill $p; done' >/dev/null 2>&1; break; fi
  S=$(k exec "$HEAD" -- bash -c "grep -acE 'step:[0-9]+ -' $LOGF || true" 2>/dev/null | num)
  VIEWS=$(k exec "$HEAD" -- bash -c "grep -c 'endpoint view: 8/8' $LOGF || true" 2>/dev/null | num)
  MOVES=$(k exec "$HEAD" -- bash -c "grep -ac 'MOVE\[' $LOGF || true" 2>/dev/null | num)
  XMOVES=$(k exec "$HEAD" -- bash -c "grep -ac 'cross_node=1' $LOGF || true" 2>/dev/null | num)
  PARKS=$(k exec "$HEAD" -- bash -c "grep -ac 'FLOWCONTROL park' $LOGF || true" 2>/dev/null | num)
  PS1=$(pullsrc "$IP1"); PS2=$(pullsrc "$IP2")
  INST=$(k exec "$HEAD" -- bash -c "grep -ac 'PULLSRC instrument installed' $LOGF || true" 2>/dev/null | num)
  BLIND=$(k exec "$HEAD" -- bash -c "grep -a 'FLEET\\[' $LOGF | tail -300" 2>/dev/null | python3 -c '
import re,sys
busy=live=0
for line in sys.stdin:
    parts=re.findall(r"(\d+)=kv([\d.]+)/w(\d+)/r(\d+)/p(\d+)/q(\d+)",line)
    if len(parts)<2 or sum(int(p[5]) for p in parts)<20: continue
    busy+=1; live+=any(float(p[1])>0 or int(p[3])>0 for p in parts)
print("BLIND" if busy>=40 and live==0 else f"ok busy={busy} live={live}")')
  if [ "${BLIND:-}" = BLIND ]; then log "METRICS BLIND - stopping"; DONE="METRICS_BLIND"
    k exec "$HEAD" -- bash -c 'for p in $(pgrep -f "p40_arm[.]sh") $(pgrep -f "verl[.]trainer[.]main_ppo"); do kill $p; done' >/dev/null 2>&1; break; fi
  BLK=$(k exec "$(echo "$WORKERS" | sed -n 1p)" -c ray-worker -- bash -c "$SNAPCMD" 2>/dev/null)
  SIG=$(printf %s "$BLK" | grep -E 'num_requests_(running|waiting)\{|generation_tokens_total\{' | md5sum)
  RW=$(printf %s "$BLK" | grep -E 'num_requests_(running|waiting)\{' | awk '{s+=$2} END{print s+0}')
  if [ "$SIG" = "$LAST" ] && [ "${RW:-0}" -gt 0 ]; then STALL=$((STALL+1)); else STALL=0; LAST="$SIG"; fi
  log "  $TAG t=$(( ($(date +%s)-START)/60 ))min steps=${S:-0}/$STEPS views8/8=${VIEWS:-0} moves=${MOVES:-0} cross=${XMOVES:-0} parks=${PARKS:-0} pullsrc_installed=${INST:-0} pull[w1]=${PS1:-?} pull[w2]=${PS2:-?} metrics=${BLIND:-?} run+wait=${RW:-?} stall=${STALL}min"
  if [ "$STALL" -ge "$STALL_MIN" ]; then log "WEDGED"; DONE="WEDGED"
    k exec "$HEAD" -- bash -c 'for p in $(pgrep -f "p40_arm[.]sh") $(pgrep -f "verl[.]trainer[.]main_ppo"); do kill $p; done' >/dev/null 2>&1; break; fi
done
k exec "$HEAD" -- cat "$LOGF" > "$OUT/${ARM}.log" 2>/dev/null
i=0; for wpod in $WORKERS; do i=$((i+1))
  k exec "$wpod" -c ray-worker -- bash -c "awk -v t=$SINCE 'BEGIN{p=0} /^SNAP /{p=(\$2>=t)} p' /tmp/snap13.log" > "$OUT/${ARM}_scrape_w$i.log" 2>/dev/null
  k exec "$wpod" -c ray-worker -- bash -c 'grep -h PULLSRC /tmp/ray/session_latest/logs/worker-*.out 2>/dev/null' > "$OUT/pullsrc_w$i.log" 2>/dev/null
  k get pod "$wpod" -o jsonpath='{.spec.nodeName} {.status.podIP}{"\n"}' >> "$OUT/workers.txt" 2>/dev/null
done
grep -a "PULLSRC" "$OUT/${ARM}.log" > "$OUT/pullsrc_driver.log" 2>/dev/null
k logs deploy/mooncake-master --since=24h 2>/dev/null | grep -iE "mount|segment|register" | tail -n 200 > "$OUT/mooncake_master.log"
echo "${DONE:-TIMEOUT}" > "$OUT/${ARM}.done"
log "finished [${DONE:-TIMEOUT}] steps=$(grep -acE 'step:[0-9]+ -' "$OUT/${ARM}.log" 2>/dev/null) pullsrc lines: driver=$(wc -l < "$OUT/pullsrc_driver.log") w1=$(wc -l < "$OUT/pullsrc_w1.log") w2=$(wc -l < "$OUT/pullsrc_w2.log")"
log "=== gates ==="
if [ "$ARM" = fcstore ]; then
python3 "$JT/sched_gate.py" "$OUT/${ARM}.log" --engines 8 | tee "$OUT/gate.txt"
python3 "$JT/crossnode_gate.py" "$OUT" | tee -a "$OUT/gate.txt"
else
log "gates skipped for ARM=$ARM (control arm: no hook, no tier)"
fi
