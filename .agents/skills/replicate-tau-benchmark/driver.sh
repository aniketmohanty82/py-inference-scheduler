#!/bin/bash
# Driver for replicating the tau-bench router benchmark (cache_aware vs
# round_robin vs py-inference-scheduler) on a GKE 8-GPU pod.
#
# Every phase is idempotent, verifies its own outcome, and fails with a
# pointed error. Run phases in order; re-run any phase after fixing its error.
# See SKILL.md for the phase map and troubleshooting table.
#
# Required env:
#   RLS_CONTEXT   kubectl context of the GKE cluster (must have GMP enabled)
# Optional env:
#   POD           pod name                      (default: tau-repl)
#   REPO_REF      py-inference-scheduler ref    (default: tau-benchmark-v1 tag)
#   RUN_STEPS     rollout steps per arm         (default: 3; paper runs used 20)
#   GEMINI_KEY_FILE  local file with Gemini API key (required by tau-setup)
#   HF_SECRET     k8s secret holding hf_api_token (default: hf-secret)
set -u

POD=${POD:-tau-repl}
REPO_REF=${REPO_REF:-tau-benchmark-v1}
RUN_STEPS=${RUN_STEPS:-3}
HF_SECRET=${HF_SECRET:-hf-secret}
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"
KC() { kubectl --context "$RLS_CONTEXT" "$@"; }
PEXEC() { KC exec "$POD" -- bash -c "$1"; }
die() { echo "FAIL[$PHASE]: $*" >&2; exit 1; }
ok() { echo "OK[$PHASE]: $*"; }

# tau-bench fork pin (JD-ETH fork, feature/litellm-retry branch)
TAU_FORK=https://github.com/JD-ETH/tau-bench
TAU_PIN=09c26a85efd1d65168cfb57865ca2ca278c8153d

require_pod_running() {
  KC get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null | grep -q Running \
    || die "pod $POD not Running (spot preemption? re-run pod-up)"
}

PHASE=${1:-help}
case "$PHASE" in

preflight)
  command -v kubectl >/dev/null || die "kubectl not installed"
  [ -n "${RLS_CONTEXT:-}" ] || die "set RLS_CONTEXT to your GKE cluster context"
  KC get nodes >/dev/null || die "context $RLS_CONTEXT unreachable"
  KC get secret "$HF_SECRET" >/dev/null 2>&1 \
    || die "secret $HF_SECRET missing: kubectl create secret generic $HF_SECRET --from-literal=hf_api_token=hf_..."
  KC get nodes -o custom-columns=N:.metadata.name,GPU:.status.capacity.nvidia\\.com/gpu \
    | grep -q 8 || echo "WARN: no ready 8-GPU node visible; pod-up may wait on autoscaler"
  ok "kubectl, context, $HF_SECRET present"
  ;;

pod-up)
  # Name-override the reference pod spec; PodMonitoring selector matches app label.
  sed -e "s/name: tau-slime$/name: $POD/" -e "s/app: tau-slime$/app: $POD/" \
    "$REPO_ROOT/configs/benchmarks/tau-slime-pod.yaml" | KC apply -f - || die "pod apply failed"
  sed -e "s/name: tau-slime-engines$/name: $POD-engines/" -e "s/app: tau-slime$/app: $POD/" \
    "$REPO_ROOT/configs/benchmarks/tau-slime-podmonitoring.yaml" | KC apply -f - \
    || die "podmonitoring apply failed (is GMP enabled on the cluster?)"
  KC wait --for=condition=Ready "pod/$POD" --timeout=900s \
    || die "pod not Ready in 15min: kubectl describe pod $POD (spot capacity / image pull)"
  ok "pod Ready + PodMonitoring applied"
  ;;

env-verify)
  require_pod_running
  GPUS=$(PEXEC 'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l') || die "nvidia-smi failed (nvidia hostPath mount?)"
  [ "$GPUS" = 8 ] || die "expected 8 GPUs, got $GPUS"
  PEXEC 'python3 -c "import sglang, slime" && python3 -m pip show sglang-router | head -2' \
    || die "slime image missing expected packages"
  ok "8 GPUs, slime/sglang/sglang-router importable"
  ;;

model-prep)
  require_pod_running
  if PEXEC 'test -f /root/model_prep.done'; then ok "already done"; exit 0; fi
  # Runs detached on the pod: survives kubectl disconnect. NEVER background a
  # piped heredoc chain with & (stdin becomes /dev/null and files land empty).
  PEXEC 'nohup setsid bash -c "
    set -ex
    hf download Qwen/Qwen3-14B --local-dir /root/Qwen3-14B
    cd /root/slime && source scripts/models/qwen3-14B.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
      \${MODEL_ARGS[@]} --hf-checkpoint /root/Qwen3-14B --save /root/Qwen3-14B_torch_dist
    touch /root/model_prep.done
  " > /root/model_prep.log 2>&1 < /dev/null &' || die "launch failed"
  ok "model prep launched (~2-3h). Poll: driver.sh model-verify"
  ;;

model-verify)
  require_pod_running
  PEXEC 'test -f /root/model_prep.done' \
    || { PEXEC 'tail -3 /root/model_prep.log'; die "not finished (see log above)"; }
  PEXEC 'ls /root/Qwen3-14B_torch_dist/ | head -3' || die "torch_dist dir missing"
  ok "Qwen3-14B + torch_dist conversion present"
  ;;

tau-setup)
  require_pod_running
  # Key delivery: pre-placed /root/gemini_key wins (e.g. piped from a k8s
  # secret, never touching local disk); else GEMINI_KEY_FILE is copied up.
  if ! PEXEC 'test -s /root/gemini_key'; then
    [ -n "${GEMINI_KEY_FILE:-}" ] && [ -f "$GEMINI_KEY_FILE" ] \
      || die "no key on pod and GEMINI_KEY_FILE unset: pipe a secret to $POD:/root/gemini_key or set GEMINI_KEY_FILE"
    KC cp "$GEMINI_KEY_FILE" "$POD:/root/gemini_key"
  fi
  KC cp "$SKILL_DIR/patches/slime-v0.3.0-patches.diff" "$POD:/root/slime-patches.diff"
  PEXEC "set -e
    if [ ! -d /root/tau-bench-src ]; then
      git clone $TAU_FORK /root/tau-bench-src
    fi
    cd /root/tau-bench-src && git fetch && git checkout $TAU_PIN
    pip install -e /root/tau-bench-src
    cd /root/slime && git diff --quiet || true
    git apply --check /root/slime-patches.diff 2>/dev/null && git apply /root/slime-patches.diff || echo 'patches already applied (or conflict - verify manually)'
    mkdir -p /root/tau-bench
    python3 -c 'from tau_bench.envs import get_env; print(\"tau import OK\")'
    python3 - <<PY
import json
from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
from tau_bench.envs.retail.tasks_dev import TASKS_DEV
def dump(tasks, path):
    with open(path, 'w') as f:
        for i, t in enumerate(tasks):
            row = {'index': i, 'metadata': {'user_id': t.user_id,
                'actions': [{'name': a.name, 'kwargs': a.kwargs} for a in t.actions],
                'instruction': t.instruction, 'outputs': t.outputs}}
            f.write(json.dumps(row) + '\n')
dump(TASKS_TRAIN, '/root/tau-bench/retail_train_tasks.jsonl')
dump(TASKS_DEV, '/root/tau-bench/retail_dev_tasks.jsonl')
print('task jsonls written:', len(TASKS_TRAIN), len(TASKS_DEV))
PY
  " || die "tau setup failed"
  PEXEC '[ $(wc -l < /root/tau-bench/retail_train_tasks.jsonl) = 500 ]' \
    || die "retail_train_tasks.jsonl wrong size (expected 500 rows)"
  ok "tau fork @ ${TAU_PIN:0:8}, slime patched, Gemini key installed"
  ;;

router-deploy)
  require_pod_running
  cd "$REPO_ROOT" && git fetch origin >/dev/null 2>&1
  TMP=$(mktemp -d) && git archive "$REPO_REF" | tar -x -C "$TMP" \
    || die "git archive $REPO_REF failed"
  KC cp "$TMP" "$POD:/root/pis" && rm -rf "$TMP"
  KC cp "$SKILL_DIR/prof_champion.yaml" "$POD:/root/prof_champion.yaml"
  # Boot check runs entirely pod-side, launched detached: a kubectl exec whose
  # foreground command backgrounds a daemon can hang the exec stream forever.
  KC cp "$SKILL_DIR/boot_check.sh" "$POD:/root/boot_check.sh"
  PEXEC 'nohup setsid bash /root/boot_check.sh > /root/boot_check.out 2>&1 < /dev/null & sleep 1; echo LAUNCHED' \
    | grep -q LAUNCHED || die "boot check launch failed"
  sleep 15
  PEXEC 'cat /root/boot_check.result 2>/dev/null' | grep -q PASS \
    || { PEXEC 'cat /root/boot_check.result /root/router_boot_check.log 2>/dev/null | head -8'; die "router boot check failed"; }
  ok "router deployed from $REPO_REF and boots clean"
  ;;

run-vanilla|run-rr|run-ours)
  require_pod_running
  ARM=${PHASE#run-}
  KC cp "$SKILL_DIR/arm_runner.sh" "$POD:/root/arm_runner.sh"
  PEXEC "RUN_STEPS=$RUN_STEPS nohup setsid bash /root/arm_runner.sh $ARM > /root/arm_${ARM}_runner.out 2>&1 < /dev/null & sleep 3
    pgrep -f 'bash /root/arm_runner[.]sh' >/dev/null && echo LAUNCHED" | grep -q LAUNCHED \
    || die "arm runner failed to launch"
  ok "$ARM arm launched ($RUN_STEPS steps, ~10min/step). Poll: driver.sh status"
  ;;

status)
  require_pod_running
  PEXEC 'tail -4 /root/arm_evidence.log 2>/dev/null; ls /root/arm_*.done 2>/dev/null
    L=$(ls -t /root/arm_*.log 2>/dev/null | grep -v runner | head -1)
    [ -n "$L" ] && echo "steps done: $(grep -c "rollout.py:.* perf" $L 2>/dev/null) ($L)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -2'
  ;;

metrics-verify)
  # GMP names use UNDERSCORES (sglang_...), never the colon form engines expose.
  [ -n "${RLS_PROJECT:-}" ] || die "set RLS_PROJECT to the GCP project id"
  # ADC token, not print-access-token: corp CBA policies 401 the latter.
  TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null) \
    || die "run: gcloud auth application-default login"
  R=$(curl -s "https://monitoring.googleapis.com/v1/projects/$RLS_PROJECT/location/global/prometheus/api/v1/query" \
    --data-urlencode "query=sum(rate(sglang_generation_tokens_total[2m]))" \
    -H "Authorization: Bearer $TOKEN")
  echo "$R" | grep -q '"error"' && die "API error (auth/project): $(echo "$R" | head -c 300)"
  echo "$R" | grep -q '"result":\[{' || die "query OK but no samples: PodMonitoring not scraping, or no run active in the last 2m (engines only expose /metrics while running). Response: $(echo "$R" | head -c 200)"
  ok "GMP returning engine samples: $(echo "$R" | head -c 200)"
  ;;

collect)
  require_pod_running
  OUT="$REPO_ROOT/benchmark-results/replication-$(date +%m%d)"
  mkdir -p "$OUT"
  KC exec "$POD" -- bash -c 'cd /root && tar czf - arm_*.log arm_evidence.log router_arm_*.log 2>/dev/null' \
    | tar xzf - -C "$OUT" || die "no run logs found"
  for f in "$OUT"/arm_*.log; do
    [ -f "$f" ] || continue
    echo "== $f"; grep -oE "rollout_time.: [0-9.]+" "$f" | head -25
  done
  ok "logs in $OUT"
  ;;

kill-router)
  # Self-healing: kills the retitled 'scheduler' process AND whatever holds
  # port 8000 (found via /proc/net/tcp inode), surviving zombie routers.
  require_pod_running
  PEXEC 'pkill -x scheduler 2>/dev/null; sleep 2
    INODE=$(awk "\$2 ~ /:1F40\$/ {print \$10}" /proc/net/tcp | head -1)
    if [ -n "$INODE" ]; then
      for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "socket:\[$INODE\]" && kill -9 ${p#/proc/} 2>/dev/null; done
    fi
    sleep 1; curl -s -m 2 localhost:8000/workers || echo "port 8000 free"'
  ok "router killed / port free"
  ;;

*)
  cat <<EOF
Usage: RLS_CONTEXT=<ctx> $0 <phase>
Phases in order:
  preflight pod-up env-verify model-prep model-verify tau-setup
  router-deploy run-vanilla run-rr run-ours status metrics-verify collect
Helpers: kill-router
EOF
  ;;
esac
