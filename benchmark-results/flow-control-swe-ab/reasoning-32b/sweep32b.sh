#!/bin/bash
# Reasoning-scale flow-control A/B: Qwen3-32B on swe13, batch 64 x n4 = 256
# trajectories, 25 turns, 2 steps per arm. The gate (kv 0.90 / w 4) is the
# only difference between arms. This is the regime the whole campaign was
# aiming at: 32B recompute plus long contexts make each preemption ~5x
# costlier than the 7B grid, where the gate's preemption cut was real but
# too cheap to show up in throughput.
# Self-detach: kubectl exec's stream stays open while any child holds its fds.
exec </dev/null >/dev/null 2>&1
export RAY_ADDRESS=http://127.0.0.1:8265
OUT=/work/sweep32b
mkdir -p "$OUT"
cd /work

reset_sandboxes() {
  python3 /work/reset_sandboxes.py >> "$OUT/progress.txt" 2>&1
}

run() {
  local name=$1 env=$2
  reset_sandboxes
  echo "[$(date -u +%H:%M:%S)] $name starting (pool reset above)" >> "$OUT/progress.txt"
  for attempt in 1 2; do
    echo "[$(date -u +%H:%M:%S)] $name attempt=$attempt starting" >> "$OUT/progress.txt"
    ray job submit --address http://127.0.0.1:8265 \
        --runtime-env "$env" \
        -- bash -c "cd /work && exec bash integration/verl/examples/run_swe_fc_ab.sh ++actor_rollout_ref.model.path=Qwen/Qwen3-32B ++actor_rollout_ref.actor.strategy=fsdp2 ++actor_rollout_ref.rollout.tensor_model_parallel_size=4 ++trainer.nnodes=2 ++trainer.use_v1=False ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 ++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 ++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768" \
        > "$OUT/${name}.attempt${attempt}.log" 2>&1
    if [ $? -eq 0 ]; then
      cp "$OUT/${name}.attempt${attempt}.log" "$OUT/${name}.FINAL.log"
      echo "[$(date -u +%H:%M:%S)] $name SUCCEEDED" >> "$OUT/progress.txt"
      return 0
    fi
    echo "[$(date -u +%H:%M:%S)] $name attempt=$attempt FAILED" >> "$OUT/progress.txt"
    sleep 120
  done
  echo "[$(date -u +%H:%M:%S)] $name EXHAUSTED" >> "$OUT/progress.txt"
}

echo "=== 32B reasoning-scale A/B started $(date -u) ===" >> "$OUT/progress.txt"
run baseline-32b integration/verl/examples/runtime-env-32b-off.yaml
run gate-kv90-32b integration/verl/examples/runtime-env-32b-on.yaml
echo "=== 32B reasoning-scale A/B finished $(date -u) ===" >> "$OUT/progress.txt"
