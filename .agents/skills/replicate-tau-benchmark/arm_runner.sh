#!/bin/bash
# Pod-side runner for one benchmark arm: vanilla | rr | ours.
# Generates the slime run script from the verified template, cold-starts the
# arm, samples engine counters every 20s, and logs evidence. Resumable via
# /root/arm_<arm>.done markers.
ARM=$1
RUN_STEPS=${RUN_STEPS:-3}
EV=/root/arm_evidence.log
[ -f /root/arm_$ARM.done ] && { echo "arm $ARM already done"; exit 0; }

kill_router() {
  pkill -x scheduler 2>/dev/null; sleep 2
  INODE=$(awk "\$2 ~ /:1F40\$/ {print \$10}" /proc/net/tcp | head -1)
  if [ -n "$INODE" ]; then
    for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "socket:\[$INODE\]" && kill -9 ${p#/proc/} 2>/dev/null; done
    sleep 1
  fi
}

# Arm-specific sglang-router flags for the slime job.
case "$ARM" in
  vanilla) ROUTER_LINES="" ;;
  rr)      ROUTER_LINES="   --router-policy round_robin" ;;
  ours)    ROUTER_LINES="   --sglang-router-ip 127.0.0.1
   --sglang-router-port 8000" ;;
  *) echo "usage: arm_runner.sh vanilla|rr|ours"; exit 1 ;;
esac

# The verified batch-32 tau-retail GRPO job (see SKILL.md workload table).
# save-interval MUST exceed num-rollout or the run pays a checkpoint save.
cat > /root/slime/examples/tau-bench/run_arm.sh <<EOF
#!/bin/bash
pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 python
sleep 3; pkill -9 ray; pkill -9 python
set -ex
rm -rf /root/Qwen3-14B_slime   # cold start from ref weights every arm
export PYTHONUNBUFFERED=1
SCRIPT_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "\${SCRIPT_DIR}/../../scripts/models/qwen3-14B.sh"
CKPT_ARGS=( --hf-checkpoint /root/Qwen3-14B/ --ref-load /root/Qwen3-14B_torch_dist/
   --load /root/Qwen3-14B_slime/ --save /root/Qwen3-14B_slime/ --save-interval 100 )
ROLLOUT_ARGS=( --prompt-data /root/tau-bench/retail_train_tasks.jsonl --input-key index
   --rollout-shuffle --num-rollout $RUN_STEPS --rollout-batch-size 32 --n-samples-per-prompt 8
   --rollout-max-response-len 1024 --rollout-temperature 1 --global-batch-size 256
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data )
EVAL_ARGS=( --eval-interval 5 --eval-prompt-data retail-dev /root/tau-bench/retail_dev_tasks.jsonl
   --n-samples-per-eval-prompt 1 --eval-max-response-len 1024 --eval-top-k 1 )
PERF_ARGS=( --tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 1
   --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu 9216 )
GRPO_ARGS=( --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
   --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 )
OPTIMIZER_ARGS=( --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98 )
SGLANG_ARGS=( --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.6
$ROUTER_LINES
)
MISC_ARGS=( --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32 --attention-backend flash )
CUSTOM_ARGS=( --custom-generate-function-path generate_with_tau.generate )
export MASTER_ADDR=\${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=8
ray start --head --node-ip-address \${MASTER_ADDR} --num-gpus \${NUM_GPUS} --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir /root/shared/ray_temp
RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"/root/Megatron-LM/:\${SCRIPT_DIR}\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="\${RUNTIME_ENV_JSON}" \
  -- python3 train.py --actor-num-nodes 1 --actor-num-gpus-per-node \${NUM_GPUS} \
  --rollout-num-gpus \${NUM_GPUS} --colocate \${MODEL_ARGS[@]} \${CKPT_ARGS[@]} \${ROLLOUT_ARGS[@]} \
  \${OPTIMIZER_ARGS[@]} \${GRPO_ARGS[@]} \${DISTRIBUTED_ARGS[@]} \${PERF_ARGS[@]} \${EVAL_ARGS[@]} \
  \${SGLANG_ARGS[@]} \${MISC_ARGS[@]} \${CUSTOM_ARGS[@]}
EOF
chmod +x /root/slime/examples/tau-bench/run_arm.sh

echo "[$(date -u +%FT%T)] ARM $ARM start ($RUN_STEPS steps)" >> $EV
kill_router
rm -f /root/arm_$ARM.log /root/arm_metrics_$ARM.log

if [ "$ARM" = ours ]; then
  rm -f /root/router.log
  cd /root/pis && nohup setsid python -m integration.slime --host 0.0.0.0 --port 8000 \
    --config /root/prof_champion.yaml > /root/router.log 2>&1 < /dev/null &
  sleep 8
  W=$(curl -s -m 3 localhost:8000/workers)
  echo "[$(date -u +%FT%T)] ARM ours router: workers=$W" >> $EV
  if [ "$W" != '{"workers":[]}' ]; then echo "[$(date -u +%FT%T)] ROUTER FAIL - abort" >> $EV; exit 1; fi
fi

# 20s engine-counter sampler (cache-hit source): prompt vs cached tokens.
echo running > /root/arm_state_$ARM
POD_IP=$(hostname -i | awk '{print $1}')
nohup setsid bash -c "while ! grep -q done /root/arm_state_$ARM 2>/dev/null; do echo \"=== \$(date -u +%FT%T) ===\"; for port in 15000 15002 15004 15006 15008 15010 15012 15014; do m=\$(curl -s -m 2 $POD_IP:\$port/metrics 2>/dev/null); pt=\$(echo \"\$m\" | grep -E \"^sglang:prompt_tokens_total\" | awk \"{print \\\$2}\"); ct=\$(echo \"\$m\" | grep -E \"^sglang:cached_tokens_total\" | awk \"{print \\\$2}\"); [ -n \"\$pt\" ] && echo \"\$port prompt=\$pt cached=\$ct\"; done; sleep 20; done >> /root/arm_metrics_$ARM.log 2>&1" < /dev/null > /dev/null 2>&1 &

cd /root/slime && bash examples/tau-bench/run_arm.sh > /root/arm_$ARM.log 2>&1
# ray job submit can exit while the job still runs (log-stream drop): wait
# for the real job to finish, then pull authoritative logs from ray itself.
WAITED=0
while ray job list 2>/dev/null | grep -q RUNNING && [ $WAITED -lt 5400 ]; do
  sleep 60; WAITED=$((WAITED+60))
done
JID=$(ray job list 2>/dev/null | grep -oE "raysubmit_[A-Za-z0-9]+" | tail -1)
[ -n "$JID" ] && ray job logs "$JID" > /root/arm_${ARM}_rayjob.log 2>/dev/null
STEPS=$(grep -c "rollout_time" /root/arm_${ARM}_rayjob.log 2>/dev/null)
[ "${STEPS:-0}" -gt 0 ] || STEPS=$(grep -c "rollout_time" /root/arm_$ARM.log)
echo done > /root/arm_state_$ARM
[ "$ARM" = ours ] && cp /root/router.log /root/router_arm_ours.log
kill_router
touch /root/arm_$ARM.done
if [ "$STEPS" -lt "$RUN_STEPS" ]; then
  echo "[$(date -u +%FT%T)] ARM $ARM done SHORT: $STEPS/$RUN_STEPS steps" >> $EV
else
  echo "[$(date -u +%FT%T)] ARM $ARM done: $STEPS steps" >> $EV
fi
