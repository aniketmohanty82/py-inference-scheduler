#!/bin/bash
# Sustained-pressure smoke: 2-step STORE run at the p34 regime.
#
# The work gates are unchanged from p33. What is new is that the PRESSURE gate
# is no longer "preemptions happened somewhere in the run" - pair-v2 passed
# that gate and still spent half its steps in a healthy, low-occupancy state
# that made the whole comparison a wash. The pressure gate now lives in
# pressure_gate.py, is judged host-side against the engine scrape, and demands
# occupancy be sustained across the run rather than merely reached.
set -u
cd /opt/py-inference-scheduler

# A crashed engine leaves its unix socket file behind and the next launch
# fails to bind it ("Address already in use"); no process owns these once the
# run is gone.
pgrep -f "verl[.]trainer[.]main_ppo" >/dev/null || rm -f /tmp/lookup_rpc_port_* 2>/dev/null

OVR=(
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=32
  actor_rollout_ref.actor.ppo_mini_batch_size=32
)

ARM=store STEPS=2 SWE_DATA_DIR=/opt/swe-data \
  MODEL=Qwen/Qwen2.5-32B-Instruct GMU=${GMU:-0.38} BATCH=${BATCH:-128} GROUP_N=4 \
  bash integration/verl/examples/run_swe_ab.sh "${OVR[@]}" \
  > /tmp/p34_smoke.log 2>&1
RC=$?

BAD=$(grep -aoE "actor/entropy:[0-9.]+" /tmp/p34_smoke.log \
      | cut -d: -f2 | awk '$1 >= 1.0' | wc -l)
STEPS_DONE=$(grep -acE "step:[0-9]+ -" /tmp/p34_smoke.log)
TURNS=$(grep -aoE "num_turns/mean:[0-9.]+" /tmp/p34_smoke.log \
        | cut -d: -f2 | awk '$1 < 5' | wc -l)
NOTOOL=$(grep -aoE "timing_s/agent_loop/tool_calls/mean:[0-9.e-]+" /tmp/p34_smoke.log \
         | cut -d: -f2 | awk '$1 <= 0' | wc -l)
echo "rc=$RC steps=$STEPS_DONE high_entropy=$BAD low_turn_steps=$TURNS no_tool_steps=$NOTOOL" \
  > /tmp/p34_smoke.done
