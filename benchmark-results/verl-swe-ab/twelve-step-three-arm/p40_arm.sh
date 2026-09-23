#!/bin/bash
# One arm of the swe13 (vLLM 0.29.0 + verl 0.9.0) comparison.
#
# ARM=recompute|stock|store is passed through to run_swe_ab.sh, which owns the
# kv_transfer_config for each. Everything outside that block is identical
# across arms - that invariant is the whole comparison.
#
# Used for both the 2-step smoke gate and the 12-step measurement runs; only
# STEPS differs, so the smoke exercises the exact code path the real run uses.
set -u
cd /opt/py-inference-scheduler

ARM=${ARM:?set ARM=recompute|stock|store}
STEPS=${STEPS:-2}
GMU=${GMU:-0.38}
BATCH=${BATCH:-128}
TAG=${TAG:-smoke}

# A dead engine leaves a socket behind and blocks every later rebind. Only
# safe to clear when no trainer is live.
pgrep -f "verl[.]trainer[.]main_ppo" >/dev/null || rm -f /tmp/lookup_rpc_port_* 2>/dev/null

OVR=(
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.use_dynamic_bsz=True
  # Back to 16384: swe13 compiles flash-attn from source, so SP=2 is restored
  # and the SP group shares 2x16384 - the same budget as every earlier pair.
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=32
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  "trainer.experiment_name=swe13_${TAG}_${ARM}"
)

ARM=$ARM STEPS=$STEPS SWE_DATA_DIR=/opt/swe-data \
  MODEL=Qwen/Qwen2.5-32B-Instruct GMU=$GMU BATCH=$BATCH GROUP_N=4 \
  bash integration/verl/examples/run_swe_ab.sh "${OVR[@]}" \
  > "/tmp/p40_${TAG}_${ARM}.log" 2>&1
echo "rc=$?" > "/tmp/p40_${TAG}_${ARM}.done"
