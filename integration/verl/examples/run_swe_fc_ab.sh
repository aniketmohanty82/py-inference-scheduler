#!/usr/bin/env bash
# Flow-control A/B on verl's native SWE agent loop, at a KV-pressure regime.
#
# Workload = run_swe.sh (main's SWE setup: dataset, agent loop, 4096-token
# prompts, 28672-token responses, tp=2, batch 64). Only the knobs that create
# KV pressure are changed, because main's config is explicitly a "smoke
# training run": measured on 8xH100 it peaks at kv=0.375 against a 0.90 gate
# threshold, so the gate can never fire and the A/B is a guaranteed null.
#
# The pressure regime mirrors the store A/B's validated pressure pair
# (benchmark-results/verl-swe-ab/pressure-pair): Qwen2.5-32B-Instruct + LoRA
# r32/a32, dynamic bsz, safetensors load, no sleep.
#
# MODEL SIZE: 14B, not the store A/B's 32B. On an 80GB H100 a 32B at tp=2 has
# NO working gmu, measured both ways: the FSDP actor and the vLLM engine share
# each card and must both be resident during weight sync.
#   gmu 0.56 (44.2GB/GPU) -> weights+KV fit, but actor_rollout_update_weights
#                            OOM'd with 636MB free needing 1.82GB
#   gmu 0.50 (39.5GB/GPU) -> "No available memory for the cache blocks":
#                            32GB weights plus ~8GB workspace leaves no KV
# The store A/B ran 32B on 141GB H200s for exactly this reason. gmu is a
# FRACTION OF THE CARD and does not port across GPU sizes: their 0.317 is
# 44.7GB/GPU, which would be gmu 0.57 here.
#
# 14B at tp=2 is 14GB/GPU of weights, so the window reopens with room to spare:
# gmu 0.45 gives ~35.5GB/GPU, leaving ~13.5GB/GPU (~27GB, ~138k tokens) of KV
# per engine, while the actor needs only ~18GB of the remaining ~43GB.
# 64 concurrent trajectories over 4 engines is 16/engine at up to 32768 tokens
# = ~524k tokens of demand against a ~138k pool: ~4x oversubscribed, which is
# the pressure the gate exists to relieve.
# Verify against the "KV cache size" line the engine logs at startup.
#
# ARM=fcon  -> scheduler hook + simple_backpressure admission gate
# ARM=fcoff -> scheduler hook, no flow_control block
# The hook is in BOTH arms so admission control is the only variable.
set -euo pipefail

ARM=${ARM:?set ARM=fcon|fcoff}
STEPS=${STEPS:-2}
GMU=${GMU:-0.45}
GROUP_N=${GROUP_N:-4}
TURNS=${TURNS:-25}

# ++ (not +) on keys run_swe.sh already sets: hydra rejects a duplicate plain
# assignment, ++ means "add or override".
exec bash "$(dirname "$0")/run_swe.sh" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager \
    ++actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
    ++actor_rollout_ref.model.lora_rank=32 \
    ++actor_rollout_ref.model.lora_alpha=32 \
    ++actor_rollout_ref.rollout.load_format=safetensors \
    ++actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
    ++actor_rollout_ref.rollout.free_cache_engine=False \
    ++actor_rollout_ref.rollout.n="$GROUP_N" \
    ++actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$TURNS" \
    ++actor_rollout_ref.actor.use_dynamic_bsz=True \
    ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    ++trainer.total_training_steps="$STEPS" \
    ++trainer.test_freq=-1 \
    ++trainer.val_before_train=False \
    ++trainer.save_freq=-1 \
    ++trainer.logger='["console"]' \
    ++trainer.project_name='swe-flow-control-ab' \
    ++trainer.experiment_name="swe_${ARM}" \
    "$@"
