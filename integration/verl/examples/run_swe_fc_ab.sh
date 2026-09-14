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
# GMU is NOT copied from that pair. 0.317 is H200-specific: on a 141GB card it
# leaves ~9GB of KV after 32GB/GPU of weights, but on an 80GB H100 it is 25GB
# and cannot hold the weights at all.
#
# On an 80GB H100 the usable window is narrow, because the FSDP actor and the
# vLLM engine share the card and must both be resident during weight sync:
#   floor   gmu >= 0.46  KV pool must hold one max-length request
#                        (4096 prompt + 28672 response = 32768 tok, ~256KB/tok
#                        for 32B GQA => ~8.6GB/engine on top of 32GB/GPU weights)
#   ceiling gmu <= 0.58  actor needs ~31.5GB resident plus ~2GB of sync headroom
# 0.56 sat at the ceiling and OOM'd in actor_rollout_update_weights when the
# sync asked for 1.82GB. 0.50 sits mid-window: ~15GB KV/engine (~58k tokens,
# under two max-length requests) which is deliberately tight, since a small
# pool is what creates the pressure the gate exists to relieve.
# Verify against the "KV cache size" line the engine logs at startup.
#
# ARM=fcon  -> scheduler hook + simple_backpressure admission gate
# ARM=fcoff -> scheduler hook, no flow_control block
# The hook is in BOTH arms so admission control is the only variable.
set -euo pipefail

ARM=${ARM:?set ARM=fcon|fcoff}
STEPS=${STEPS:-2}
GMU=${GMU:-0.50}
GROUP_N=${GROUP_N:-4}
TURNS=${TURNS:-25}

# ++ (not +) on keys run_swe.sh already sets: hydra rejects a duplicate plain
# assignment, ++ means "add or override".
exec bash "$(dirname "$0")/run_swe.sh" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager \
    ++actor_rollout_ref.model.path=Qwen/Qwen2.5-32B-Instruct \
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
