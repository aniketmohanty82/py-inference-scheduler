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
# PRESSURE COMES FROM POOL SIZE, NOT MODEL SIZE. Demand/pool is what makes the
# gate fire, and gmu shrinks the pool directly, so the smallest model that
# still runs main's workload gives the most headroom for the same pressure.
#
# Bigger models were tried first and do not fit on an 80GB H100, because the
# FSDP actor and the vLLM engine share every card and the actor all-gathers
# the FULL model at weight sync:
#   32B gmu 0.56 -> sync OOM;  32B gmu 0.50 -> no KV blocks at all
#   14B gmu 0.45 -> actor 40.8GB, OOM;  14B gmu 0.35 -> actor 49.0GB, OOM
# Lowering gmu does not help there: the caching allocator expands into the
# freed space (40.8GB -> 49.0GB), and halving ppo_max_token_len_per_gpu
# changed the failure not at all (byte-identical OOM), so the 49GB is
# structural to update_weights rather than activations.
#
# 7B at gmu 0.22: ~17GB/GPU engine (7.5GB weights + workspace + ~6GB/GPU KV)
# and a ~26GB actor peak = ~43GB of 79GB, with a ~100k-token pool per engine
# against ~524k tokens of demand (16 trajectories/engine at up to 32768
# tokens) = ~5x oversubscribed. Floor is gmu ~0.18, where the pool can no
# longer hold a single max-length request.
# Note this is main's own model: only gmu departs from run_swe.sh.
# Verify against the "KV cache size" line the engine logs at startup.
#
# ARM=fcoff -> BASELINE: verl untouched. The scheduler hook is NOT installed,
#              so verl's own load balancer routes, exactly as upstream.
# ARM=fcon  -> the same verl routing PLUS admission gating: the hook is
#              installed with RLS_ADMISSION_ONLY=1, so it parks while every
#              replica is saturated and then returns no selection, leaving
#              placement to verl's balancer.
#
# Neither arm uses any scorer, so routing is identical in both and admission
# is the only variable. An earlier pair ran waiting_queue 5.0 + least_queue
# 2.0 + kv_cache 1.0 in both arms; that policy already steers away from
# saturated engines, doing much of the gate's job and masking its effect.
set -euo pipefail

ARM=${ARM:?set ARM=fcon|fcoff}
STEPS=${STEPS:-2}
GMU=${GMU:-0.22}
GROUP_N=${GROUP_N:-4}
TURNS=${TURNS:-25}

# ++ (not +) on keys run_swe.sh already sets: hydra rejects a duplicate plain
# assignment, ++ means "add or override".
# Baseline installs no hook at all; treatment adds only the hook.
HOOK_ARG=()
if [ "$ARM" = "fcon" ]; then
  HOOK_ARG=(+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager)
fi

exec bash "$(dirname "$0")/run_swe.sh" \
    "${HOOK_ARG[@]}" \
    ++actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
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
