#!/usr/bin/env bash
# Store-vs-recompute A/B on verl's native SWE agent loop.
#
# ARM=recompute -> no KV offload (baseline)
# ARM=store     -> Mooncake DecodeKVSavingConnector on the rollout engines
#
# The two arms differ by EXACTLY the kv_transfer_config block below (plus the
# experiment name); everything else is shared, which is the invariant the
# whole comparison rests on - see benchmark-results/.../METHODOLOGY.md.
#
# Routing is held constant: the scheduler hook is deliberately NOT installed
# in either arm, so this measures KV offload alone and avoids PR #62's
# "verl 0.8.x untested" hook caveat.
set -euo pipefail

ARM=${ARM:?set ARM=store|recompute}
STEPS=${STEPS:-12}
SWE_DATA_DIR=${SWE_DATA_DIR:-/home/ray/data/swe}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
# gmu 0.30 (not upstream's 0.5): a small KV pool is what creates the eviction
# pressure the store is meant to relieve. Same value both arms.
GMU=${GMU:-0.30}
BATCH=${BATCH:-16}
GROUP_N=${GROUP_N:-4}

STORE_ARGS=()
if [ "$ARM" = "store" ]; then
  STORE_ARGS=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config={kv_connector: DecodeKVSavingConnector, kv_connector_module_path: py_inference_scheduler.datalayer.connectors.mooncake.decode_save, kv_role: kv_both, kv_connector_extra_config: {save_decode_kv: true}}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.prefix_caching_hash_algo=sha256_cbor"
  )
fi

set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$SWE_DATA_DIR/train.parquet" \
    data.val_files="$SWE_DATA_DIR/test.parquet" \
    data.return_raw_chat=True \
    data.train_batch_size="$BATCH" \
    data.max_prompt_length=4096 \
    data.max_response_length=28672 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.seed=42 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n="$GROUP_N" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=25 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.agent.agent_loop_config_path=integration/verl/examples/swe_agent_loop.yaml \
    "${STORE_ARGS[@]}" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='swe-store-ab' \
    trainer.experiment_name="swe_${ARM}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps="$STEPS" \
    "$@"
