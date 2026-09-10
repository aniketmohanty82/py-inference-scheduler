#!/usr/bin/env bash
# Flow-control A/B on verl's native SWE agent loop.
#
# ARM=fcon  -> scheduler hook + simple_backpressure admission gate
# ARM=fcoff -> scheduler hook, no flow_control block (routing policy identical)
#
# The scheduler HOOK is installed in BOTH arms, so routing policy is held
# constant and the ONLY difference is whether the gate can park a request.
# Which policy is loaded comes from ROUTER_CONFIG_PATH -> the mounted
# ConfigMap, swapped between arms.
#
# No KV offload in either arm (no kv_transfer_config): this isolates admission
# control, not the store.
set -euo pipefail

ARM=${ARM:?set ARM=fcon|fcoff}
STEPS=${STEPS:-3}
SWE_DATA_DIR=${SWE_DATA_DIR:-/home/ray/data/swe}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
# Small KV pool = the eviction pressure the gate is supposed to prevent.
GMU=${GMU:-0.25}
BATCH=${BATCH:-8}
GROUP_N=${GROUP_N:-4}
TURNS=${TURNS:-15}
RESP_LEN=${RESP_LEN:-8192}

set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$SWE_DATA_DIR/train.parquet" \
    data.val_files="$SWE_DATA_DIR/test.parquet" \
    data.return_raw_chat=True \
    data.train_batch_size="$BATCH" \
    data.max_prompt_length=4096 \
    data.max_response_length="$RESP_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.seed=42 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
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
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n="$GROUP_N" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$TURNS" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.agent.agent_loop_config_path=integration/verl/examples/swe_agent_loop.yaml \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='swe-flow-control-ab' \
    trainer.experiment_name="swe_${ARM}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps="$STEPS" \
    "$@"
