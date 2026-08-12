#!/usr/bin/env bash
# Gate-2 smoke: 2 training steps, Qwen3-4B + LoRA r32, verl backend, colocated
# sync loop (simplest gateway-audited configuration), docker sandboxes.
#
# Prerequisites (see integration/rllm/patches/PINS.md):
#   1. rllm @ pinned SHA + routing-policy patch, installed with
#      uv pip install --override integration/rllm/image/overrides.txt -e ".[verl]"
#      plus: uv pip install docker   (DockerSandbox needs docker-py)
#   2. python prepare_r2egym_subset.py   (registers r2egym_smoke, 8 tasks)
#   3. docker pull of the images in the printed manifest
#
# Gate-2 observables after the run:
#   - >=2 steps completed, non-constant reward
#   - sqlite3 $TRACE_DB 'select count(*) from traces where prompt_token_ids is not null'
#     grouped by rollout_id -> token IDs present (harness path, not TITO)
#   - CheckpointEngineManager weight-sync log lines each step

set -euo pipefail
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
TRACE_DB=${TRACE_DB:-/tmp/rllm-smoke-traces.db}
SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -u "$SMOKE_DIR/train_smoke.py" \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    rllm.algorithm.use_rllm=true \
    rllm.gateway.store=sqlite \
    rllm.gateway.db_path="$TRACE_DB" \
    rllm.workflow.n_parallel_tasks=4 \
    rllm.trainer.total_batches=2 \
    rllm.rollout.train.temperature=0.6 \
    rllm.rollout.val.temperature=0.6 \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    +model.name="$MODEL_PATH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora.rank=32 \
    actor_rollout_ref.model.lora.alpha=32 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console']" \
    trainer.project_name=rllm-convergence \
    trainer.experiment_name=gate2-smoke-qwen3-4b \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    "$@"
