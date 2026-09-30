#!/usr/bin/env bash
# Store-vs-recompute A/B on verl's native SWE agent loop.
#
# ARM=recompute -> no KV offload (baseline): verl's own sticky/least-inflight router
# ARM=stock     -> upstream MooncakeStoreConnector, what verl's docs prescribe
# ARM=store     -> ours: upstream + the pull-admission policy
# ARM=sched     -> no KV offload, routing by py-inference-scheduler (the hook)
#                  under ROUTER_CONFIG_PATH; differs from recompute by routing ALONE
# ARM=fcstore   -> our store PLUS the hook: the cross-node KV benchmark
# SAVE_DECODE_CACHE=false turns off the store arms' per-step decode-KV saves
#                   (upstream's save_decode_cache), to price that path on its own.
# ARM=cpuoffload -> vLLM's own CPU KV offload (kv_offloading_size GiB per engine,
#                   native backend), verl's router, no Mooncake: a per-node tier
#                   that cannot serve another node. CPU_OFFLOAD_GIB sets the size.
#                  (NNODES=2). Routing + flow control from ROUTER_CONFIG_PATH.
#
# The arms differ by EXACTLY the kv_transfer_config block below (plus the
# experiment name); everything else is shared, which is the invariant the
# whole comparison rests on - see benchmark-results/.../METHODOLOGY.md.
#
# stock vs store is the kill decision: on vLLM 0.29.0 upstream ships both
# features our connector was built for (reset_cache, save_decode_cache), so
# if those two arms tie, our connector has no reason to exist here.
#
# Sleep is disabled in BOTH arms (free_cache_engine=False): vLLM sleep-mode
# page remapping corrupts the engine under the connector's one-time pinned
# RDMA registration - see benchmark-results/verl-swe-ab/entropy-diagnosis.
#
# Routing is held constant across the three KV arms: the scheduler hook is
# NOT installed in recompute/stock/store, so those measure KV offload alone.
# ARM=sched is the one arm that installs it; against recompute it measures
# routing alone (same no-offload engines, same everything else).
#
# flash-attn is COMPILED FROM SOURCE in the swe13 image (no wheel exists for
# torch 2.13/cu130), so the actor runs the same configuration as every earlier
# pair: remove_padding on, ulysses SP=2, flash_attention_2. The sdpa/SP=1
# detour is gone - sdpa materialized full attention scores on 29k-token
# sequences (33-48 GiB single allocations) and OOM'd all three arms.
#
# trainer.use_v1=False keeps us on verl's legacy main_ppo_v0 TaskRunner. verl
# 0.9.0 defaults use_v1=true, whose TaskRunnerV1.run() unconditionally does
# `import transfer_queue` and forces config.transfer_queue.enable=True - and
# transfer_queue is not on PyPI and is not declared by verl, so that path
# cannot even start here. It is also the wrong path for this comparison:
# verl 0.8.0, which produced every earlier result, has no use_v1 key, no
# TaskRunnerV1 and no transfer_queue import, so the legacy runner is the
# like-for-like choice. Adopting V1 would confound a trainer rewrite with the
# vLLM upgrade we are actually measuring.
set -euo pipefail

ARM=${ARM:?set ARM=store|stock|recompute|sched|fcstore|cpuoffload}
STEPS=${STEPS:-12}
SWE_DATA_DIR=${SWE_DATA_DIR:-/home/ray/data/swe}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
# gmu 0.30 (not upstream's 0.5): a small KV pool is what creates the eviction
# pressure the store is meant to relieve. Same value both arms.
GMU=${GMU:-0.30}
BATCH=${BATCH:-16}
GROUP_N=${GROUP_N:-4}
# Nodes in the Ray cluster the trainer may use; 8 GPUs each. 2 = the
# cross-node KV benchmark (8 engines at tp=2 over two a3-ultra nodes).
NNODES=${NNODES:-1}

# Both offload arms enable save_decode_cache, upstream's own decode-KV knob,
# so arm store differs from arm stock by the pull-admission policy ALONE -
# see RLPullPolicyConnector for why nothing else of ours survives on 0.29.
STORE_ARGS=()
case "$ARM" in
  stock)
    STORE_ARGS=(
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config={kv_connector: MooncakeStoreConnector, kv_role: kv_both, kv_connector_extra_config: {save_decode_cache: ${SAVE_DECODE_CACHE:-true}}}"
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.prefix_caching_hash_algo=sha256_cbor"
    )
    ;;
  store)
    STORE_ARGS=(
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config={kv_connector: RLPullPolicyConnector, kv_connector_module_path: py_inference_scheduler.datalayer.connectors.mooncake.rl_pull_policy, kv_role: kv_both, kv_connector_extra_config: {save_decode_cache: ${SAVE_DECODE_CACHE:-true}}}"
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.prefix_caching_hash_algo=sha256_cbor"
    )
    ;;
  recompute) ;;
  cpuoffload)
    STORE_ARGS=(
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_offloading_size=${CPU_OFFLOAD_GIB:-128}"
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_offloading_backend=native"
    )
    ;;
  sched)
    : "${ROUTER_CONFIG_PATH:?ARM=sched needs ROUTER_CONFIG_PATH in the pod env}"
    STORE_ARGS=(
      "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager"
    )
    ;;
  fcstore)
    : "${ROUTER_CONFIG_PATH:?ARM=fcstore needs ROUTER_CONFIG_PATH in the pod env}"
    STORE_ARGS=(
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config={kv_connector: RLPullPolicyConnector, kv_connector_module_path: py_inference_scheduler.datalayer.connectors.mooncake.rl_pull_policy, kv_role: kv_both, kv_connector_extra_config: {save_decode_cache: ${SAVE_DECODE_CACHE:-true}}}"
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.prefix_caching_hash_algo=sha256_cbor"
      "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager"
    )
    ;;
  *) echo "unknown ARM=$ARM (want store|stock|recompute|sched|fcstore|cpuoffload)" >&2; exit 2 ;;
esac

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
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n="$GROUP_N" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=25 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.agent.agent_loop_config_path=integration/verl/examples/swe_agent_loop.yaml \
    "${STORE_ARGS[@]}" \
    algorithm.use_kl_in_reward=False \
    trainer.use_v1=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='swe-store-ab' \
    trainer.experiment_name="swe_${ARM}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps="$STEPS" \
    "$@"
