#!/bin/bash
# Single arm of the sustained-pressure pair. ARM and STEPS come from the
# environment; every other override is identical between arms so the only
# difference is the connector.
#
# Regime change vs p33 (pressure-pair-v2), and why: that pair was bistable.
# Steps 1-2 ran the KV pool at 93-96% and recomputed 65% of prefill; steps 3-4
# ran it at 20-27% and served 91% from local_cache_hit, on identical work
# (prompt_length/mean, num_turns/mean, perf/total_num_tokens all flat). The
# escape is self-sustaining in both directions, so the regime has to make even
# the LOW-occupancy attractor exceed the pool:
#
#   BATCH 64 -> 160   2.5x trajectories in flight
#   GMU  0.45 -> 0.38 pool 186k -> ~106k tokens/engine (424k total)
#
# Pool does not scale linearly with gmu - weights are a fixed cost. Fitting the
# two recorded points (0.317 -> 34k, 0.45 -> 186k) gives
# tokens/engine ~= 1,143k*gmu - 328k. Together those are 4.4x the occupancy
# ratio, which puts pair-v2's healthy attractor at 88% and its collapsed one
# at 263% - both oversubscribed, so there is no low state to fall into.
set -u
cd /opt/py-inference-scheduler

pgrep -f "verl[.]trainer[.]main_ppo" >/dev/null || rm -f /tmp/lookup_rpc_port_* 2>/dev/null

ARM=${ARM:?}
STEPS=${STEPS:?}
GMU=${GMU:-0.38}
BATCH=${BATCH:-128}
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

ARM=$ARM STEPS=$STEPS SWE_DATA_DIR=/opt/swe-data \
  MODEL=Qwen/Qwen2.5-32B-Instruct GMU=$GMU BATCH=$BATCH GROUP_N=4 \
  bash integration/verl/examples/run_swe_ab.sh "${OVR[@]}" \
  > "/tmp/p34_${ARM}_driver.log" 2>&1
echo "rc=$?" > "/tmp/p34_${ARM}.done"
