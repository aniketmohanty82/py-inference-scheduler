#!/usr/bin/env bash
# Flow-control A/B on verl's native SWE agent loop.
#
# This is a THIN WRAPPER around run_swe.sh -- the workload (model, batch 64 x
# n8, 28672-token responses, 32 turns, gmu 0.5, tp=2) is main's SWE setup,
# unmodified. Drifting from it would make the numbers incomparable to every
# other SWE result in this repo.
#
# ARM=fcon  -> scheduler hook + simple_backpressure admission gate
# ARM=fcoff -> scheduler hook, no flow_control block
#
# The hook is installed in BOTH arms (unlike run_swe.sh's scheduler A/B, which
# toggles the hook itself): holding routing constant is the only way to
# attribute a delta to admission control rather than to routing. Which policy
# loads comes from ROUTER_CONFIG_PATH, set per arm in the runtime env.
#
# Operational overrides only, all justified:
#   total_training_steps  - A/B length, not a workload change
#   test_freq/val_before_train - validation spawns 500 sandboxes at once
#   save_freq             - no checkpoints needed to compare rollout timing
#   logger                - console only; W&B is backfilled from logs
#   experiment_name       - must differ per arm
set -euo pipefail

ARM=${ARM:?set ARM=fcon|fcoff}
STEPS=${STEPS:-4}

# free_cache_engine=False (no sleep mode) is required on 80GB cards: run_swe.sh
# is calibrated for the 141GB H200s the SWE harness was built on, and with
# gmu 0.5 plus the FSDP actor and ref model resident, vLLM's post-training
# wake_up OOMs in cumem_allocator on an H100. Applied to BOTH arms, and it is
# not a workload change -- the KV pool, batch, n, response length and turn
# count are untouched, so in-rollout KV pressure is identical. This also
# matches run_swe_ab.sh, which runs no-sleep in both of its arms.
#
# ++ (not +) on keys run_swe.sh already sets: hydra errors on a duplicate
# plain assignment, and ++ means "add or override".
exec bash "$(dirname "$0")/run_swe.sh" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager \
    ++actor_rollout_ref.rollout.free_cache_engine=False \
    ++trainer.total_training_steps="$STEPS" \
    ++trainer.test_freq=-1 \
    ++trainer.val_before_train=False \
    ++trainer.save_freq=-1 \
    ++trainer.logger='["console"]' \
    ++trainer.project_name='swe-flow-control-ab' \
    ++trainer.experiment_name="swe_${ARM}" \
    "$@"
