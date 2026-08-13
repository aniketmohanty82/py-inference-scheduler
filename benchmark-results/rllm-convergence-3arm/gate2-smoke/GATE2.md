# Gate 2 + Gate 3: PASSED 2026-08-12 (attempt 8)

Full RL loop (rollout → reward → update → weight sync) through the Model
Gateway on the rls-ab-west cluster: Qwen3-4B + LoRA r32, verl backend,
colocated sync, 1×8×H200 spot node, 8 frozen r2egym tasks, R2E-Gym task
containers as k8s pods (vendored backend). RayJob `rllm-gate2-smoke`,
image rllm-verl-mooncake:dev (rllm @1d1109a6 + vendored patch), branch
commit 3dafd46. Full metrics: submitter.log.

## Observables

- **Training step 1 completed** with a real actor update: 32 trajectories in
  8 GRPO groups of 4, 3.26M tokens, `global_seqlen/mean 407k`,
  `mfu 0.299`, `timing_s/step 770`, entropy 0.128. Step 2 (validation) ran;
  job SUCCEEDED at `rllm.trainer.total_batches=2`.
- **Harness/gateway path proven, no TITO bypass**: enrich succeeded
  (`batch/empty: 0.0`, 509 rows), off-policy diagnostics computed from
  gateway-captured token IDs (`rollout_probs_pearson 0.70`). Trace DB grew
  to 142MB and the traces table is consumed by the trainer after enrichment.
- **Weight sync**: `timing_s/update_weights 2.52` (verl CheckpointEngineManager).
- **Multi-turn agentic shape** (the overscheduling target workload):
  6-37 turns/trajectory (mean ~16), prompts to 13.2k tokens, all rollouts
  ENV_DONE, sandbox pod setup ~15s / teardown <0.1s.
- **Gate 3 (engine-kwargs passthrough) proven en route**:
  `++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true`
  and `tool_call_parser=hermes` reached the `vllm serve` argv (tool-choice
  400s disappeared) - the same channel arm C uses for `kv_transfer_config` +
  `scheduler_cls`. Plain scalar kwargs; nested-dict form still untested.

## Caveats for the 3-arm runs

1. **Reward was 0.0 on all 32 rollouts** (`solve_none: 1.0`, every group
   "too_hard"): Qwen3-4B solves none of the 8 orange3 tasks at a 4096-token
   output cap. Mechanics fine; no learning signal. The 32B arms need
   reward variance - verify early (Gate 6a), consider r2egym_lite or a
   pass-rate-screened subset if 32B also flatlines.
2. `prompt_length/clip_ratio 0.53` at `data.max_prompt_length=8192` - half
   the training rows were prompt-truncated. Raise for the arms (SWE prompts
   reach 13k+ by mid-trajectory).
3. Trace table is emptied by consumption: token-audit harvesting must copy
   the DB **during** the run or rely on training metrics.

## Failure ladder (all committed, each fix one commit)

head OOM on dataset prep (8cb2f1a) → RFC-1123 pod names (e764bd9) →
pods/exec GET RBAC (c286711) → verifier task dirs baked at RLLM_HOME
(e083c65) → task_from_row honors task_path [vendored patch] (4706527) →
vllm tool-choice flags via engine_kwargs (e4e1feb) → 4096 output cap
(3dafd46).
