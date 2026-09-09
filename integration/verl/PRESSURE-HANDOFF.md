# HANDOFF: high-KV-pressure store-vs-recompute pair (larger model)

Written 2026-09-09 by the session that produced the first valid pair.
Everything below is either recorded fact or explicitly labeled estimate.

## 1. Mission

Run the next store-vs-recompute A/B on verl's native SWE harness with a
LARGER MODEL and REAL KV PRESSURE, on the existing rig. The completed pair
(`benchmark-results/verl-swe-ab/gvisor-nosleep-pair/`) is valid but
low-pressure: `num_preemptions_total` ended at exactly 0, sampled
`kv_cache_usage_perc` peaked at 0.28%, the local prefix cache served 93.77%
of prompt tokens and the external tier 0.22%. The store was measured
harmless (-3.8% gen, traj LLM -5.9% at 10/12, tool-time parity +1.4%) but
had nothing to rescue. Your job: give it something to rescue, and measure.

Deliverable: `benchmark-results/verl-swe-ab/pressure-pair/` on branch
`verl-swe-store-ab` (origin = aniketmohanty82 fork), same README shape as
`gvisor-nosleep-pair/README.md` (validity gates table, per-step results
with N-of-M, by_source split with the exact identity check, S5
interpretation, caveats), committed and pushed
(push recipe: `GIT_ASKPASS=<askpass script echoing gh auth token> git push
https://github.com:443/aniketmohanty82/py-inference-scheduler.git
verl-swe-store-ab` - plain `git push` hangs on SSH key-touch).

## 2. Standards in force (non-negotiable)

- RECORDED, never inferred. Metrics come from driver logs, /metrics
  scrapes, or k8s objects; anything derived is labeled DERIVED. If a
  number wasn't recorded, write "not recorded".
- Valid runs only in benchmark-results; failed/bugged runs are quarantined
  outside the repo (or under an invalid-*/ dir with the failure recorded).
- Both arms differ by EXACTLY the kv_transfer_config flags
  (`integration/verl/examples/run_swe_ab.sh`, ARM=store|recompute).
- Same node, back-to-back, fresh verified-empty mooncake master before the
  store arm, seed 42, image pinned, `pair_compare_verl.py`-style per-step
  deltas with N-of-M consistency.
- Terminology: step = sampling+training; rollout = the whole sampling
  phase; trajectory = one task attempt; batch x n = trajectories/rollout.

## 3. The rig as it stands (all live and verified)

| thing | state |
|---|---|
| cluster | `gke-gpu-rdma-cluster`, region us-south1, context `gke_aniket-gke-dev_us-south1_gke-gpu-rdma-cluster` |
| RayCluster | `swe-ab` from `integration/verl/k8s/swe-raycluster.yaml`; head bench-pool-64, worker 8xH200 a3-ultra (rdma-gpu-pool), 2200Gi limit, IPC_LOCK, mooncake-rdma-8 claim |
| image | `rllm-verl-mooncake:swe2` = vLLM 0.22.1 + verl 0.8.0 + mooncake + upstream PR-62 swe modules + dataset baked at `/opt/swe-data` (256 train/16 eval, registry-rewritten to the AR mirror). Rebuild: `integration/verl/image/build.sh` then `Dockerfile.dataset` |
| sandboxes | agent-sandbox CRs under gVisor in `agents-system`; RBAC = upstream `configs/swe_sandbox_rbac.yaml` PLUS `integration/verl/k8s/swe-sandbox-rbac-rayserviceaccount.yaml` (our SA is `rllm-sandbox-runner`, not default) |
| gvisor pool | `gvisor-sandbox-pool` e2-standard-16, autoscale 1-8, 3 nodes up |
| mooncake | master deployment `mooncake-master`; per-TP-RANK segments from configmap `mooncake-config` (`integration/verl/k8s/cm-mooncake-config.yaml`, currently 128gb x 8 ranks = 1TiB) |
| kernel fix | `integration/verl/k8s/rdma-softlockup-tuning.yaml` DaemonSet (watchdog_thresh=60, softlockup_panic=0) - REQUIRED; without it the node kernel-panics at every store-arm exit (upstream bug d056bc45b62b absent from COS 6.12.y; serial console has the stack) |

## 4. Constraints that will bite you (each one cost this rig a run)

1. **Sleep mode is FORBIDDEN with the store connector.**
   `actor_rollout_ref.rollout.free_cache_engine=False` in BOTH arms. vLLM
   sleep's physical-page remap under the connector's one-time RDMA
   registration corrupts the engine after the first wake (entropy 0.29 ->
   4.3, proven by three isolation runs - `benchmark-results/verl-swe-ab/
   entropy-diagnosis/`). This is THE constraint that makes model sizing a
   memory-budget problem (section 5).
2. **`/opt/py-inference-scheduler` is IMAGE-BAKED.** Editing the git
   worktree changes nothing in-pod. Pass Hydra overrides at invocation
   (the arm script ends in `"$@"`), and pre-flight EVERY run by grepping
   the flag on the live command line:
   `head -1 <driver.log> | tr ' ' '\n' | grep <flag>`.
3. **Smoke-gate the suspect arm on the failure mode you hunt.** The gate
   that works: 2-step STORE smoke requiring rc=0, 2 steps, entropy < 1.0,
   num_turns/mean >= 5, tool_calls/mean > 0. A hollow pair once passed a
   weaker gate with 2-turn zero-tool trajectories (RBAC denial is SILENT -
   no rollout-error). FOR THIS MISSION add a PRESSURE gate: the smoke must
   record `vllm:num_preemptions_total > 0` or sampled
   `kv_cache_usage_perc > 0.9`; otherwise the regime is wrong - retune
   before burning a pair.
4. **Run everything in-pod** (`nohup setsid` a script in the head pod;
   `kubectl cp` drops +x - always `chmod +x` before launching; a mode-644
   script dies as a silent zombie). Operator sessions die; the run must
   not.
5. **Engines bind uvicorn to the POD IP on ephemeral ports** - never
   127.0.0.1. The metrics-scraper sidecar in the raycluster yaml is
   netns-fixed and archives every engine /metrics to its container log
   (Cloud Logging survives everything), but the log ROTATES (~10k lines
   retained): final cumulative counters survive; per-step deltas don't.
   For per-step by_source, also run an in-worker snapshotter
   (port-mapped scrape per vLLMHttpServer pid) plus a host-side puller
   every 5 min. `vllm:prompt_tokens_by_source_total{local_compute,
   local_cache_hit,external_kv_transfer}` sums EXACTLY to
   `vllm:prompt_tokens_total` per engine - always print this identity.
6. **Preemption counters**: verl's `agent_loop/*/num_preempted` is -1 on
   this stack (vLLM 0.22.1 CompletionOutput lacks the field). The recorded
   source is `vllm:num_preemptions_total` from the engine scrape.
7. **Worker pod replacement loses /tmp** (it happened 6 times). Dataset is
   safe (baked); your scripts/snapshotter are not - re-stage after any
   replacement. Health-barrier between phases: wait for `ray status` to
   show 8 GPU before starting the next arm.
8. **Store arm LAST**, recompute first: recompute cannot be corrupted by
   store state and needs no master reset after it.
9. **Sandboxes have NO network egress** (DNS fails in-pod; recorded).
   Model `pip install` attempts burn the 60s SWE_CMD_TIMEOUT_S; affects
   both arms equally. Don't "fix" mid-pair.
10. **Mooncake sizing**: segments are PER TP RANK. nranks x segment +
    4gb local buffer each must fit the 2200Gi container alongside the
    trainer (512gb x 8 OOMKilled the container; 256gb x 8 tripped Ray's
    95% monitor; 128gb x 8 = 1TiB is proven). `PYTHONHASHSEED=0` is
    already in the pod env (store keys are content hashes).
11. Watch for `CREATE_MKEY failed, status no resources(0xf)` in engine
    logs: leaked NIC firmware mkeys used to be cleared by the reboots the
    kernel fix stopped. Remedy: reboot that node once, deliberately.

## 5. Model + pressure design (the actual decision you own)

No-sleep means rollout engine AND FSDP trainer coexist per H200 (141GB).
Budget per GPU: `gmu x 141` for the engine (weights + KV pool) plus
trainer residency. The proven 7B pair ran gmu 0.30, full FT.

**Recommended config: Qwen2.5-32B-Instruct + LoRA, tp=2, gmu ~0.32.**

- Rollout: tp=2 -> 32GB weights/GPU; gmu 0.32 -> 45GB budget -> **~13GB
  KV pool per GPU** (ESTIMATE - read the real "GPU KV cache size" line
  from engine init logs). At ~131KB/token/GPU for a 32B GQA model at tp=2
  (ESTIMATE), that's ~100k tokens of KV per engine against 16 concurrent
  trajectories x up to 29k tokens each = guaranteed eviction and
  preemption. That is the point.
- Trainer: LoRA is mandatory at 32B. From the rllm-convergence memory
  ladder, verbatim or it OOMs/deadlocks:
  - `actor_rollout_ref.model.lora_rank=32` and `lora_alpha=32` as FLAT
    keys (the nested `model.lora.rank` form is silently ignored -> full FT).
  - `actor_rollout_ref.actor.strategy=fsdp2` (fsdp1 LoRA merge gathers the
    whole unsharded model per GPU).
  - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` (already in pod env).
    `expandable_segments` is BANNED (breaks checkpoint-engine IPC weight
    sync). `offload_policy` is BANNED (deadlocks weight sync over RDMA).
    Fused entropy/logit kernels are BANNED (verl #3512, #2656).
  - `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384` (the 32k
    default allocates a ~10GB logits tensor at 152k vocab).
- Keep batch 16 x n 4, 12 steps, max_response 28672, 25 turns, seed 42
  (dataset has 256 rows -> 16 steps of fresh tasks max; do not raise batch
  above 16 without accepting fewer steps).
- Fallback if 32B won't fit no-sleep (smoke OOMs at init and gmu/token
  tuning can't save it): drop to Qwen2.5-14B-Instruct full FT tp=2 at gmu
  0.20-0.25 - still a larger model than 7B and a much smaller KV pool.
  Record whichever regime actually ran; don't force it.
- Tune gmu so that the SMOKE records preemptions > 0. If preemptions are
  extreme (engines thrash, trajectories time out), step gmu up by 0.03.
  Every tuning run is a discarded run - keep them out of the results dir.

## 6. Run recipe (mirror of the proven one)

1. Pre-flight (S3): node Ready + 8 GPUs 0MiB; `master_total_capacity_bytes
   0` after a master restart; dataset gate (`ls /opt/swe-data`); RBAC
   `kubectl auth can-i create sandboxes.agents.x-k8s.io -n agents-system
   --as=system:serviceaccount:default:rllm-sandbox-runner` = yes; DaemonSet
   sysctls live (`cat /proc/sys/kernel/softlockup_panic` = 0 on the GPU
   node); pin `gvisor-sandbox-pool` to a fixed size for the whole pair
   (autoscale events mid-arm are a confound; 64 sandboxes x 1cpu need >=5
   e2-standard-16, or cut requests - sandboxes measured at 2-73 mCPU).
2. In-pod orchestrator (head pod, nohup setsid, .done markers with rc):
   2-step STORE smoke with the work + pressure gates -> health barrier ->
   recompute 12 -> health barrier -> store 12. Master is reset once before
   the smoke; recompute doesn't touch the store; verify-empty again
   between smoke and store arms if the smoke wrote data
   (`kubectl delete pod -l app=mooncake-master` + poll capacity 0 - only
   while no engines hold segments).
3. Watchers (host, run_in_background): 5-min incremental harvest of driver
   logs + snapshotter log + tripwires (pod Error states, >=8
   rollout-error, log frozen 60min). Verify the Hydra command line before
   walking away (constraint 2 above).
4. Harvest: driver logs, sidecar log (`kubectl logs -c metrics-scraper`),
   in-worker snapshotter log, master /metrics. gzip into the results dir,
   README, commit, push.

## 7. What "success" reports

- Both arms 12/12 rc=0, entropy in the 0.2-0.5 band all steps, turns ~40+,
  zero rollout-errors, tool-time delta within a few % (symmetry check).
- RECORDED pressure: `num_preemptions_total` final per engine (nonzero!),
  kv_cache_usage samples, prefix_cache_hits vs queries.
- by_source split with the exact identity, plus mooncake op counters
  (save_put/load_get/failed_keys) and master counters.
- Per-step table: timing_s/gen, timing_s/step, agent_loop generate mean/
  max, tool mean, response_length, with store-vs-recompute deltas and
  N-of-M. Compare the store's delta here vs the low-pressure pair's
  (-3.8% gen / -5.9% traj LLM): the hypothesis under test is that pressure
  widens it.
- If the store arm is WORSE under pressure, that is a finding - record it,
  don't retune it away.

## 8. Reference material

- `benchmark-results/verl-swe-ab/gvisor-nosleep-pair/README.md` - the
  house style and the baseline numbers.
- `benchmark-results/verl-swe-ab/entropy-diagnosis/README.md` - why
  no-sleep; do not relitigate.
- `integration/verl/swe/PINS.md` - PR-62 pin + store key structure (no
  weight version in keys; with LoRA the merged weights DO change per step
  once training has signal - watch offpolicy/KL drift vs the store arm's
  own step 1, and treat rising external hits on stale prefixes as a gate).
- `.claude/skills/swe-rl/SKILL.md` on the PR branch (`git show
  pr-62-swe-rl:.claude/skills/swe-rl/SKILL.md`) - failure-mode catalog.
- Job scratch from the prior session (scripts to reuse nearly verbatim):
  `/usr/local/google/home/aniketmohanty/.claude/jobs/68141fe0/tmp/`
  {gvisor_ab_orchestrator.sh, gvisor_ab_watch.sh, engine_scrape.py,
  worker_snapshotter.sh, pair_compare_verl.py, git_askpass.sh}.
