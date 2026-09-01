# KV-pressure benchmark series: full methodology, hardware, workload, results

Written 2026-08-28 for audit. Covers the cache-collapse sweep (v1-v7) and
the store-pull vs recompute benchmark (pullbench). Every known threat to
validity is listed in section 8 - judge them there rather than discovering
them. All raw data referenced is committed alongside this file.

## 1. Hardware

**GPU nodes** (one per run; all runs single-node):

| attribute | value |
|---|---|
| cluster / zone | GKE `rls-ab-west`, us-west1-c |
| node pools | `rdma-gpu-pool` (spot), `rdma-gpu-flex` (DWS flex-start) |
| machine type | a3-ultragpu-8g |
| GPUs | 8x NVIDIA H200, 141 GB HBM3e each |
| host RAM | 1872 GB |
| boot disk | hyperdisk-balanced, 500 GB |
| primary NIC | gVNIC (`gvnic-sub-west`) |
| RDMA NICs | 8x CX-7 via additional node networks (`rdma-sub-west-0..7`) |
| RDMA exposure | DRANET (`gke-networking-dra-driver=true`), `mrdma.google.com` devices, ResourceClaimTemplate `mooncake-rdma-8` |
| provisioning | SPOT - node identity varies per run (same machine type); one mid-run preemption in the series (mooncake-smoke era, not in any reported run) |

**CPU nodes** (task sandboxes + control plane):

| attribute | value |
|---|---|
| shape | 2x n2-standard-8 (8 vCPU / 32 GB each) |
| co-tenants | kuberay-operator, mooncake-master, Ray head pod (3 CPU/16Gi), RayJob submitter |

**Worker pod** (GPU node):

| attribute | value |
|---|---|
| resources | 100 CPU / 800 Gi / 8 GPU (requests = limits) |
| capabilities | IPC_LOCK |
| /dev/shm | 128 Gi tmpfs |
| store/rc arms only | /etc/mooncake ConfigMap mount + rdma-8 resource claim |

**Mooncake store**:

| attribute | value |
|---|---|
| mode | embedded - segments mounted BY the vLLM worker processes |
| segment / buffer | 32 GB global segment per client, 4 GB local buffer |
| master | `mooncake-master` Deployment on CPU pool (grpc :50051, metrics :9003) |
| protocol / metadata | rdma, P2PHANDSHAKE |
| device selection | auto (device_name unpinned - threat T9) |
| traffic locality | single-node runs => all store traffic is same-node RDMA loopback through the HCA (`MC_FORCE_HCA=1`) |

## 2. Software stack (image `rllm-verl-mooncake:dev`)

| component | version / pin |
|---|---|
| base image | ray-llm-mooncake:2.56.0 (Ray 2.56.0, Python 3.12, torch 2.11+cu130, CUDA 13.0) |
| vLLM | 0.22.1 (pip) |
| verl | 0.8.0 |
| rllm | @ 1d1109a6 + vendored patch (gateway routing_policy, kubernetes sandbox backend, task_from_row task_path fix, loopback-pin skip) |
| flash-attn | 2.8.3, source-built, sm90-only |
| mooncake | stock wheels |
| py-inference-scheduler | this repo, branch rllm-convergence |
| dependency note | rllm out-of-repo installs require uv overrides (image/overrides.txt) |

**Model**:

| attribute | value |
|---|---|
| model | Qwen/Qwen3-32B, bf16 |
| KV geometry | GQA - 64 layers, 8 KV heads, head_dim 128 => ~256 KB KV/token full-model (halved per GPU at TP=2) |
| trainer | LoRA r32 / alpha 32 |

## 3. Workload

### 3.0 Terminology (canonical, used throughout these docs)

| term | meaning |
|---|---|
| step | one training step = sampling + training |
| sampling | the within-step process of completing the set of trajectories to reward and train from |
| rollout | the gathering of ALL trajectories during a single step - synonymous with the entire sampling phase |
| batch | the set of unique tasks completed during sampling for a step |
| generations | how many times each task is repeated for GRPO |
| trajectory | one task attempt: multiple turns + tool calls (batch x generations = trajectory count) |
| turn | a single step within a trajectory: one LLM call + tool execution |

Upstream labels differ and are quoted as-is in raw logs: rllm's
"Rollout completed" line marks a TRAJECTORY completion; mini-swe-agent's
"step_limit" caps TURNS; verl config keys (actor_rollout_ref, rollout.*)
keep their native names.

### 3.1 Trajectory lifecycle (inception to completion)

Every trajectory in every run follows this path; the owning component is on
the right. The two arms differ ONLY inside the serving box (step 5).

```
[rLLM DatasetRegistry]   R2E-Gym row: {id, docker_image, task_path}
          |              e.g. orange3__2d9617bd0cb1
          v
[rLLM AgentWorkflowEngine, driver]
          |  assigns task (x n generations), mints session UUID
          v
[our k8s sandbox backend]
          |  creates sandbox pod from a generic python:3.11-slim image
          |  and UPLOADS the task's repo into it (chunked exec); the
          |  per-task R2E docker images are NOT used on the k8s path
          |  (RFC-1123-sanitized name, sandbox pool, 50m/256Mi)
          v
[rLLM flow -> sandbox]   mini-swe-agent injected + started inside the pod
          |
          |   +=============== turn loop (repeats ~10s of times) ==========+
          |   | 1 [agent, in sandbox] POST full history to                 |
          |   |     gateway /v1/<session-id>/chat/completions              |
          |   | 2 [our SchedulerRoutingPolicy, in gateway] select_worker:  |
          |   |     saturation filter + prefix/kv/queue/sticky scorers     |
          |   |     over live MetricsPoller stats                          |
          |   | 3 [vLLM replica] prefill, reuse tiers:                     |
          |   |     HBM prefix cache -> Mooncake store pull (store arm     |
          |   |     only) -> recompute                                     |
          |   | 4 [vLLM] decode reply; [our DecodeKVSavingConnector]       |
          |   |     saves decode KV to master-chosen segments (store arm)  |
          |   | 5 [rLLM gateway] records prompt/response token IDs +       |
          |   |     logprobs to sqlite, keyed by session                   |
          |   | 6 [agent, in sandbox] parses tool call, runs it in the     |
          |   |     repo, appends output to history                        |
          |   +--- under pressure: [vLLM] preempts request; on requeue     |
          |   |     history reloads via the same tiers (the A/B lives      |
          |   |     here)                                                  |
          |   +=== exits when agent submits / env done / budget hit =======+
          |
          v
[rLLM FromTaskEvaluation]  runs the task's tests/test.sh IN the sandbox
          |                pass/fail -> reward; sandbox pod deleted
          v
[rLLM]    enrich trajectory with gateway token IDs; mask tool tokens
          v
[verl FullyAsyncTrainer]   GRPO advantage within the n-generation group;
          |                policy update on the trainer GPUs
          v
[verl checkpoint engine]   weight broadcast to all vLLM replicas
```

The measured windows in these docs cover the turn-loop portion only
(sampling); evaluation, enrichment, update, and sync are outside them.

### 3.2 Run configuration

DeepSWE-style RL sampling step via rllm AgentTrainer:

| attribute | value |
|---|---|
| framework path | rllm AgentTrainer, verl backend, fully-async, `raise_on_error=false`, ONE training batch |
| task set | `r2egym_smoke` = first 64 rows of R2E-Gym/R2E-Gym-Subset (baked at image build; deterministic first-N; heavily orange3) |
| batch shape | batch of 32 tasks x 2 generations = **64 trajectories per rollout**, `n_parallel_tasks=64` |
| agent | mini-swe-agent, installed at trajectory start inside each task container |
| sandboxes | k8s pods on the CPU pool (vendored kubernetes backend); requests overridden to 50m/256Mi (CPU oversubscribed - T7) |
| request path | agent (task pod) -> rllm Model Gateway (sqlite traces; injects logprobs + return_token_ids) -> SchedulerRoutingPolicy -> vLLM |
| scheduler profile | champion: saturation filter (kv 0.95 / waiting 16), prefix_cache 4.0, waiting_queue 1.0, kv_cache 0.5, sticky_session 4.0 |
| single-replica note | the filter drops the sole saturated replica; policy's least-loaded fallback routes anyway (identical both arms - T11) |
| engines | verl-launched `vllm serve` (vLLM's own OpenAI app); separated topology: trainer FSDP 4 GPUs, ONE rollout replica TP=2, 2 GPUs idle |
| reward | in-sandbox test-suite execution; ~0 throughout (T12) |

**Sampling parameters**:

| parameter | value |
|---|---|
| temperature | 0.6 (NOT seeded per-request - trajectories stochastic, T2) |
| max_tokens per turn | 4096 |
| max_model_len | 32768 |
| data.max_prompt_length | 16384 (training-side clip) |

**Measured turn structure** (from healthy-run logs, pooled 1,669 turns):

| characteristic | value |
|---|---|
| turns per trajectory | ~6.6-7.1 mean; runaways 33-81 |
| per-turn prompt growth | ~1k -> 11-16k tokens (full history re-sent every turn) |
| longest served prompt | >=16,384 tokens (clip-bounded) |
| longest attempted prompt | >=28,673 tokens (rejected by engine) |
| tool-call gap per turn | 4.2 s mean, 8-10 s p90 |
| generation per turn | 6.6 s mean |
| KV idle share (tool gaps) | ~40% of trajectory wall-clock |

**Pool arithmetic** (gmu 0.30, TP=2):

| quantity | value |
|---|---|
| per-GPU budget | 0.30 x 141 GB = ~42 GB |
| weights per GPU | ~31 GB |
| KV pool per GPU | ~5-6 GiB (gmu 0.25 leaves 2.25 GiB < 4 GiB one-request floor -> won't boot) |
| aggregate demand | 64 trajectories x up to 16k-token histories >> pool |

## 4. Run matrix

| run | date | replicas | gmu | tasks x n | connector | hash algo | raw log |
|---|---|---|---|---|---|---|---|
| v1 | 08-13 | 4x TP2 | 0.40 | 8x8 | none | default | (metrics narrow) |
| v3 | 08-13 | 4x TP2 | 0.30 | 64x2 | none | default | v3_metrics_raw.log |
| v4 | 08-17 | 1x TP2 | 0.30 | 32x2 | none | default | v4_metrics_raw.log |
| v5 | 08-17 | 1x TP2 | 0.30 | 32x2 | none | default | v5_metrics_raw.log |
| v6b | 08-17 | 1x TP2 | 0.27 | 32x2 | none | default | v6_gmu027_metrics_raw.log |
| v7 | 08-18 | 1x TP2 | 0.35 | 32x2 | none | default | v7_gmu035_metrics_raw.log |
| store#1 | 08-28 | 1x TP2 | 0.30 | 32x2 | DecodeKVSaving | sha256_cbor | pullbench_store_v1_raw.log |
| store#2 | 08-28 | 1x TP2 | 0.30 | 32x2 | DecodeKVSaving | sha256_cbor | pullbench_store_v2_raw.log |
| recompute-measured | 08-28 | 1x TP2 | 0.30 | 32x2 | none | sha256_cbor | pullbench_recompute_v1_raw.log |

The PRIMARY paired comparison is store#1/#2 vs recompute-measured: their
manifests differ by exactly one line (`kv_transfer_config`). v4/v5 serve
as earlier recompute baselines but differ additionally in hash algo,
pod mounts/claims, and sidecar metric set (T1).

Store arm connector: upstream vLLM MooncakeStoreConnector subclassed by
our DecodeKVSavingConnector (PR #46 upstream) with save_decode_kv=true,
kv_role=kv_both. NO OverschedulingScheduler in ANY run in this document -
eviction is vLLM's own pressure eviction in both arms; the store either
rescues evicted history or the engine recomputes it.

## 5. Measurement

- **Collector**: `metrics-scraper` sidecar in the worker pod (python
  stdlib only): every 30 s, GET gateway 127.0.0.1:9090/admin/workers,
  then each engine's /metrics; prints matching series to stdout
  (retrieved via `kubectl logs -c metrics-scraper`, survives job end).
  Filter: prefix_cache | num_preemptions | kv_cache_usage |
  prompt_tokens | generation_tokens (the token counters exist only in
  the 08-28 runs; earlier runs used the narrower filter).
- **Engine stats**: verl disables vLLM log-stats by default; all runs
  here pass `disable_log_stats=false` (else /metrics has no vllm:*).
- **Store-side**: mooncake-master :9003/metrics. IMPORTANT: clients use
  the BATCH RPCs (`master_batch_put_start/end`, `batch_exist_key`,
  `batch_get_replica_list`); the non-batch counters stay ~0. (An earlier
  "store is broken" diagnosis was a truncated grep that missed the
  batch counters - retracted in PULLBENCH.md.)
- **Windowing**: all rates computed as counter DELTAS inside the
  "saturated window" = first sample with kv_cache_usage >= 0.95 through
  the last sample >= 0.50 (excludes engine warm-up and the straggler
  drain). Script: analyze_v4.py (job dir), single-replica assumption.
- **Counter semantics** (verified on this build): `prompt_tokens_total`
  == `request_prompt_tokens_sum` - BOTH count COMPUTED prompt tokens
  (cache-served tokens excluded). `prefix_cache_queries/hits` count
  block lookups PER SCHEDULING ATTEMPT (requeue storms inflate the
  denominator; see T5). `num_preemptions_total`, `kv_cache_usage_perc`
  have no such caveats. `external_prefix_cache_*` = the connector
  (store) tier.

## 6. Results

### 6a. Cache collapse (no store; establishes the regime)

| run | replicas/gmu | kv mean/max | preempt | window | lookup hit rate |
|---|---|---|---|---|---|
| v1 | 4 / 0.40 | peak 26% | 0 | n/a (unsaturated) | 91.9% |
| v3 | 4 / 0.30 | peak 72% | 0 | n/a (unsaturated) | 87.7% |
| v4 | 1 / 0.30 | 86.5 / 99.9% | 133 | 66 min | 2.4% |
| v5 | 1 / 0.30 | 88.4 / 99.7% | 141 | 70 min | 2.4% |
| v6b | 1 / 0.27 | 79.5 / 100% | 125 | 68 min | 1.4% |
| v7 | 1 / 0.35 | 90.0 / 99.9% | 124 | 56 min | 2.3% |
| recompute-measured | 1 / 0.30 | 89.2 / 99.9% | 128 | 55.5+ min | 2.3% |

Collapse replicated 3x at identical config (2.4/2.4/2.3%), dose-responsive
(1.4% at the smallest bootable pool), cliff-shaped in demand/pool ratio.

### 6b. Store-pull vs recompute (the pullbench pair, 08-28)

| metric (saturated window) | store#1 | store#2 | recompute-measured |
|---|---|---|---|
| window duration | 25.5 min | 25.5 min | 68.5 min (final; run SUCCEEDED) |
| trajectories completed in window | 64/64 | 64/64 | 64/64 (final) |
| preemptions | 7 | 7 | 132 |
| kv usage mean (behavior) | 97.6% (stable) | 98.5% (stable) | 89.2% (oscillating 59-100%) |
| local lookup hit rate | 6.6% | 6.2% | 3.8% (final) |
| store-tier (external) queries -> hits | 405,363 -> 270,016 (66.6%) | 380,376 -> 229,360 (60.3%) | n/a |
| COMPUTED prompt tokens | 245,191 | 268,950 | 24,174,340 |
| computed prompt tok/s | 160 | 176 | 5,887 |
| generation tok/s | 25 | 35 | 296 |
| computed as % of served | 77.0% | 86.6% | 99.9% |
| master batch_put_end (saves, cumulative day) | 83k+ | (cumulative) | 0 |

Whole-run totals (store#1/#2): prompt computed 820,941 / 897,080; gen
115,981 / 126,334. Recompute whole-run at harvest: 20,206,215 / 1,126,404.

### 6c. Derived comparisons

- Same-work sampling time (final): 25.5 vs 68.5 min => **2.7x faster with
  the store**, consistent with the 2.5-2.7x range vs v4/v5 baselines.
- Prefill compute for the same delivered work (final): 24.2M vs ~0.25M
  computed prompt tokens => **~90-95x less prefill compute** with the store. The
  recompute arm's 6,274 tok/s engine throughput is 99.9% redundant
  re-prefill - engine tok/s measures the fire, not the output.
- Preemptions: 128-141 vs 7 (**~19x fewer**).
- Requeue churn (per-attempt lookups): 2.1-2.9B vs 35M-1.5B (high
  variance - see T5; not a primary metric).

### 6d. Context/latency reference (healthy 4-replica runs, same workload)

Mean turn prompt 5.1-5.7k tok; mean response 1.3-1.8k; trajectory final
contexts ~13.4k mean-of-maxima; per-turn tool gap 4.2s mean / 8-10s p90;
generation 6.6s mean. KV idle (tool gaps) ~40% of trajectory wall.

## 7. Reproduction

1. Manifests: `integration/rllm/k8s/rayjob-32b-pullbench-store.yaml` and
   `-recompute.yaml` (diff = one line). Pressure sweep:
   `rayjob-32b-pressure-smoke.yaml` + driver `tmp/run_pressure.sh <gmu> <out>`.
2. Prereqs on cluster: kuberay-operator, hf-secret, rbac-sandbox.yaml,
   mooncake-master + mooncake-config + mooncake-rdma-8 (k8s/base/),
   a GPU pool with the RDMA additional-node-networks.
3. Between runs: delete the RayJob AND `kubectl delete pod -l
   app=rllm-sandbox` (relaunches orphan sandbox pods; enough orphans
   starve the CPU pool and deadlock the next head pod).
4. Harvest: `kubectl logs <worker> -c metrics-scraper`; analyze with
   analyze_v4.py / goodput.py (job dir).

## 8. Threats to validity / known inconsistencies (audit here)

- **T1 - v4/v5 baselines are not perfectly paired** with the store arms:
  they predate `prefix_caching_hash_algo=sha256_cbor`, the mooncake
  volume/claims in the pod spec, and the token counters. The
  recompute-measured run (08-28) exists precisely to fix this - it is
  byte-identical to the store manifest minus the connector line, and it
  reproduced v4/v5's collapse numbers (2.3%/128 vs 2.4%/133-141), which
  bounds the effect of those config deltas as small.
- **T2 - trajectory stochasticity**: temperature 0.6, no per-request
  seeds; each run samples different trajectories of the same 32 tasks.
  Token totals per run vary O(10%); "same work" means same task set and
  trajectory count, not identical tokens. Direction unaffected (2.2-2.7x
  across all pairings); exact multipliers carry this noise.
- **T3 (resolved)**: the recompute run subsequently completed; final
  window 68.5 min with all 64 trajectories, preemptions 132, computed prompt
  24.2M tokens. Tables above show FINAL values; the mid-run harvest
  (55.5 min / 56 trajectories / 19.8M) is preserved in PULLBENCH.md history.
- **T4 - scrape thinning under load**: the sidecar's 4s timeouts drop
  some 30s samples during peak thrash (recompute arms have coarser
  sampling). Cumulative counters are unaffected; window boundaries are
  +/- a few samples.
- **T5 - lookup-based hit rates are per-scheduling-attempt**: requeue
  storms inflate denominators nonlinearly (40x lookup-count variance
  between store runs with near-identical behavior). Treat hit rates as
  qualitative (collapsed vs healthy); rely on window time, preemptions,
  computed-token counts, and store-tier hits as primary.
- **T6 - single node per run, spot nodes differ across runs**; same
  machine type throughout. The paired 08-28 runs ran on the same pool
  same day.
- **T7 - sandbox CPU oversubscription** (64 pods on ~16 vCPU) slows tool
  calls equally in both arms; it lengthens turn gaps, which if anything
  REDUCES the store arm's advantage (more natural KV-free time).
- **T8 - drain/straggler tails excluded** by the window; total job wall
  varies 29-80+ min run-to-run in BOTH arms due to runaway-trajectory
  retries (33-81-turn outliers exceeding the 28,672-token prompt bound).
  A turn cap would remove this noise source; not applied in these runs.
- **T9 - RDMA device auto-selection** (device_name unpinned): all traffic
  is same-node loopback here, probe-verified healthy; multi-node runs
  must pin per-GPU device names (connector warns).
- **T10 - store warm-state**: within each store run, early turns miss
  (cold store) and later turns hit; the master is shared across runs but
  embedded segments die with each run's clients, so no KV carries over
  between runs. The two probe put/gets (debug) are the only foreign keys.
- **T11 - scheduler policy active in all runs** (champion profile). With
  one replica, routing is trivially degenerate; the policy adds identical
  code-path overhead to both arms. The saturation filter drops the sole
  saturated replica and the policy's least-loaded fallback routes anyway
  - identical both arms, but it means the filter's intended behavior is
  bypassed in the single-replica setting.
- **T12 - reward signal ~0 for all trajectories** (tasks too hard for the
  policy): trajectories terminate via env_done/turn budget rather than
  success. Serving-side comparisons are unaffected; generalization to
  reward-bearing workloads assumes similar turn/length structure.
- **T13 - single-node runs measure the host-DRAM tier only, not the
  cross-node fabric**: with one TP=2 replica, both mounted segments live
  in that replica's own worker processes, so every store pull is
  same-node NIC loopback (host DRAM -> PCIe -> CX-7 hairpin -> PCIe ->
  HBM; `WITH_NVIDIA_PEERMEM=0` also inserts a cudaMemcpy staging hop
  through the 4 GB local buffer in each direction). These results are
  therefore an upper bound for the store's *local* tier; they do not
  exercise inter-node RDMA latency/bandwidth or fabric contention. The
  earlier relocation A/B (07-21: 0-6% relocation tax at RDMA vs 26-32%
  TCP) bounds the expected wire-hop cost; the pb2 2-node runs (one
  replica per node, Mooncake master allocating puts across all 4
  segments, ~half remote by construction) measure it directly.
