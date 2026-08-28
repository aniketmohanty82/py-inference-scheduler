# KV-pressure benchmark series: full methodology, hardware, workload, results

Written 2026-08-28 for audit. Covers the cache-collapse sweep (v1-v7) and
the store-pull vs recompute benchmark (pullbench). Every known threat to
validity is listed in section 8 - judge them there rather than discovering
them. All raw data referenced is committed alongside this file.

## 1. Hardware

**GPU nodes** (one per run; runs are single-node):
- GKE `rls-ab-west`, zone us-west1-c, node pools `rdma-gpu-pool` (spot) /
  `rdma-gpu-flex` (DWS flex-start), machine `a3-ultragpu-8g`:
  8x NVIDIA H200 (141 GB HBM3e each), 1872 GB RAM, hyperdisk-balanced
  500 GB boot, gVNIC primary NIC + 8x CX-7 RDMA NICs attached as
  additional node networks (`gvnic-sub-west`, `rdma-sub-west-0..7`),
  DRANET (`cloud.google.com/gke-networking-dra-driver=true`) exposing
  `mrdma.google.com` devices via ResourceClaimTemplate `mooncake-rdma-8`.
- Nodes are SPOT: identity varies per run (same machine type); one
  mid-run preemption occurred in the series (mooncake smoke era, not in
  any run reported below).

**CPU nodes** (task sandboxes + control plane): 2x n2-standard-8 (8 vCPU /
32 GB each), also hosting kuberay-operator, mooncake-master, the Ray head
pod (3 CPU/16Gi), and the RayJob submitter.

**Worker pod shape** (GPU node): whole-node pod - 100 CPU / 800Gi /
8 GPU requests+limits, IPC_LOCK, /dev/shm 128Gi tmpfs, plus (store arm and
recompute-measured arm only) /etc/mooncake ConfigMap mount and the rdma-8
resource claim.

**Mooncake store**: embedded mode - segments are mounted BY the vLLM
worker processes themselves (32 GB per client, `local_buffer_size` 4 GB),
coordinated by the `mooncake-master` Deployment (grpc :50051, metrics
:9003) on the CPU pool. protocol=rdma, metadata P2PHANDSHAKE, device_name
auto-selected (unpinned - see threat T9). Single-node runs => all
store traffic is same-node RDMA loopback through the HCA (`MC_FORCE_HCA=1`).

## 2. Software stack (image `rllm-verl-mooncake:dev`)

Base `ray-llm-mooncake:2.56.0` (Ray 2.56.0, Python 3.12, torch 2.11+cu130,
CUDA 13.0) plus: vLLM 0.22.1 (pip), verl 0.8.0, rllm @ 1d1109a6 with our
vendored patch (gateway routing_policy plumbing, kubernetes sandbox
backend, task_from_row task_path fix, loopback-pin skip), flash-attn 2.8.3
(source-built, sm90-only), stock mooncake wheels, py-inference-scheduler
(this repo, branch rllm-convergence). Dependency resolution requires
rllm's uv overrides (image/overrides.txt: numpy>=1.26 etc.).

Model: Qwen/Qwen3-32B (bf16, GQA: 64 layers, 8 KV heads, head_dim 128 =>
~256 KB KV per token full-model; TP=2 splits this per GPU). LoRA r32/a32
on the trainer side.

## 3. Workload

DeepSWE-style RL sampling step, driven by rllm's AgentTrainer (verl
backend, fully-async mode, `raise_on_error=false`, ONE training batch):

- **Tasks**: `r2egym_smoke` = first 64 rows of R2E-Gym/R2E-Gym-Subset
  (HF), materialized at image build (deterministic first-N; heavily
  orange3-repo). Runs use `data.train_batch_size=32` tasks x GRPO
  `rollout.n=2` samples = **64 rollouts (trajectories) per run**.
- **Agent**: mini-swe-agent, installed at rollout time INSIDE each task
  container. Task containers = k8s pods on the CPU pool (our vendored
  kubernetes sandbox backend; requests overridden to 50m/256Mi so all 64
  fit - CPU is oversubscribed, tool latency absorbs it).
- **Turn structure**: each turn = one OpenAI /chat/completions request
  carrying the FULL conversation history (system + instruction + all
  prior responses + tool outputs). Measured turn structure (from the
  healthy-runs logs): ~6.6-7.1 turns/trajectory mean, runaways to 33-81
  turns; per-turn prompt grows ~1k -> 11-16k tokens (max served >=16,384;
  max attempted >=28,673, rejected); mean per-turn tool-call gap 4.2 s,
  mean generation 6.6 s (pooled 1,669 turns).
- **Request path**: agent (in task pod) -> rllm Model Gateway (sqlite
  trace store; injects logprobs+return_token_ids) -> our
  SchedulerRoutingPolicy (champion profile: saturation filter kv 0.95/
  waiting 16, prefix_cache 4.0, waiting_queue 1.0, kv_cache 0.5,
  sticky_session 4.0) -> vLLM replica(s). With ONE replica the filter's
  drop is overridden by the policy's least-loaded fallback (identical in
  both arms).
- **Sampling params**: temperature 0.6, max_tokens 4096/turn,
  max_model_len 32768, data.max_prompt_length 16384 (training-side clip).
  NOT seeded per-request: trajectories are stochastic run-to-run (T2).
- **Reward**: task test-suite execution in-sandbox (rewards were ~0
  throughout - irrelevant to serving-side measurements, identical across
  arms).
- **Engines**: verl-launched `vllm serve` (vLLM's own OpenAI app),
  separated topology - trainer FSDP on 4 GPUs, rollout replica(s) on the
  rest. Pressure/pullbench runs: ONE replica at TP=2 (2 GPUs), 2 GPUs
  idle. `gpu_memory_utilization` (gmu) is the pool-size knob.

Pool arithmetic at gmu 0.30 / TP=2: 0.30x141 GB budget/GPU minus ~31 GB
weights/GPU leaves ~5-6 GiB KV/GPU (vLLM reported 2.25 GiB at gmu 0.25,
below the 4 GiB single-32k-request floor -> won't boot). Aggregate demand:
64 trajectories x up to 16k-token histories >> pool.

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
| window duration | 25.5 min | 25.5 min | 55.5 min (lower bound; 56/64 done at harvest) |
| rollouts completed in window | 64/64 | 64/64 | 56/64 |
| preemptions | 7 | 7 | 128 |
| kv usage mean (behavior) | 97.6% (stable) | 98.5% (stable) | 89.2% (oscillating 59-100%) |
| local lookup hit rate | 6.6% | 6.2% | 2.3% |
| store-tier (external) queries -> hits | 405,363 -> 270,016 (66.6%) | 380,376 -> 229,360 (60.3%) | n/a |
| COMPUTED prompt tokens | 245,191 | 268,950 | 19,808,935 |
| computed prompt tok/s | 160 | 176 | 5,956 |
| generation tok/s | 25 | 35 | 318 |
| computed as % of served | 77.0% | 86.6% | 99.9% |
| master batch_put_end (saves, cumulative day) | 83k+ | (cumulative) | 0 |

Whole-run totals (store#1/#2): prompt computed 820,941 / 897,080; gen
115,981 / 126,334. Recompute whole-run at harvest: 20,206,215 / 1,126,404.

### 6c. Derived comparisons

- Same-work sampling time: 25.5 vs >=55.5 min => **>=2.2x faster with the
  store** (2.5-2.7x against v4/v5 baselines). All 64 rollouts finished in
  the store windows; recompute had 8 unfinished at harvest.
- Prefill compute for the same delivered work: 19.8M vs ~0.25M computed
  prompt tokens => **~75-80x less prefill compute** with the store. The
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
  rollout count, not identical tokens. Direction unaffected (2.2-2.7x
  across all pairings); exact multipliers carry this noise.
- **T3 - recompute-measured window is a LOWER bound**: harvested at
  55.5 min with 8/64 rollouts still unfinished (straggler retry loops).
  Every additional minute makes recompute look worse, so the reported
  ratios are conservative.
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
- **T12 - reward signal ~0 for all rollouts** (tasks too hard for the
  policy): trajectories terminate via env_done/turn budget rather than
  success. Serving-side comparisons are unaffected; generalization to
  reward-bearing workloads assumes similar turn/length structure.
