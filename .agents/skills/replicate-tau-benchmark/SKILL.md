---
name: replicate-tau-benchmark
description: Replicate the tau-bench router benchmark on a Kubernetes 8-GPU pod — set up the tau harness on a slime pod, run the three arms (sgl-router cache_aware, round_robin, py-inference-scheduler composed+filter), optionally enable managed-Prometheus scraping, collect and compare per-step results. Use when asked to replicate, re-run, set up, or extend the tau benchmark.
---

# Replicate the tau-bench router benchmark

Compares three routing policies on a real RL rollout workload (tau-bench
retail GRPO via slime, Qwen3-14B, 8 single-GPU sglang engines): sgl-router
`cache_aware` (default), sgl-router `round_robin`, and py-inference-scheduler
with the composed+saturation-filter profile.

Everything is driven by `.agents/skills/replicate-tau-benchmark/driver.sh` —
modular phases, each idempotent and self-verifying. Run them in order; when a
phase fails, fix per Troubleshooting and re-run **that phase only**.
All paths are relative to the repo root.

## Prerequisites (user-provided; ask, never assume)

| need | detail |
|---|---|
| Kubernetes cluster | node pool with 8× H100-class GPUs (spot works — see Gotchas). GKE assumed for the optional metrics phase; see Portability |
| kubectl context | `export RLS_CONTEXT=<context>` |
| GCP project id | `export RLS_PROJECT=<project>` (optional metrics phase only) |
| HF token secret | `kubectl create secret generic hf-secret --from-literal=hf_api_token=hf_...` (model download) |
| Gemini API key | `export GEMINI_KEY_FILE=<path>` or pre-place at `<pod>:/root/gemini_key` (user simulator; billed per rollout turn) |
| this repo | with `tau-benchmark-v1` tag fetchable (the router ref that is deployed; override with REPO_REF) |

## Phase map

```bash
export RLS_CONTEXT=... RLS_PROJECT=... GEMINI_KEY_FILE=...
D=.agents/skills/replicate-tau-benchmark/driver.sh
$D preflight        # local tools, context, hf-secret
$D pod-up           # pod (configs/benchmarks/tau-slime-pod.yaml) + PodMonitoring; waits Ready
$D env-verify       # 8 GPUs, slime/sglang/sglang-router importable
$D model-prep       # HF download Qwen3-14B + Megatron torch_dist conversion (detached)
$D model-verify     # poll until model-prep done
$D tau-setup        # tau-bench fork @ pin, slime patches, task jsonls, Gemini key
$D router-deploy    # git-archive $REPO_REF -> pod /root/pis; boot check
$D run-vanilla      # sgl-router cache_aware            (RUN_STEPS, default 3)
$D run-rr           # sgl-router round_robin
$D run-ours         # py-inference-scheduler composed+filter profile
$D status           # progress of the active arm
$D metrics-verify   # managed Prometheus returning engine samples (optional, GCP)
$D collect          # pull logs local + per-step rollout_time table
```

Performance claims need `RUN_STEPS=20` with arms interleaved (V,C,V,C then RR)
— never compare across days: ambient cluster load drifts several percent.
Short runs (default 3 steps) only validate the setup.

## The workload (what a correct setup produces)

Batch 32 tasks × 8 samples = 256 episodes/step, gbs 256. Episodes are
multi-turn tool-calling conversations against a Gemini user sim: roughly ten
requests per episode, a ~4.4k-token prefix shared by every episode (system
prompt + policy doc + tools), final contexts around 8k tokens, engine context
cap 40,960. Each engine's KV pool (`mem-fraction-static 0.6`) holds ~128k
tokens and peaks near full under this batch — the pressure regime is the
point: without it, all routers perform identically.

## Components

- **Metrics scraping (optional, GCP)**: `pod-up` applies
  `configs/benchmarks/tau-slime-podmonitoring.yaml` (scrapes engine ports
  15000..15014). Query with **underscore** metric names
  (`sglang_generation_tokens_total`); the colon forms the engines expose
  (`sglang:...`) return empty from the managed endpoint. Benchmark results do
  NOT depend on this: step times come from slime logs, cache-hit % from the
  pod-side sampler.
- **Pod spec**: `configs/benchmarks/tau-slime-pod.yaml` — slimerl/slime:v0.3.0
  image, 8 GPUs, /dev/shm memory volume, hf-secret env.
- **tau harness edits**: `patches/slime-v0.3.0-patches.diff` — adds the slime
  custom-generate glue (`generate_with_tau.py` config, `trainable_agents.py`:
  async rollout loop, qwen25 tool parsing, litellm retries, per-episode
  `x-rls-session-id` header) plus a slime loss.py fix. tau-bench itself =
  JD-ETH fork `feature/litellm-retry` @ the commit pinned in driver.sh. Task
  jsonls are generated deterministically from the fork's TASKS_TRAIN/TASKS_DEV.
- **Sticky-session requirement in py-inference-scheduler**: the slime router
  must copy request headers onto `LLMRequest` so the `sticky_session` scorer
  can read `x-rls-session-id` (a ~5-line change in
  `integration/slime/server.py`, already present in the pinned `tau-benchmark-v1` tag;
  cherry-pick if deploying another ref).
- **Router profile**: `prof_champion.yaml` (prefix 4.0 with universal-block
  discount / sticky 4.0 / waiting-queue 1.0 / kv 0.5 + saturation filter
  kv 0.95, waiting 16 + max-score picker).

## Analysis rules (cause and effect; ignore at your peril)

- **Step times are resample-luck dominated.** GRPO drops zero-variance groups
  and regenerates replacements mid-step; the drop count varies by task luck
  and drives step time far more than routing does. Always log the per-step
  drop count as a covariate; never compare single steps.
- **The distribution is two-regime**: a dense normal band plus drop-wave
  "blowup" steps (roughly a quarter of steps, ~1.5x band time). Router
  differences live almost entirely in how cheaply the blowups are absorbed —
  compare means AND medians, and if arms differ in blowup count, regress step
  time on drop count and compare slopes.
- **Small effects need many steps**: tail-driven differences of a few percent
  are inside luck range at 40 steps/arm. Pre-register a decision rule (e.g.
  ">3% pooled mean with matched drops") before running.
- Only in-queue (same day, alternating arms) comparisons are valid.

## Gotchas (mechanisms, all encountered while building this)

- **pkill self-match**: a shell command *containing* the literal name it kills
  can kill itself (exit 143). Bracket-guard patterns:
  `pgrep -f 'bash /root/arm_runner[.]sh'`.
- **Zombie router**: the router retitles its process to `scheduler`; sglang
  also runs processes named `scheduler`. Never `pkill scheduler` alone — the
  driver's `kill-router` finds the port-8000 holder via `/proc/net/tcp` inode.
- **kubectl backgrounding footgun**: `cmd1 && cmd2 &` backgrounds the WHOLE
  chain, and background jobs read stdin from /dev/null — a piped `cat > file`
  lands 0 bytes while everything downstream "succeeds". Upload, verify size,
  then launch, as separate execs.
- **Detach or die**: anything long-running on the pod must be
  `nohup setsid ... < /dev/null &` — kubectl exec sessions die and SIGHUP
  whatever they own.
- **Never launch a daemon as a kubectl exec foreground command**: an exec
  whose command backgrounds a daemon (even with all fds redirected) can hang
  the exec stream indefinitely. Working pattern: upload a pod-side script,
  launch IT detached with a trailing `sleep 1; echo LAUNCHED`, poll a result
  file from a separate exec (see boot_check.sh).
- **`--save-interval` must exceed `--num-rollout`** or that arm pays a
  checkpoint save the others don't.
- **`ray job submit` can exit while the job still runs** (log-stream drop; it
  polls "Status: RUNNING" then bails). A runner that treats submit-exit as
  arm-done lets the next arm's `pkill -9` sweep kill a live step. arm_runner
  waits on `ray job list` and recovers logs via `ray job logs`.
- **Context-overflow crash**: episodes that exceed the engine context cap
  cause engine 400s, the harness's retry ladder exhausts, and a RayTaskError
  kills the run (expect it roughly once per ~60 steps). Runners are resumable
  (`.done` markers): salvage completed steps, relaunch.
- **Gemini concurrency**: 429s from the user sim → set
  `--sglang-server-concurrency 32` in SGLANG_ARGS (comment in template).
- **Spot preemption**: every phase re-checks pod Running; after preemption
  re-run `pod-up` then later phases — pod-local state is gone, so expect
  model-prep to re-run.
- **Engines only expose /metrics while a run is active** — an idle-time scrape
  check returning empty is not a monitoring failure.
- **aiohttp defaults**: the router's proxy session needs
  `TCPConnector(limit=0)` and `timeout total=None` (present in the pinned tag) —
  defaults cap at 100 connections / 300s and silently serialize or kill long
  generations.

## Troubleshooting (symptom -> fix)

| symptom | fix |
|---|---|
| pod Pending >15min | `kubectl describe pod` events. "Insufficient nvidia.com/gpu" + "didn't trigger scale-up" = fixed-size pool with its node occupied: free the node or resize the pool. Pure spot shortage: another zone/pool |
| `nvidia-smi failed` in env-verify | GPU driver mount wrong for your platform — see Portability |
| model-prep log stalls at download | HF token lacks model access or rate-limited; check `/root/model_prep.log` |
| `router boot check failed` | read `/root/router_boot_check.log`; port 8000 held → `driver.sh kill-router` |
| workers not `[]` at arm start | stale registry from a previous router — `kill-router`, re-run arm |
| RayTaskError mid-run | context overflow (see Gotchas): `driver.sh status` shows steps done; salvage + relaunch |
| managed-Prometheus query empty during a run | colon metric names used; switch to underscores |
| query 401 "restricted due to a domain admin's policies" | some corp policies reject `gcloud auth print-access-token`; use `gcloud auth application-default print-access-token` (driver does) |
| Prometheus rate() = 0 mid-run | engines are in the training phase between rollouts — decode rate is genuinely 0; the series existing proves the pipeline |
| step times wildly bimodal | normal (drop waves); analyze per rules above, don't debug the cluster |

## What a correct replication looks like

- All three arms complete their steps with step times in a common band, plus
  occasional blowup steps well above it (both regimes appearing IS the
  workload signature).
- Cache-hit separation: both custody arms (cache_aware, ours) land ~90%+;
  round_robin collapses ~20 points below them. **If vanilla and RR hit rates
  don't separate, the workload is not reproducing** (check the shared prefix,
  batch size, n-samples-per-prompt).
- The ours arm's router log shows workers self-registering, per-decision
  scorer output including sticky_session, and saturation-filter drops
  appearing only under load bursts.
- Short runs cannot rank cache_aware vs ours — that difference is tail-driven
  and needs 20-step interleaved runs (see Analysis rules).

## Portability

Only one phase is GCP-bound: `metrics-verify` (managed-Prometheus PromQL
endpoint, gcloud auth) and the `PodMonitoring` CRD it depends on. The pod
spec's GPU wiring (`hostPath` nvidia mount + `LD_LIBRARY_PATH`) is a GKE/COS
convention — on other platforms use your NVIDIA device-plugin defaults and
drop that mount. Everything that produces results (arms, runners, samplers,
collect) is plain kubectl + bash on any Kubernetes; swap PodMonitoring for
your Prometheus's scrape config or skip metrics entirely.

## Costs to warn the user about

8-GPU node-hours (a 20-step arm is several hours wall-clock), Gemini
user-sim calls (order 100k per 20-step arm), and a ~30GB model download.
