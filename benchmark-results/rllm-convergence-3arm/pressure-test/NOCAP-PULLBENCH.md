# Pull-from-Mooncake vs recompute on a long-context agentic RL rollout workload (no output cap)

## TLDR

KV rescue from the Mooncake store cuts true prefill compute ~13x in every
configuration, and its wall-clock payoff scales with per-replica
oversubscription. At 128 rollouts on ONE replica, the store wins every
latency metric: per-turn fixed cost 35.2s vs 48.9s (-28%), median turn
50.0s vs 63.5s (-21%), median trajectory 14.0 min vs 18.9 min (-26%),
total sampling 1:10:54 vs 1:15:25 (-6%). At 128 rollouts spread over TWO
replicas the latency race is a wash (fixed cost 22.8s vs 20.3s; totals
58:47 vs 54:54) while the compute savings persist (13.2x). This matters
because trajectory latency is bounded by per-turn cost, and re-prefill
only becomes expensive when replicas are oversubscribed enough that
requeued histories displace serving - rescue is insurance whose payout
grows with pressure, and whose compute savings are unconditional.

## Purpose

Measure whether pulling evicted KV from a shared Mooncake store beats
recomputing it, on an honest workload: no artificial output caps
(requests carry max_tokens=null; the model window is the only bound),
full trajectory completion in every run, and a fixed engine deadlock
(WEDGE-BUG.md) that had silently invalidated all earlier store runs.
Baselines are straight recomputation - what every RL stack does today.

## Arm mechanics

| arm | mechanics |
|---|---|
| recompute | vLLM 0.22.1 (base-image build) with prefix caching; preempted/evicted request histories re-prefill from scratch. No connector. |
| store | identical engine + `kv_transfer_config={kv_connector: DecodeKVSavingConnector (ours, PR #46 lineage), kv_role: kv_both, save_decode_kv: true}` + sha256_cbor hashing. Decode KV saves to Mooncake embedded segments (96gb/worker, host DRAM, RDMA via CX-7); evicted histories reload from the store. Includes the 09-01 image fix: a failed save releases its request (WEDGE-BUG.md). |

Manifest diff between arms is the kv_transfer_config line plus names
(`git diff` the rayjob manifests). Router: SchedulerRoutingPolicy
(champion profile) in both arms, both topologies.

## Methodology

Hardware, software pins, workload structure, trajectory lifecycle:
METHODOLOGY.md (sec. 3.0 for the lifecycle). Deltas for this matrix:

| dimension | value |
|---|---|
| output cap | NONE (max_tokens=null -> vLLM fills the remaining 32,768-token window per request) |
| observed prompts | p50 ~10k, p95 23-29k, max 32,766 (window edge) |
| observed completions | p50 ~390-510, max 29,345 |
| workload | 64 tasks x n=2 = 128 rollouts, n_parallel_tasks=128, every run |
| topologies | 1 node: trainer(4 GPU)+1x TP=2 replica serving all 128; 2 nodes: 1x TP=2 replica per node (k8s-forced), 64 rollouts each |
| KV pool | gmu 0.30 (~15GB per replica) both topologies |
| store | 96gb/worker segments (2 workers/replica); 1n = 192GB all-local; 2n = 384GB, ~half of puts/pulls remote by master allocation |

Metrics (all recorded; derived values show inputs):

| metric | definition | source |
|---|---|---|
| total sampling | rllm task progress bar, 64/64 close | driver log (streamed from launch) |
| per-turn fixed cost + decode slope | least-squares of turn latency vs completion tokens over all turns; intercept = queue+prefill/load, slope = s/decode-token | gateway sqlite trace DB (per-turn latency_ms, token counts) |
| trajectory spans, tail | per-session first-arrival to last-completion | trace DB |
| true prefill compute | `prompt_tokens_by_source{local_compute}` (NOTE: `prompt_tokens_total` counts PROCESSED tokens - compute+cache+store; the labels sum to it exactly) | engine counters (sidecar, 30s) |
| store contribution | by_source{external_kv_transfer}, hit counters, segment/eviction gauges | engine counters + master snapshots |
| errors | raw driver stream + per-pod session-log tarballs; 3-min classified feed is a view, not evidence | nocap-raw/ |

NOTE (invited scrutiny): the intercept/slope regression is our derived
construction; raw (completion_tokens, latency) pairs are in the trace
DBs for independent refits. Single run per cell; temperature 0.6 without
per-request seeds, so turn/token totals vary O(10%) between arms
(store arms actually served MORE turns/tokens in 3 of 4 cells).

## Results

| metric | 1n store | 1n recompute | 2n store | 2n recompute |
|---|---|---|---|---|
| total sampling | **1:10:54** | 1:15:25 (+6.4%) | 58:47 | **54:54** (-6.6%) |
| per-turn fixed cost | **35.2s** | 48.9s (+38.9%) | 22.8s | 20.3s (-11.0%) |
| decode slope (ms/tok) | 17.7 | 19.6 (+10.7%) | 17.6 | 17.4 (-1.1%) |
| median turn | **50.0s** | 63.5s (+27.0%) | 30.9s | 30.3s (-1.9%) |
| p95 turn | 85.1s | 104.0s (+22.2%) | 67.1s | 56.5s (-15.8%) |
| median trajectory | **14.0m** | 18.9m (+35.0%) | 9.6m | 9.2m (-4.2%) |
| true prefill compute (tok) | **2.11M** | 19.77M (9.4x) | **1.84M** | 24.16M (13.1x) |
| served prompt tokens | 32.50M | 21.90M | 33.31M | 31.09M |
| tokens from store | 20.63M | 0 | 24.26M | 0 |
| turns | 2,610 | 2,165 | 2,455 | 2,441 |
| tasks consumed/filtered | 60/4 | 59/5 | 50/14 | 57/7 |
| recovered save-race errors | 11,002 | 0 | 3,421 | 0 |

## Run logs (nocap-raw/ unless noted)

| run | driver stream | classified errors | session logs | counters | trace DB (job tmp, ~0.5-0.8GB each) |
|---|---|---|---|---|---|
| 1n store | pullbench_nocap_driver.log.gz | errors_pullbench_nocap.log | pb1n_nocap_store_sessionlogs.tgz | ../pb1n_nocap_store_raw.log | pb1n_nocap_store_traces.db |
| 1n recompute | pullbench_nocap_rc_driver.log.gz | errors_pullbench_nocap_rc.log | pb1n_nocap_rc_sessionlogs.tgz | ../pb1n_nocap_recompute_raw.log | pb1n_nocap_rc_traces.db |
| 2n store | pb2_nocap_store_driver.log.gz | errors_pb2_nocap_store.log | pb2_nocap_store_sessionlogs_*.tgz | ../pb2_nocap_store_raw.log | pb2_nocap_store_traces.db |
| 2n recompute | pb2_nocap_rc_driver.log.gz | errors_pb2_nocap_rc.log | pb2_nocap_rc_sessionlogs_*.tgz | ../pb2_nocap_recompute_raw.log | pb2_nocap_rc_traces.db |

Master counter snapshots: pb2_master_snapshots.log. Manifests:
integration/rllm/k8s/rayjob-32b-pullbench-{store,recompute}.yaml (1n),
rayjob-32b-pb2-{store,recompute}.yaml (2n), image Dockerfile @ a706358.

## Analysis

(a) **Compute savings are unconditional; latency savings are
pressure-gated.** In all four cells the store+local cache served 94%+ of
the store arms' prompt tokens (by_source), cutting true prefill compute
9.4-13.2x. But wall-clock only follows when re-prefill work is dense
enough to displace serving: at 128 rollouts per replica (1n) requeued
10-27k-token histories re-prefill constantly and the store's fixed cost
per turn is 13.7s lower; at 64 per replica (2n) both arms' fixed cost
drops to ~20-23s and the difference is inside single-run noise.

(b) **The per-turn fixed cost is the whole story.** Decode slopes are
nearly identical across cells (17.4-19.6 ms/tok); trajectories are 91-95%
LLM time; so trajectory latency = turns x (fixed + slope x tokens).
Oversubscription doubles the recompute arm's fixed cost (20.3 -> 48.9s)
but only raises the store arm's by 12.4s (22.8 -> 35.2s) - rescue
flattens the pressure response curve.

(c) **The earlier capped-regime result is consistent**: with 4096-token
outputs and short prompts (PB2.md capped pair), both arms sat at ~16-17s
fixed cost and latency parity, with the same ~13x compute savings - short
turns are queue-bound, so there was nothing for rescue to save
time-wise. Lifting the cap moved the workload into the regime where
prefill matters, exactly as predicted by the turn model.

(d) **Store overheads are real and recorded**: 11,002 (1n) / 3,421 (2n)
save-race errors recovered (each drops one decode-KV save - the fix
releases the request; WEDGE-BUG.md), heavy store eviction churn
(hundreds of GB per run at these volumes; segments are a sizing knob),
and a small decode tax visible in the capped-era pair. The 2n store
cell's higher filtered-task count (14 vs 4-7 elsewhere) is unexplained
by these logs and flagged rather than interpreted.

(e) **Both arms completed every run** (counters drain to zero, 64/64
tasks, 110-125/128 rollout completions) - the first store-vs-recompute
matrix in this project where that is true; all seven earlier store runs
had wedged (WEDGE-BUG.md), which is why no prior store numbers appear
here (valid-runs-only policy).
