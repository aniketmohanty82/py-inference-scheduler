# verl-native SWE A/B - upstream-scale pair (32B + LoRA, 256-wide)

Store-vs-recompute on verl 0.8.0's native SWE agent loop, 2026-09-10.
Qwen2.5-32B-Instruct + LoRA r32/a32 (fsdp2, dynamic-bsz 16384), **4 GRPO
steps per arm, batch 64 x n 4 = 256 trajectories per rollout** (upstream
PR-62 scale), seed 42, single 8xH200 node, tp=2 (4 engines), **gmu 0.45**,
`free_cache_engine=False` both arms, 32 assistant turns,
`SWE_OBS_MAX_CHARS=20000`, gVisor pool pinned at 21 nodes, dataset baked in
image `swe8` (256 rows). 4 x 64 consumes each task **exactly once**, so no
cross-epoch task repeats and no cross-epoch staleness caveat.

Arms differ by exactly the kv_transfer_config flags; both verified on the
live Hydra command line before running (recompute: 0 store flags; store:
kv_transfer_config + sha256_cbor). Mooncake master verified empty at pair
start; the recompute arm writes nothing, so the store arm began on a clean
tier.

## Validity (all recorded, all pass)

| gate | recompute | store |
|---|---|---|
| steps completed / rc | 4/4, rc=0 | 4/4, rc=0 |
| wall-clock audit (sum step times vs wall) | 5,909s vs 6,201s (292s init) | 5,637s vs 5,909s (272s init) |
| actor/entropy range | 0.188-0.202 | 0.198-0.775 (see DRIFT below) |
| num_turns/mean range | 46.2-48.8 | 45.9-48.1 |
| rollout-error / OOM / EngineDeadError | 0 / 0 / 0 | 0 / 0 / 0 |
| by_source identity | EXACT | EXACT |
| PRESSURE: num_preemptions_total | 57 | 30 |
| PRESSURE: kv_cache_usage_perc max | 0.9952 | 0.9954 |
| connector: force-fails / errors / dropped signals | n/a | 0 / 0 / 0 |

## Serving split (final cumulative counters, 4 engines, identity EXACT)

| source | recompute | share | store | share |
|---|---|---|---|---|
| local_compute | 58,065,234 | 38.26% | **14,619,750** | **9.76%** |
| local_cache_hit | 93,707,616 | 61.74% | 119,627,440 | 79.86% |
| external_kv_transfer | 0 | 0% | 15,545,040 | 10.38% |
| TOTAL (= prompt_tokens_total) | 151,772,850 | | 149,792,230 | |

Workloads matched to 1.3% on total prompt tokens. **local_compute: 58.1M -> 14.6M
tokens = 3.97x reduction** - the headline result, and the one that does not
depend on timing noise.

Note the second-order effect: the store arm's LOCAL cache-hit share is also
higher (79.9% vs 61.7%) and its per-engine prefix-hit rates run 29-67% vs
21-27%. external_kv_transfer tokens are cached locally on arrival, so the external
tier raises local_cache_hit rather than competing with it. The store arm also
took **half the preemptions** (30 vs 57): recomputing less means holding
fewer blocks for less time.

Mooncake ops (store arm, zero failed keys): save_put 62,365 ops @ 12.9ms
mean, load_get 4,258 ops @ 100.0ms mean. Master evicted 36 times / 1.89TB
over the run (tier turns over at this scale) with no failures.

## Per-step results (4 steps; deltas store vs recompute)

| verl metric | recompute | store | delta | store lower |
|---|---|---|---|---|
| timing_s/agent_loop/generate_sequences/mean | 197.22s | 82.45s | **-58.2%** | 3/4 |
| timing_s/agent_loop/generate_sequences/max | 669.80s | 439.93s | -34.3% | 4/4 |
| timing_s/agent_loop/slowest/generate_sequences | 308.56s | 60.86s | **-80.3%** | 2/4 |
| timing_s/agent_loop/slowest/tool_calls | 862.91s | 1045.34s | +21.1% | 1/4 |
| timing_s/gen | 1177.37s | 1111.65s | -5.6% | 2/4 |
| timing_s/step | 1477.37s | 1409.26s | -4.6% | 2/4 |
| timing_s/agent_loop/tool_calls/mean | 131.75s | 138.58s | +5.2% | 2/4 |
| response_length/mean | 11,067 | 10,999 | -0.6% | 2/4 |
| num_turns/mean | 47.59 | 47.13 | -1.0% | 4/4 |
| critic/score/mean | 0.0508 | 0.0430 | -4.3% | 2/4 |
| perf/throughput (NOT interpretable - see below) | 251.4 | 262.5 | +4.4% | - |
| actor/perf/cpu_memory_used_gb | 157.6 | 1218.7 | +673% | - |
| actor/grad_norm | 0.0022 | 0.0088 | +301% | - |

The `slowest/*` pair is mechanically suggestive but **weakly supported**: on
the trajectory that gated the step, slowest/generate_sequences fell 80.3%
while that same trajectory's slowest/tool_calls rose 21.1%, consistent with
the gating trajectory ceasing to be generate_sequences-bound and becoming
tool_calls-bound. It is 2/4 and 1/4 on consistency, and `slowest/*` describes
a *different single trajectory* in each step and each arm, selected by
`np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)`
(agent_loop.py:1146) - so the two arms' rows are not even the same task. Read
it as an illustration of the mechanism, not as evidence for it. The evidence
is the by_source split and the occupancy table below.

NOTE (perf/throughput is reported but must NOT be read as a serving
result): verl computes it as `total_num_tokens / (timing_raw["step"] *
n_gpus)` (`compute_throughout_metrics`, verl/trainer/ppo/metric_utils.py -
note the upstream typo in the name). Both terms defeat it here. The
numerator is tokens IN THE BATCH, identical across arms by construction
(-0.6%), so the store's 3.97x reduction in tokens actually COMPUTED is
invisible to it. The denominator is the full step wall clock, which this
workload spends largely idle waiting on gVisor sandboxes - measured at
35-69% of each rollout in THIS pair (see the straggler dead-time table
below). The result is
~251 vs ~262 tok/s/GPU for a 32B model on H200s, which is a sandbox
measurement, not a serving one; excluding training only moves it to 315 vs
332. verl emits no sampling-only throughput (`perf/*` is just throughput,
time_per_step, total_num_tokens; `perf/mfu/actor_infer` is set from
`old_log_prob_mfu`, a training-side pass). The rate that does discriminate
is prompt tokens COMPUTED per sampling second: 1,541 vs 411 per GPU
(DERIVED) - the same 3.8x as the by_source split.

`actor/perf/cpu_memory_used_gb` records the store's host-memory cost: the
8 x 128GB mooncake segments. `actor/grad_norm` rising 301% is a
training-signal difference that moves with the entropy drift below; with
N=4 it is reported, not explained.

NOTE (read the -58.2% per step, not as a mean):

| step | rc generate_sequences/mean | store | ratio |
|---|---|---|---|
| 1 | 326.4s | 104.6s | 3.1x |
| 2 | 325.0s | 100.5s | 3.2x |
| 3 | 79.1s | 58.3s | 1.36x |
| 4 | 58.5s | 66.5s | 0.88x (store slower) |

## THIS PAIR IS BISTABLE: steps 1-2 and 3-4 are different regimes

The decay above is **not** task difficulty. The workload is provably
identical across all four steps, from the driver logs:

| per step | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| `prompt_length/mean` | 519.4 | 515.7 | 522.2 | 515.4 |
| `num_turns/mean` | 48.8 | 46.2 | 48.4 | 47.0 |
| `response_length/mean` | 11,088 | 11,144 | 10,606 | 11,428 |
| `perf/total_num_tokens` | 2,971,459 | 2,985,002 | 2,848,848 | 3,057,639 |
| `timing_s/gen` | 1,112 | 1,291 | 1,149 | 1,157 |

(`prompt_length/mean` is bit-identical between arms, confirming both saw the
same tasks in the same order.) What changed is the engine's operating point:

| recompute | `kv_cache_usage_perc` avg | `local_compute`/turn | `local_cache_hit`/turn | preempt | peak running |
|---|---|---|---|---|---|
| step 1 | 0.601 | 1,982 | 1,076 | 24 | 127 |
| step 2 | 0.566 | 2,119 | 1,049 | 29 | 137 |
| step 3 | 0.273 | 410 | 2,694 | 4 | 75 |
| step 4 | 0.200 | 265 | 2,867 | 0 | 53 |

Tokens presented per turn are flat at 3,029-3,169 in all eight steps of both
arms; only the SPLIT moves. A request here is one turn, and between turns a
trajectory holds zero allocated blocks - its KV sits in the free-but-cached
tier. So the loop closes on itself: at ~100% occupancy the cached tier is
squeezed to nothing, every returning turn re-prefills its whole context,
engine service lengthens, and occupancy stays pinned. Below ~60% the cached
tier survives, turns prefill only new tokens, and occupancy stays low. Both
states are self-sustaining; steps 1-2 sat in the first, steps 3-4 in the
second. **What tipped it between step 2 and step 3 is not determined by
anything recorded here** - the policy is ruled out (`actor/entropy` flat
0.188-0.202, `grad_norm` ~0.002 at lr 1e-6), leaving system state that
evolves within a job and resets when it restarts.

The store arm is FLAT at 0.313 / 0.314 / 0.253 / 0.283 - **it never enters
the collapsed state at all.** Keeping per-turn prefill small even on a local
miss (369 tokens/turn recomputed in step 1 vs recompute's 1,982, with 501
supplied by `external_kv_transfer`) is what stops the collapse. That is a
stronger claim than the latency delta and it is visible in one recorded
gauge.

Consequence for reading this pair: **only steps 1-2 tested the pressure
regime.** Steps 3-4 dilute every mean. Report by regime; the 4-step
arithmetic means understate the pressured case and overstate the typical one.

### Generation throughput (decode tokens per trajectory-second)

Normalizing decode tokens by the summed `generate_sequences` time removes
both the task-mix and the straggler-tail confounds:

| step | recompute | store | ratio |
|---|---|---|---|
| 1 | 6.83 | 25.87 | **3.8x** |
| 2 | 7.35 | 25.04 | **3.4x** |
| 3 | 29.89 | 37.51 | 1.3x |
| 4 | 40.21 | 35.29 | 0.9x |

Under real pressure the store yields 3.4-3.8x more generated tokens per
second of engine time; once pressure vanishes the arms converge.

### Why timing_s/gen barely moves: straggler dead time

`timing_s/gen` is set by ONE trajectory, so it hides all of the above:

| recompute | `timing_s/gen` | engine busy (`num_requests_running`>5) | idle | idle % |
|---|---|---|---|---|
| step 1 | 1,112s | 720s | 392s | 35% |
| step 2 | 1,291s | 720s | 571s | 44% |
| step 3 | 1,149s | 420s | 729s | 63% |
| step 4 | 1,157s | 360s | 797s | 69% |

35-69% of every rollout has the engine idle while the gating trajectory
burns `SWE_CMD_TIMEOUT_S=60` hangs (`slowest/tool_calls` 1,139s over 65 turns
= 17.5s per call). Any rate divided by `timing_s/gen` inherits this: computed
that way tool throughput looks flat at ~10.5 calls/s, while over the busy
window it is 17.3 -> 33.4 calls/s and rising.

## Cost per token (all inputs recorded; no FLOPs model)

Per turn = metric / `num_turns/mean`. Same tasks, same output length, so
the only difference is where the prompt tokens came from.

| per turn | recompute | store |
|---|---|---|
| `timing_s/agent_loop/generate_sequences/mean` | 4.14 s | 1.75 s |
| `response_length/mean` | 227-243 | 227-239 |
| `prompt_tokens_by_source{local_compute}` | 1,192 | 303 |
| `prompt_tokens_by_source{external_kv_transfer}` | 0 | 322 |

Divide the latency delta by the token delta:

| | value |
|---|---|
| avoided `local_compute` token | **2.7 ms** |
| `external_kv_transfer` token (`load_get` s / tokens) | **0.027 ms** |
| ratio | **~100x cheaper to pull** |

Consistent computed per trajectory or per turn (2.71 vs 2.69 ms/token).

NOTE: 2.7 ms is the MARGINAL SYSTEM cost, not hardware cost -
`generate_sequences` is wall time and includes queueing, so avoided prefill
also shortens the queue for everyone. Raw FLOPs for 32B at tp=2 are nearer
0.08 ms/token, i.e. ~30x queueing amplification. It is therefore the right
number for capacity planning and the wrong one for hardware sizing, and it
is REGIME-SPECIFIC: the low-pressure pair (gmu 0.30, batch 16 x n 4) showed
-5.9% on the same metric instead of -58.2%, so recompute it per regime.
It also credits the whole latency delta to avoided prefill while the arms
differ in preemptions too (57 vs 30).

## DRIFT: the store arm's entropy climbs; recompute's does not

| step | recompute entropy | store entropy | rc ppo_kl | store ppo_kl |
|---|---|---|---|---|
| 1 | 0.188 | 0.198 | -1.81e-05 | +3.53e-05 |
| 2 | 0.201 | 0.265 | -8.66e-06 | +2.96e-05 |
| 3 | 0.202 | **0.390** | -4.41e-05 | +3.60e-05 |
| 4 | 0.197 | **0.775** | -4.96e-05 | +1.50e-05 |

Recompute is flat across all four steps; the store arm rises monotonically
to ~3.9x its own step-1 value. `actor/ppo_kl` also flips sign consistently
(recompute negative, store positive) - tiny in magnitude (1e-5) but
systematic, and ppo_kl measures exactly the rollout-vs-training policy
divergence a stale-KV effect would produce.

Mechanism (documented in `../../integration/verl/swe/PINS.md` before this
run, now MEASURED): store keys contain no weight version. With LoRA the
merged rollout weights change every step, so KV written in steps 1-3 is
served back in step 4 under different weights. The step-1-self-baseline
tripwire agreed in the design phase is what this table implements.

Scope of the claim: 0.775 still clears the <1.0 validity gate and the
workload metrics (turns, response length, score) stay comparable, so the
pair is VALID. But N=4 with one pair cannot separate stale-KV from other
causes with certainty - what is recorded is (a) a flat recompute control on
identical tasks, (b) a monotonic store-only rise, (c) a consistent ppo_kl
sign flip, and (d) a mechanism that predicts exactly this. Extrapolating
the trend, entropy would breach the gate by roughly step 5-6. **A
longer-horizon store run needs per-step flush or weight-versioned keys
before it can be trusted for convergence work.** This is the first recorded
cost of the store in this program.

## Connector fix stack (this pair is the first run on it)

The prior regime could not complete a single step. Four ways the connector
dropped the block-release signal (and one sizing law) were found and fixed;
full fault tree in `../../integration/verl/CONNECTOR-INVESTIGATION.md`,
patch in `connector_v3.py` (19 unit tests run at image build time).

| symptom before | after (this pair) |
|---|---|
| silent wedges: engine pinned ~98% KV, 0 running, frozen hours | 0 occurrences |
| engine killed by `assert req_id in self.requests` | 0 |
| loads stranded by promotion starvation (vLLM only promotes on non-preempting steps) | 4,429 watchdog promotions, 0 force-fails |
| 1 key / 2MB per save put (~3.9ms RPC vs ~40us wire on RDMA) | 62,365 puts carrying ~21 keys each |
| admission starvation 71-77% of pressured samples | 0% |

**Sizing law (the actual blocker):** at gmu 0.317 the KV pool was 2,125
blocks = 34k tokens while a trajectory context reaches 28k, so one request
parked on an async pull held ~82% of the pool and all four engines
deadlocked; no in-flight cap can fix a pool below one working set. gmu 0.45
gives 11,645 blocks = 186k tokens (~6-7 contexts) and the deadlock
disappears while preemption pressure remains (KV peaks 0.995, 30-57
preemptions).

## Metric coverage and a parser defect

The driver logs record **82 step metrics per arm**; this README discusses
roughly 20. The omissions were audited after the fact rather than chosen:
`slowest/*`, `perf/throughput`, `cpu_memory_used_gb` and `grad_norm` were
recorded all along and are now included above. The full set is in the
gzipped driver logs - `zgrep -a "step:[0-9]* - " <arm>_driver.log.gz`.

One defect worth carrying forward, because it silently shrinks tables:
`perf/throughput` is the LAST field on the step line, and Ray sometimes
interleaves another actor's output onto that line, gluing an ANSI escape
directly to the value (`246.019...\x1b[36m(TaskRunner`). A parser that
anchors a field as `value(?= - |$)` then drops it - which is exactly what
happened in store step 3, surfacing as a `KeyError` that was worked around
by deleting the metric instead of fixing the parse. Any trailing metric can
vanish this way. Parsers over these logs must strip ANSI first, tolerate
trailing junk after the number, and assert that every step yields the same
key set rather than silently intersecting them.

## Measurement caveats

- **Regime differs from `../pressure-pair/`** (gmu 0.317, 12x64, 25 turns,
  6k obs cap). Cross-pair deltas are not directly comparable: this pair is
  4x wider per rollout, has a 5.5x larger KV pool, longer turns and fatter
  observations (response ~11.1k vs ~7.5k). Compare mechanisms, not numbers.
- 4 steps gives N-of-4 consistency counts; weaker than the prior pair's
  N-of-12, but each step is a 256-trajectory sample (4x larger). Because the
  pair is bistable, the effective N for the *pressured* regime is 2.
- The smoke gate for this pair required only `num_preemptions_total > 0`,
  which the run cleared in its first minutes and then left behind. A gate on
  sustained occupancy (fraction of the engine-busy window above 0.85) would
  have caught the regime shift before the pair ran; the successor pair uses
  one.
- The store arm ran immediately after recompute on the same node with the
  same warmed sandbox pool; tool-time symmetry is +5.2% at 2/4.
- Sandboxes have no network egress; `pip install` attempts burn the 60s
  command timeout in both arms and drive timing_s/gen variance.
- verl's `agent_loop/*/num_preempted` is -1 on this stack; preemption
  counts come from engine `/metrics`.
- Engine scrape probes the mooncake RDMA handshake listener, producing
  benign `SocketHandShakePlugin: malformed json` noise in both arms.

## Files

- `recompute_driver.log.gz`, `store_driver.log.gz` - full driver logs
  (verl per-step metric lines; source of the A/B and drift tables).
- `engine_scrapes.log.gz` - per-minute in-worker engine /metrics samples,
  SNAP-timestamped; arm windows recompute [1789036097,1789042298], store
  [1789042298,1789048207].
- `sidecar_final.log.gz` - netns-fixed sidecar archive tail (mooncake
  op-time histograms).
- `master_postrun.txt.gz` - mooncake master counters after the store arm.
- `connector_v3.py` - the exact connector patch both arms' image carried.
