# verl-native SWE A/B - sustained-pressure pair (32B + LoRA, 512-wide)

**TLDR.** Under KV pressure sustained across every step, the external store
cuts prefill work by **2.2x** (148.6M -> 67.0M `local_compute` tokens) and
per-trajectory generate time by **35.1%, 4/4 steps**, on a control arm that
has almost no local cache left to fall back on (`local_cache_hit` 14.32%).

**Do not quote the -37.7% on `timing_s/gen` as a store result.** That metric
is a single trajectory's makespan (see MAKESPAN below) and 155% of its gap is
the straggler's tool time, not KV. The two results that survive that scrutiny
are the token counts and `generate_sequences/mean`.

Store-vs-recompute on verl 0.8.0's native SWE agent loop, 2026-09-11.
Qwen2.5-32B-Instruct + LoRA r32/a32 (fsdp2, dynamic-bsz 16384), **4 GRPO steps
per arm, batch 128 x n 4 = 512 trajectories per rollout**, seed 42, single
8xH200 node, tp=2 (4 engines), image `swe12`. Arms differ by exactly the
kv_transfer_config flags, both verified on the live Hydra command line.

## Regime design

An external KV tier can only matter when the local prefix cache is failing, so
the regime has to hold the KV pool oversubscribed for the whole run. That is
harder than it sounds, because **KV pressure in a multi-turn agent loop is
bistable**. A request here is one turn, and between turns a trajectory holds
zero allocated blocks - its KV sits in the evictable cached tier. Above ~100%
occupancy that tier is squeezed to nothing and every returning turn re-prefills
its whole context, which sustains the occupancy; below ~60% the tier survives,
turns prefill only new tokens, and that sustains the low state. Both are
self-reinforcing, so a regime that merely *reaches* pressure can fall out of it
mid-run and average two incompatible operating points together.

The settings below are chosen so the low state does not exist to fall into:

| knob | value | why |
|---|---|---|
| `data.train_batch_size` | **128** (x n 4 = 512 trajectories) | enough concurrent trajectories to keep the working set above the pool; also divides the 256-row dataset, so 4 steps = exactly 2 epochs |
| `gpu_memory_utilization` | **0.38** | ~106k tokens/engine, 424k across 4 engines |
| `SWE_CMD_TIMEOUT_S` | **15** | a SWE shell command past 15s is a hang; long hangs idle the engine and desynchronise the cohort, which is what lets occupancy fall |
| `RLS_MAX_INFLIGHT_LOADS` | **1** | a 106k pool against a 29.2k max context spares only one parked async-load waiter |
| store flush on weight reset | **on** | `RLS_FLUSH_STORE_ON_RESET`: store keys are content hashes with no weight version, so KV written under earlier weights must not survive a weight sync. Per-boundary evidence in `store.md` |

The pool is **not** linear in gmu - weights are a fixed cost. Fitting two
measured points (gmu 0.317 -> 34k tokens/engine, 0.45 -> 186k) gives
`tokens/engine ~= 1,143k*gmu - 328k`; gmu 0.30 would have yielded ~14k against
a 29.2k max context and could not have started. Size from the fit, not
proportionally.

## Validity (all recorded, all pass)

| gate | recompute | store |
|---|---|---|
| steps completed / rc | 4/4, rc=0 | 4/4, rc=0 |
| wall-clock audit (sum `timing_s/step` vs arm wall) | 7,618s vs 7,920s (302s init) | 5,267s vs 5,640s (373s init) |
| `actor/entropy` range | 0.172-0.215 | 0.304-0.404 (flat; see analysis (e)) |
| `num_turns/mean` (4-step mean) | 28.95 | 29.65 |
| `response/aborted_ratio` | 0.000 | 0.000 |
| `prompt_length/mean` per step | 517.5 / 518.8 / 521.1 / 515.2 | **bit-identical** |
| by_source identity (sums to `prompt_tokens_total`) | EXACT | EXACT |
| connector: `Client not available` / `RPC_FAIL` / `EngineDeadError` / force-fails | n/a | 0 / 0 / 0 / 0 |
| mooncake failed keys (all ops) | n/a | 0 |
| metric coverage | 82/82 keys x 4 steps | 82/82 keys x 4 steps |

**Sustained-pressure gate** (`pressure_gate.py`, run on the engine scrape):
the fraction of the engine-busy window above `kv_cache_usage_perc` 0.85.

| | hot_frac | active snaps | mean_kv | preemptions |
|---|---|---|---|---|
| smoke (gating run) | **0.68** (bar 0.60) | 28 | 0.783 | 179 |
| recompute arm | **0.72** | 69 | 0.802 | **317** |
| store arm | 0.56 | 50 | 0.736 | 325 |

The store arm scoring *below* the bar is the finding, not a failure: the gate
exists to prove the RECOMPUTE arm faced real pressure, and an arm that relieves
its own pressure on identical work is the thing being measured. Peak occupancy
does not discriminate - both arms touch 0.99 - which is why the gate is a
fraction of the busy window. The 0.60 bar is calibrated against a known
bistable run on this harness, whose collapsed steps score 0.50 on this gate
and whose healthy steps score 0.00.

## Serving split (final cumulative counters, 4 engines)

| source | recompute | share | store | share |
|---|---|---|---|---|
| `local_compute` | 148,596,234 | **85.68%** | **67,037,475** | 37.73% |
| `local_cache_hit` | 24,844,960 | 14.32% | 32,874,000 | 18.50% |
| `external_kv_transfer` | 0 | 0% | 77,760,400 | **43.77%** |
| TOTAL (= `prompt_tokens_total`) | 173,441,194 | | 177,671,875 | |

Workloads matched to 2.4% on total prompt tokens. **`local_compute`: 148.6M ->
67.0M = 2.22x reduction, 81.6M tokens of prefill avoided.**

The number that makes this pair readable is the recompute arm's
`local_cache_hit` share: **14.32%**. The control arm has almost no local cache
to fall back on, which is the condition under which an external tier is
supposed to matter at all. Against that, the store supplied **43.77%** of its
prompt tokens from the remote tier.

Mooncake ops (store arm, **zero** failed keys): `save_put` 19,247 ops /
404,142 keys (21.0 keys/op), `load_get` 4,182 ops / 2,547,176 keys,
`lookup_exists` 186,539 ops / 264,393,476 keys.

## Results (4-step means; per-step detail in `recompute.md` / `store.md`)

| verl metric | recompute | store | delta | store lower |
|---|---|---|---|---|
| `timing_s/step` | 1,904s | 1,317s | **-30.9%** | 4/4 |
| `timing_s/gen` | 1,558s | 971s | **-37.7%** | 4/4 |
| `timing_s/agent_loop/generate_sequences/mean` | 348.2s | 226.2s | **-35.1%** | 4/4 |
| `timing_s/agent_loop/generate_sequences/max` | 1,147s | 884s | **-22.9%** | 4/4 |
| `timing_s/agent_loop/tool_calls/mean` | 57.1s | 59.7s | +4.5% | 1/4 |
| `timing_s/agent_loop/tool_calls/max` | 986s | 559s | -43.3% | 3/4 |
| `perf/throughput` | 225.5 | 328.2 | **+45.6%** | 0/4 |
| `num_turns/mean` | 28.95 | 29.65 | +2.4% | 0/4 |
| `response_length/mean` | 6,252 | 6,250 | -0.0% | 2/4 |
| `perf/total_num_tokens` | 3,466,187 | 3,465,082 | -0.0% | 2/4 |
| `actor/entropy` | 0.195 | 0.365 | **+86.6%** | 0/4 |
| `critic/score/mean` | 0.0288 | 0.0303 | +5.1% | 1/4 |
| `actor/grad_norm` | 0.0023 | 0.0084 | +263.3% | 1/4 |
| `actor/perf/cpu_memory_used_gb` | 165 GB | 1,222 GB | **+641.0%** | 0/4 |

`steps store lower` counts steps where the store's value is the smaller one,
so for `perf/throughput` - the one metric here where higher is better - 0/4
means the store was **higher in all four steps**.

Two metrics are reported but should not be read as serving results.
`perf/throughput` is `total_num_tokens / (timing_raw["step"] * n_gpus)`: a
batch-size numerator, identical across arms by construction, over a wall-clock
denominator - so it says "the step finished sooner", not "the engine served
faster". `cpu_memory_used_gb` is the store's host-memory bill, the 8 x 128GB
mooncake segments, not a performance number.

`slowest/*` is omitted from this table on purpose. It reports one trajectory
chosen by `argmax(generate_sequences + tool_calls + compute_score)`
(`agent_loop.py:1146`), so it names a **different trajectory in each step and
each arm** and is not comparable between them: `slowest/tool_calls` reads
-92.5% where the selection-free `tool_calls/max` is -43.3%, and
`slowest/generate_sequences` reads +58.1% where `generate_sequences/max` is
-22.9% in the store's favour. The rows are recorded and kept in the per-arm
files.

### MAKESPAN: `timing_s/gen` and `timing_s/step` are one trajectory, not the rollout

`timing_s/gen` equals the slowest trajectory's `generate_sequences +
tool_calls` to within 0.1-3% in **every step of both arms**:

| | slowest gen | slowest tools | sum | `timing_s/gen` | ratio |
|---|---|---|---|---|---|
| recompute, 4-step mean | 559s | 986s | 1,544s | 1,558s | 99.1% |
| store, 4-step mean | 884s | 75s | 958s | 971s | 98.7% |

So the rollout wall is a makespan set by one trajectory out of 512, and its
gap decomposes as:

| component | contribution to the 588s mean gap |
|---|---|
| straggler's tool time (986s vs 75s) | **+911s** for the store |
| straggler's generation time (559s vs 884s) | **-325s** against the store |
| net | 586s (observed: 588s) |

**155% of the wall-clock win is the straggler's tool time**, partially offset
by the store's straggler generating *longer*. The tool pathology behind it is
characterised in analysis (b) and is not a KV effect. `perf/throughput`
inherits the same denominator and the same caveat.

What is NOT affected: the by_source token counts (a direct count), and
`generate_sequences/mean` (a mean over 512 trajectories, where the straggler
contributes ~1s of 348s).

### Pressure held in every step

The per-step engine decomposition lives in `recompute.md` and `store.md`.
The summary that matters: the recompute arm held `hot%` at 60-73% in **all
four** steps with `local_cache_hit` never above 20.8%, so there is no
low-occupancy step diluting the means above, and no boundary the 4-step
means average across.

### Generation throughput (decode tokens per trajectory-second)

`generation_tokens_total` over `generate_sequences/mean x 512 trajectories` -
independent of how many tokens a step happened to present. 4-step mean:

| | recompute | store | ratio |
|---|---|---|---|
| decode tokens per trajectory-second | 5.61 | 9.72 | **1.73x** |

Per-step values (1.46x-2.18x, store higher in 4/4) are in the per-arm files.

### Cost per avoided token (recorded inputs only, no FLOPs model)

| | value |
|---|---|
| `generate_sequences/mean`, 4-step mean | rc 348.2s, st 226.2s (delta 122.1s/trajectory) |
| `local_compute` per trajectory | rc 72,557, st 32,733 (delta 39,824 tokens) |
| **cost per avoided `local_compute` token** | **3.07 ms** |
| `external_kv_transfer` per trajectory (store) | 37,969 tokens |

This is a **capacity** number, not a hardware one. `generate_sequences` is
wall time and includes queueing, so removing prefill also shortens the queue
for every other trajectory; the figure therefore carries a large contention
amplification and rises with load. It also credits the entire latency delta to
avoided prefill, while the arms differ in scheduling too. Recompute it per
regime rather than carrying 3.07 ms anywhere else.

## Analysis

**(a) The defensible win is prefill work, not wall clock.** `local_compute`
falls 2.22x and `generate_sequences/mean` falls 35.1% at 4/4 - both immune to
the straggler. The wall-clock numbers are makespan artifacts; see MAKESPAN.

**(b) The tool-time tail is NOT worse in recompute - it is normal, and the
store arm got three lucky draws.** Mean tool time is identical (+4.5%, store
*higher*); only the per-step maximum differs, and only in steps 2-4.
Normalising the tail by `num_turns/max` (the tail trajectory is a max-turns
trajectory) against every arm-run recorded on this harness:

| run | arm | tool tail, s/turn per step | cmd timeout |
|---|---|---|---|
| prior run, no flush | recompute | 11.0, 16.6, 16.8, 17.5 | 60s |
| prior run, no flush | store | 16.7, 16.2, 17.3, 14.1 | 60s |
| this pair, smoke | store | 14.5, 16.4 | 15s |
| this pair | recompute | 13.6, 16.2, 15.5, 15.3 | 15s |
| this pair | store | 14.3, **6.5, 6.1, 7.4** | 15s |

Fifteen of eighteen observations fall in 14-17.5 s/turn and recompute sits in
the middle of them. The three outliers are this pair's store arm, steps 2-4 -
while **that same arm's step 1 (14.3) and its own smoke (14.5, 16.4) match the
norm**. `tool_calls/max` is 9-17x `tool_calls/mean`, so the per-step maximum is
one extreme order statistic out of 512 trajectories; three consecutive low
draws from a tail that heavy is unremarkable.

Note that ~16 s/turn also appears in the prior run, where the timeout was 60s
and nothing was near a ceiling. So ~16s is simply what the worst trajectory's
commands cost; in this pair it coincides with the 15s cap.

Two mechanisms were tested and ruled out. Sandbox contention is **refuted by
direct measurement** - a 60s sampler across both arms (`sandbox_fleet.log`)
shows indistinguishable occupancy:

| phase | samples | sandboxes mean | p90 | max | pods not Running (max) |
|---|---|---|---|---|---|
| recompute | 127 | 152.2 | 364 | 515 | 183 |
| store | 90 | 154.4 | 364 | 539 | 178 |

Exec retries are also not visible (0 `SandboxError` in both arms), though that
is weak evidence because `SandboxClient.exec` swallows the exception without
logging.

**Consequence for the headline.** `timing_s/gen` is 99% the straggler's
makespan and the straggler's tool time is 155% of the wall-clock gap, so the
-37.7% rests on these three tail draws. That is the concrete reason the
wall-clock numbers are demoted rather than merely caveated.

To make the tail measurable instead of anecdotal, record per-exec latency and
turn count for the tail trajectory, and log rc-124 distinguishing a
command-kill from an RPC-deadline hit.

**(c) In the store arm the gating trajectory is capped, not slow.**
`slowest/response_length` is **28,672 in all four store steps** - exactly
`data.max_response_length` - against 5,299-16,478 in recompute. So the store's
worst trajectory is one that generated until the configured ceiling stopped
it, while recompute's is one that was still grinding. On the selection-free
metric the store is faster at generation too (`generate_sequences/max`
-22.9%, 4/4). The honest reading is that the store moved the binding
constraint onto a config limit, which also means **the response cap is now
truncating the store arm's tail and any longer-horizon run should raise it or
report the clip rate.**

**(d) Step 1 is not comparable to steps 2-4 in either arm.** `num_turns/mean`
is 45.5 at step 1 and 23.0-24.9 afterwards, in both arms, on the same tasks.
Steps 2-4 are stable to within 6%, so the pair reads as one cold step plus
three steady ones. It is symmetric across arms on identical tasks, so it does
not change any sign in the comparison.

**(e) Unexplained: the store arm runs at ~2x the recompute arm's entropy.**
`actor/entropy` is 0.304-0.404 in the store arm against 0.189-0.215 in
recompute (+86.6% on the mean). It is FLAT across all four steps and well
inside the <1.0 validity gate, and it is present from step 1, so it is a level
offset rather than a drift. It appeared alongside a 43.77% external-transfer
share, so the leading hypothesis is numerical - KV recomputed locally and KV
pulled from the tier are not bit-identical, because a prefix assembled from
pulled blocks hits different attention chunk boundaries - and at this transfer
share the difference may stop being negligible. `actor/grad_norm` reaching
0.021 by step 4 in the store arm (recompute: 0.003) may be the same effect.
The competing explanation is simply that the arms sampled divergent
trajectories and visited different states; `critic/score/mean` is nearly
identical between arms (0.0288 vs 0.0303), which argues the task outcomes are
comparable but does not settle it. **Not a blocker for this pair, but it
should be characterised before the store is used for convergence work** - the
cheap test is a 1-step run comparing logprobs of pulled versus recomputed
blocks on identical prompts.

`actor/ppo_kl` shows no systematic split between the arms (recompute
+1.2e-05..+2.5e-04, store +6.4e-05..-3.2e-04), so the usual off-policy
tripwire is quiet.

## Measurement caveats

- 4 steps = exactly 2 epochs of the 256-row dataset, so **every task appears
  twice**, once in steps 1-2 and once in steps 3-4 (reshuffled). Identical in
  both arms. With the flush on, a repeated task gets no cross-step KV reuse by
  construction; if it did, that would show as a flush leak.
- The store arm ran immediately after recompute on the same node and the same
  warmed sandbox pool. Tool-time symmetry is +4.5% on the mean.
- The wipe is issued by **each engine's** tp_rank 0, so `remove_all(force=True)`
  runs up to 4x per boundary (visible as different key counts per generation).
  Idempotent and harmless, but it is 4 global wipes where 1 would do.
- verl's `agent_loop/*/num_preempted` is -1 on this stack; preemption counts
  come from engine `/metrics`.
- **`tool_calls` is not purely tool time.** `simple_timer("tool_calls")` wraps
  `await sandbox_future`, so a trajectory's FIRST tool call is charged the
  whole sandbox boot - pod create + `wait_ready` (600s default) +
  `BASELINE_CMD` (120s) - minus whatever turn-1 generation already hid. It
  also absorbs queueing for the shared executor: 512 trajectories against
  8 AgentLoopWorkers x 32 threads = 256. This plausibly explains why the store
  arm's `tool_calls/mean` is the HIGHER of the two (+4.5%): its faster turn-1
  generation hides less of the boot. Timing the boot separately is the fix.
- `slowest/*` selects `argmax(generate_sequences + tool_calls + compute_score)`
  (`agent_loop.py:1146`), so it is a **different trajectory** in each step and
  each arm - the two arms' rows are not the same task. This bit the first
  draft of this README, which read -92.5% off `slowest/tool_calls` where the
  selection-free `tool_calls/max` is -43.3%. Prefer `/max` over `slowest/*`
  for any arm-to-arm comparison.
- The store arm's `slowest/response_length` is pinned at `max_response_length`
  (28,672) in every step, so its generation tail is censored by config.

## Files

- **`metrics.csv`** - every recorded number in one flat file: 93 metrics x
  4 steps x 2 arms, plus per-arm means and the delta. 82 verl step metrics
  carry their own names; the 11 derived from the engine scrapes are prefixed
  `engine/`. Regenerate with `python3 -c "import harvest34 as h; ..."` - see
  `write_csv` in `harvest34.py`.
- **`recompute.md`, `store.md`** - per-step detail for each arm: all 22 verl
  step metrics, the engine-side pressure decomposition, and (store) the flush
  events and mooncake op totals. This README carries only 4-step means.
- `recompute_driver.log.gz`, `store_driver.log.gz` - full driver logs, source
  of every verl table here (`zgrep -a "step:[0-9]* - "`).
- `recompute_scrape.log.gz`, `store_scrape.log.gz` - per-minute in-worker
  engine `/metrics`, SNAP-timestamped; source of by_source and pressure.
- `smoke_driver.log.gz`, `smoke_scrape.log.gz`, `smoke_gate.txt` - the gating
  run and its verdict.
- `sandbox_fleet.log.gz` - 60s samples of live sandbox CRs / pending pods
  across both arms; the evidence that refutes the contention explanation in
  analysis (b).
- `harvest34.py` - regenerates every table above from the logs.
- `pressure_gate.py` - the sustained-pressure gate; run it on any scrape.
- `p34_arm.sh`, `p34_smoke.sh` - the exact run scripts (regime rationale in
  the header comments).
