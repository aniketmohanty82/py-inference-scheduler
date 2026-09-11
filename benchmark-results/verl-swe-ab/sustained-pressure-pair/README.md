# verl-native SWE A/B - sustained-pressure pair (32B + LoRA, 512-wide)

**TLDR.** Under KV pressure that lasts the whole run instead of two steps, the
external store cuts recompute work by **2.2x** and the rollout wall clock by
**37.7%, 4/4 steps** - where the previous pair could only show 5.6% at 2/4.
The per-step flush added since that pair also holds: the store arm's entropy
drift (0.198 -> 0.775 monotonic) is **gone**, flat at 0.30-0.40 across four
steps against a flat recompute control.

Store-vs-recompute on verl 0.8.0's native SWE agent loop, 2026-09-11.
Qwen2.5-32B-Instruct + LoRA r32/a32 (fsdp2, dynamic-bsz 16384), **4 GRPO steps
per arm, batch 128 x n 4 = 512 trajectories per rollout**, seed 42, single
8xH200 node, tp=2 (4 engines), image `swe12`. Arms differ by exactly the
kv_transfer_config flags, both verified on the live Hydra command line.

## Why the regime changed (read this before comparing to pair-v2)

`../pressure-pair-v2/` turned out to be **bistable**: its steps 1-2 ran the KV
pool at 93-96% and its steps 3-4 at 20-27%, on provably identical work, so its
4-step means averaged across a boundary they should not cross. This pair is
built so that no low-occupancy state exists to fall into.

| knob | pair-v2 | here | why |
|---|---|---|---|
| `data.train_batch_size` | 64 | **128** | 2x trajectories in flight; still divides the 256-row dataset, so 4 steps = exactly 2 epochs |
| `gpu_memory_utilization` | 0.45 | **0.38** | pool 186k -> ~106k tokens/engine (424k total) |
| `SWE_CMD_TIMEOUT_S` | 60 | **15** | 35-69% of every pair-v2 rollout was engine-idle behind one straggler's 60s hangs |
| `RLS_MAX_INFLIGHT_LOADS` | 2 | **1** | a 106k pool against a 29.2k max context spares only one parked waiter |
| store flush on weight reset | off | **on** | see DRIFT below |

The pool is **not** linear in gmu - weights are a fixed cost. Fitting the two
recorded points (0.317 -> 34k tokens/engine, 0.45 -> 186k) gives
`tokens/engine ~= 1,143k*gmu - 328k`; gmu 0.30 would have yielded ~14k and
could not have started.

## Validity (all recorded, all pass)

| gate | recompute | store |
|---|---|---|
| steps completed / rc | 4/4, rc=0 | 4/4, rc=0 |
| wall-clock audit (sum `timing_s/step` vs arm wall) | 7,618s vs 7,920s (302s init) | 5,267s vs 5,640s (373s init) |
| `actor/entropy` range | 0.172-0.215 | 0.304-0.404 (flat; see DRIFT) |
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
fraction of the busy window. pair-v2's collapsed steps score 0.50 on this same
gate and its healthy steps score 0.00.

## Serving split (final cumulative counters, 4 engines)

| source | recompute | share | store | share |
|---|---|---|---|---|
| `local_compute` | 148,596,234 | **85.68%** | **67,037,475** | 37.73% |
| `local_cache_hit` | 24,844,960 | 14.32% | 32,874,000 | 18.50% |
| `external_kv_transfer` | 0 | 0% | 77,760,400 | **43.77%** |
| TOTAL (= `prompt_tokens_total`) | 173,441,194 | | 177,671,875 | |

Workloads matched to 2.4% on total prompt tokens. **`local_compute`: 148.6M ->
67.0M = 2.22x reduction, 81.6M tokens of prefill avoided.**

The headline difference from pair-v2 is the recompute arm's `local_cache_hit`
share: **14.32% here vs 61.74% there.** That is the regime doing its job - the
recompute arm now genuinely has almost no local cache to hit, which is the
condition under which an external tier is supposed to matter. The store
supplied 43.77% of all prompt tokens from the remote tier (pair-v2: 10.38%).

Mooncake ops (store arm, **zero** failed keys): `save_put` 19,247 ops /
404,142 keys (21.0 keys/op), `load_get` 4,182 ops / 2,547,176 keys,
`lookup_exists` 186,539 ops / 264,393,476 keys.

## Results (4 steps; delta = store vs recompute on the 4-step mean)

| verl metric | rc s1 | rc s2 | rc s3 | rc s4 | st s1 | st s2 | st s3 | st s4 | delta | store lower |
|---|---|---|---|---|---|---|---|---|---|---|
| `timing_s/step` | 2,641 | 1,931 | 1,321 | 1,725 | 2,020 | 1,013 | 1,172 | 1,063 | **-30.9%** | **4/4** |
| `timing_s/gen` | 2,072 | 1,657 | 1,054 | 1,450 | 1,479 | 763 | 879 | 762 | **-37.7%** | **4/4** |
| `timing_s/agent_loop/generate_sequences/mean` | 888.7 | 175.5 | 174.7 | 154.1 | 577.8 | 77.7 | 130.8 | 118.4 | **-35.1%** | **4/4** |
| `timing_s/agent_loop/tool_calls/mean` | 86.9 | 48.4 | 43.0 | 50.2 | 89.5 | 59.7 | 44.7 | 44.9 | +4.5% | 1/4 |
| `timing_s/agent_loop/slowest/tool_calls` | 886 | 1,055 | 1,009 | 992 | 43.7 | 31.8 | 74.4 | 147.6 | -92.5% | 4/4 |
| `timing_s/agent_loop/slowest/generate_sequences` | 1,137 | 599 | 44.3 | 455 | 1,388 | 730 | 804 | 613 | +58.1% | 0/4 |
| `perf/throughput` | 267.0 | 177.9 | 256.9 | 200.1 | 333.5 | 316.1 | 311.1 | 352.1 | **+45.6%** | **4/4** |
| `num_turns/mean` | 45.48 | 23.45 | 23.85 | 23.02 | 46.14 | 23.53 | 24.93 | 24.00 | +2.4% | 0/4 |
| `response_length/mean` | 10,499 | 4,848 | 4,783 | 4,877 | 10,007 | 4,485 | 5,173 | 5,333 | -0.0% | 2/4 |
| `perf/total_num_tokens` | 5.64M | 2.75M | 2.72M | 2.76M | 5.39M | 2.56M | 2.92M | 2.99M | -0.0% | 2/4 |
| `critic/score/mean` | 0.051 | 0.012 | 0.016 | 0.037 | 0.047 | 0.016 | 0.020 | 0.039 | +5.1% | 1/4 |
| `actor/grad_norm` | 0.003 | 0.001 | 0.002 | 0.003 | 0.003 | 0.002 | 0.008 | 0.021 | +263% | 1/4 |
| `actor/perf/cpu_memory_used_gb` | 154.0 | 167.1 | 169.1 | 169.5 | 1,213 | 1,222 | 1,226 | 1,226 | +641% | 0/4 |

`cpu_memory_used_gb` is the store's host-memory cost: the 8 x 128GB mooncake
segments. `perf/throughput` is included because it is now 4/4, but it remains
`total_num_tokens / (timing_raw["step"] * n_gpus)` - a batch-size numerator
over a wall-clock denominator - so read it as "the step finished sooner", not
as a serving rate.

### Per-step pressure decomposition (engine scrape)

| recompute | busy min | kv_avg | hot% | `waiting` peak | preempt | `local_compute` | `local_cache_hit` | hit% |
|---|---|---|---|---|---|---|---|---|
| step 1 | 29 | 0.827 | 73% | 299 | 162 | 66,551,848 | 5,937,072 | 8.2% |
| step 2 | 14 | 0.725 | 67% | 95 | 63 | 27,549,011 | 5,866,304 | 17.6% |
| step 3 | 14 | 0.678 | 60% | 95 | 48 | 28,237,380 | 5,384,976 | 16.0% |
| step 4 | 12 | 0.774 | 69% | 94 | 44 | 26,040,109 | 6,818,800 | 20.8% |

| store | busy min | kv_avg | hot% | `waiting` peak | preempt | `local_compute` | `local_cache_hit` | `external_kv_transfer` |
|---|---|---|---|---|---|---|---|---|
| step 1 | 20 | 0.817 | 76% | 303 | 170 | 30,308,327 | 5,709,712 | 36,039,888 |
| step 2 | 9 | 0.564 | 20% | 50 | 49 | 8,824,131 | 12,446,288 | 10,639,920 |
| step 3 | 10 | 0.670 | 55% | 108 | 58 | 14,310,666 | 6,343,872 | 16,229,568 |
| step 4 | 11 | 0.584 | 33% | 97 | 48 | 13,584,767 | 8,018,528 | 14,839,568 |

Recompute holds hot% 60-73% in **every** step with hit% never above 20.8%.
That is the property pair-v2 lacked, and it is why the deltas above are 4/4
instead of 2/4.

### Generation throughput (decode tokens per trajectory-second)

`generation_tokens_total` for the rollout, over
`generate_sequences/mean x 512 trajectories`. Independent of how many tokens a
step happened to present:

| step | recompute | store | ratio |
|---|---|---|---|
| 1 | 2.61 | 4.05 | 1.55x |
| 2 | 7.41 | 16.14 | 2.18x |
| 3 | 5.80 | 9.04 | 1.56x |
| 4 | 6.63 | 9.67 | 1.46x |

### Cost per avoided token (recorded inputs only, no FLOPs model)

| | value |
|---|---|
| `generate_sequences/mean`, 4-step mean | rc 348.2s, st 226.2s (delta 122.1s/trajectory) |
| `local_compute` per trajectory | rc 72,557, st 32,733 (delta 39,824 tokens) |
| **cost per avoided `local_compute` token** | **3.07 ms** |
| `external_kv_transfer` per trajectory (store) | 37,969 tokens |

3.07 ms here vs 2.7 ms in pair-v2 - the marginal cost rises with contention,
as expected, since `generate_sequences` is wall time and includes queueing.
Same caveats as `../pressure-pair-v2/`: this is a capacity number, not a
hardware number, and it credits the entire latency delta to avoided prefill.

## DRIFT: fixed

| step | rc entropy | st entropy | pair-v2 st entropy (no flush) |
|---|---|---|---|
| 1 | 0.189 | 0.404 | 0.198 |
| 2 | 0.172 | 0.355 | 0.265 |
| 3 | 0.215 | 0.304 | **0.390** |
| 4 | 0.206 | 0.395 | **0.775** |

The monotonic climb is gone. The flush fires at every weight-update boundary
and removes a growing key set - `removed 57596 keys`, `81450`, `140745`,
`188303` - with zero `Client not available` / `RPC_FAIL` / `EngineDeadError`.
`actor/ppo_kl`'s consistent sign flip in pair-v2 (rc negative, store positive)
also **does not reproduce**: here rc is +1.2e-05..+2.5e-04 and store is
+6.4e-05..-3.2e-04, i.e. no systematic split. Both are consistent with
stale-KV having been the cause and the flush having removed it.

**NEW, unexplained: a level offset.** The store arm now sits at ~2x the
recompute arm's entropy from step 1 onward (+86.6% on the mean) where pair-v2
step 1 differed by only 5%. It is FLAT, well inside the <1.0 gate, and cannot
be staleness (step 1 has no prior weights to be stale against). It appeared
together with the jump to 43.77% external transfer, so the leading hypothesis
is numerical: KV recomputed locally and KV pulled from the tier are not
bit-identical (different attention chunk boundaries), and at this transfer
share the difference is no longer negligible. `actor/grad_norm` rising to
0.021 by step 4 in the store arm may be the same effect. **Not a blocker, but
it should be characterised before convergence work** - the cheap test is a
1-step run comparing logprobs of pulled vs recomputed blocks on identical
prompts.

## Analysis

**(a) The store's win is now in wall clock, not just per-trajectory time.**
pair-v2 moved `timing_s/gen` by 5.6% because the rollout was straggler-bound
and the straggler was sandbox-bound in both arms. Here it moves 37.7% at 4/4.

**(b) Part of that is a second-order systems effect, not KV.**
`slowest/tool_calls` is 886-1,055s in recompute and 32-148s in store - a 20x
gap - while `tool_calls/mean` is within 4.5%. So the *mean* tool cost is
identical and only the worst trajectory differs. The plausible mechanism is
that recompute's 40%-longer rollouts keep more sandboxes alive concurrently,
so sandbox exec RPCs queue and hit the 45s client ceiling
(`SWE_CMD_TIMEOUT_S=15` + 30s margin) far more often. That makes some of the
37.7% a virtuous circle (shorter rollout -> less sandbox contention -> shorter
rollout) rather than a direct KV saving. **This is a correlation with a
mechanism, not a demonstrated cause**; separating it needs a run with
sandbox-side per-exec latency recorded, which is not in this pair.

**(c) The bottleneck moved to generation, which is the honest place for it.**
`slowest/generate_sequences` is *higher* in the store arm (+58.1%, 0/4): its
gating trajectory spends 1,388s of 1,432s total in generation, while
recompute's splits 1,137s generation / 886s tools. The store did not make the
gating trajectory faster at generating - it removed the tool-timeout tail.

**(d) Step 1 is not comparable to steps 2-4 in either arm.** `num_turns/mean`
is 45.5 at step 1 and 23.0-24.9 afterwards, in both arms, on the same tasks.
Steps 2-4 are stable to within 6%, so the pair reads as one cold step plus
three steady ones. Unlike pair-v2 this is symmetric across arms and does not
change any sign.

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
- `slowest/*` selects `argmax(generate_sequences + tool_calls + compute_score)`
  (`agent_loop.py:1146`), so it is a **different trajectory** in each step and
  each arm - the two arms' rows are not the same task.

## Files

- `recompute_driver.log.gz`, `store_driver.log.gz` - full driver logs, source
  of every verl table here (`zgrep -a "step:[0-9]* - "`).
- `recompute_scrape.log.gz`, `store_scrape.log.gz` - per-minute in-worker
  engine `/metrics`, SNAP-timestamped; source of by_source and pressure.
- `smoke_driver.log.gz`, `smoke_scrape.log.gz`, `smoke_gate.txt` - the gating
  run and its verdict.
- `harvest34.py` - regenerates every table above from the logs.
- `pressure_gate.py` - the sustained-pressure gate; run it on any scrape.
- `p34_arm.sh`, `p34_smoke.sh` - the exact run scripts (regime rationale in
  the header comments).
