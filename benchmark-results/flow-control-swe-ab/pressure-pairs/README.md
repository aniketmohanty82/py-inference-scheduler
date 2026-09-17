# Pressure pairs: making preemptions real, then gating them

Two attempts to put the (fixed) integration under genuine KV pressure so the
saturation gate has something to prevent: growing the batch (128 x n4 = 512
trajectories), and shrinking the pool (gmu 0.22 -> 0.19 at the standard
batch 64). Both arms of both pairs run the shared-inflight ledger and
complete endpoint views; the gate (`lq-kv90`: kv 0.90 / waiting 4) is the
only difference between arms.

## TLDR

Batch growth is NOT a pressure knob on this workload -- the sandbox tier
throttles turn arrivals, so 2x trajectories left engines at kv <= 0.71 with
zero preemptions in every healthy step. Pool shrink IS: at gmu 0.19
(~60k-token pool/engine) the baseline saturates for real (kv p99 0.97,
spikes to 1.00) and preempts 22 times. Against that, the gate parked 113
requests, dropped saturated engines 216 times, and produced **12
preemptions (-45%) and fewer saturated samples while generating 5.9% MORE
tokens**. LLM-side throughput is flat (+0.4% on the tool-independent
instrument); an earlier +13.6% wall-throughput claim is RETRACTED -- see
the adversarial-audit correction below.

## Headline pair: gmu 0.19 (batch 64, 64-task dataset, 2 steps)

Identical to the ledger A/B except `GMU=0.19`; both counters cross-checked
by the hook-free engine scraper (agreement exact: 746->768, 768->780).

| Arm | Parks | Drops | Gen wall | Tokens | LLM tok/s* | vs base | Preempt | Preempt/Mtok | kv p50/p90/p99 | kv>=0.9 samples | Turns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-gmu19 | 0 | 0 | 2851.3 s | 4.245 M | 12.35 | — | **22** | 5.18 | 0.08 / 0.57 / 0.97 | 89 / 3655 (2.4%) | 44.5, 41.6 |
| gate-kv90-gmu19 | 113 | 216 | 2657.3 s | 4.494 M | 12.41 | +0.4% | **12** | **2.67** | 0.09 / 0.47 / 0.92 | 57 / 3930 (1.5%) | 43.6, 43.2 |

*Per-trajectory LLM-side rate: mean response tokens / `generate_sequences/mean`,
the tool-independent instrument. Gen wall is NOT a throughput denominator on
this workload (see correction).

Supporting texture:

- Drop reasons are what the design intends: engine groups read at kv
  0.91-0.97 and get excluded while a cooler engine takes the request;
  waiting=4 drops appear when queues actually form (scraper saw waiting up
  to 7 in the baseline, 5 with the gate).
- The gate also capped router-side commitment during bursts: ledger q_max 64
  in the baseline vs 29 with the gate -- parking replaces pile-up.
- Selection stays even in both arms (max share 26.3% vs 27.8%), so the gate
  is not trading balance for its wins.

## Statistics, honestly

22 vs 12 preemption events is ~1.7 sigma on a Poisson rate-ratio test --
directionally strong, not individually conclusive. What earns the result
weight is coherence: preemptions, saturated-sample count, tail kv, and
engine waiting depth improve together while the gate arm does MORE decode
work (+5.9% tokens). Throughput is a wash on the valid instrument (+0.4%
LLM-side). A replicate pair at deeper pressure (gmu 0.185 x 4 steps) is
running to firm up the rate ratio.

## Correction (adversarial audit, 09-17)

An earlier version of this document claimed **+13.6% throughput** for the
gate arm, computed as tokens / sum(`timing_s/gen`). That metric is invalid
on this workload and the claim is retracted:

- `timing_s/gen` tracks the single slowest trajectory per step, which is
  dominated by sandbox tool tails, not generation (house rule: never divide
  rates by timing_s/gen). The slowest-trajectory tool tails were 948/851 s
  in the baseline vs 731/668 s in the gate arm -- the "throughput win" was
  substantially the baseline's worse sandbox tail luck.
- Tool-time pairing symmetry FAILS at the step level: per-trajectory
  `tool_calls/mean` was 44.1 s (baseline) vs 64.8 s (gate) in step 2
  (+47%), far outside the ~10% symmetry band the verl-swe-ab pairs used.
  Wall-clock comparisons between these arms are therefore unresolvable.
  Direction note: the tool luck ran AGAINST the gate arm, so the gate's
  surviving wins are conservative.
- On the tool-independent instrument (response tokens per
  `generate_sequences/mean`), throughput is 12.35 vs 12.41 tok/s: +0.4%.

What survives the audit: the preemption result (counter-based,
scraper-verified, and the gate did 5.9% more decode work while preempting
45% less); the saturation-exposure result (conservative, since the
flow-control watcher polls fastest while parked, oversampling the gate's
own saturated moments); turn counts equal (+0.9%). One disclosed blemish:
the gate arm's step 2 contains 1-2 degenerate 2-turn trajectories
(`num_turns/min: 2`, sandbox-create loss; baseline floor was 10 turns) --
under 1% of arm work, direction favors the baseline.

## The batch-128 attempt (documented, excluded from comparisons)

128 tasks x n4 = 512 trajectories against the same fleet, using a
regenerated 136-task dataset (strict superset of the 64), a 6-node sandbox
pool, and pre-warmed AR image cache. Both arms degraded identically in
their second step (num_turns/mean 38.1->23.9 baseline, 43.4->21.9 gate;
num_turns/min = 2 in every step): a slice of trajectories lost their
sandbox and died at 2 turns. Two scaffold limits, not scheduler behavior:

| Limit | Mechanism |
|---|---|
| Executor ceiling | sandbox create + wait_ready (up to 600 s) blocks a thread in asyncio's default 32-thread pool per AgentLoopWorker; 64 trajectories/worker queue behind 32 threads and the tail starves |
| Cold image pulls | 72 new multi-GB images x per-(image,node) first pulls; batch-64 runs never saw this because node caches were warm from weeks of the same 64 tasks |

The decisive negative: even the healthy full-depth steps at 512
trajectories produced ZERO preemptions with kv max 0.71 and 18 parks / 34
drops. Tool execution slows under sandbox contention, turn arrivals thin
out, and engine concurrency does not scale with batch size. Demand-side
scaling attacks the wrong tier; supply-side (pool) scaling is the operative
pressure knob for agentic RL of this shape.

## Scrutiny

- One run per arm throughout; the gmu-pair deltas are coherent but n=1.
- Preemption counters are pod-cumulative offsets (746 carried in); all
  deltas are within-arm and scraper-verified.
- The batch-128 pair is internally like-for-like (both arms equally
  degraded) but is excluded from result tables per the degraded steps.
- Run artifacts: `/work/sweep3/` (batch-128) and `/work/sweep4/` (gmu pair)
  on the `swe-fc` head pod; scraper tsv archived off-pod.
- A 413 upload failure (working_dir > 100 MiB from accumulated sweep logs)
  cost one submission cycle; runtime envs now exclude `sweep*`.
