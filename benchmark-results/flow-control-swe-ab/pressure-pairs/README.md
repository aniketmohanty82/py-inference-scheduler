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
preemptions (-45%), 36% fewer saturated samples, and +13.6% throughput** on
equal work -- the first run in this series where flow control moves every
metric in the direction it was designed to.

## Headline pair: gmu 0.19 (batch 64, 64-task dataset, 2 steps)

Identical to the ledger A/B except `GMU=0.19`; both counters cross-checked
by the hook-free engine scraper (agreement exact: 746->768, 768->780).

| Arm | Parks | Drops | Gen wall | Tokens | tok/s | vs base | Preempt | Preempt/Mtok | kv p50/p90/p99 | kv>=0.9 samples | Turns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-gmu19 | 0 | 0 | 2851.3 s | 4.245 M | 1,489 | — | **22** | 5.18 | 0.08 / 0.57 / 0.97 | 89 / 3655 (2.4%) | 44.5, 41.6 |
| gate-kv90-gmu19 | 113 | 216 | 2657.3 s | 4.494 M | **1,691** | **+13.6%** | **12** | **2.67** | 0.09 / 0.47 / 0.92 | 57 / 3930 (1.5%) | 43.6, 43.2 |

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
weight is coherence: preemptions, saturated-sample count, tail kv, engine
waiting depth, and throughput all improve together, and the throughput gain
(+13.6%) has the right mechanism behind it (preempted requests recompute
their KV; fewer preemptions = less redo work). No earlier run produced
aligned signals. A replicate pair (or gmu 0.18 for larger counts) would
firm up the rate ratio; single runs on this workload have shown up to 33%
throughput variance, so treat the throughput delta as consistent-with, not
proof.

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
