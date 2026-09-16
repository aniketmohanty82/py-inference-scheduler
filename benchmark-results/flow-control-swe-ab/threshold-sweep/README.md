# Naive flow control on verl's SWE agent loop: threshold sweep

Does `simple_backpressure` produce any gain on a real agentic RL rollout?

## TLDR

No. Preemptions do not fall at any threshold (713 baseline -> 733/743/746,
rising slightly as the gate gets more aggressive), and the throughput deltas
are non-monotonic (-10.3%, -6.9%, +1.6%), which is the signature of noise
rather than effect. The gate demonstrably works -- it parks, it filters, and
at kv 0.50 it measurably suppresses peak saturation -- but on this workload
that control does not convert into fewer preemptions or more tokens per
second.

## Setup

| | |
|---|---|
| Workload | `run_swe.sh` (main), R2E-Gym SWE agent loop, 64 tasks, batch 64 x n4 = 256 trajectories, 4096 prompt / 28672 response, 25 turns, 2 steps |
| Model | Qwen2.5-7B-Instruct + LoRA r32/a32, tp=2, 4 engines on 8xH100, `free_cache_engine=False` |
| Pressure | `gpu_memory_utilization=0.22` (~100k-token KV pool/engine) |
| Routing | `least_queue` in EVERY arm, so the gate is the only variable |
| Arms | baseline (no gate), then kv 0.90/w4, kv 0.70/w3, kv 0.50/w2 |

Routing is held at `least_queue` rather than verl's own balancer because
verl's is least-inflight plus sticky session (the SWE loop reuses one
`request_id` per trajectory, so the LRU cache hits on turns 2..N). Holding a
deterministic least-loaded policy in all four arms removes stickiness as a
variable; the cost is that trajectories can change engine between turns and
lose prefix reuse, which every arm pays equally.

## Results

Throughput is `256 trajectories x response_length/mean / timing_s/gen`.
Preemptions are the pod-wide `vllm:num_preemptions_total` delta within each
arm's own window.

| Arm | Parks | Drops | Gen wall | Tokens | Throughput | vs base | Preemptions | kv>=0.9 samples |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 1968.0 s | 2.178 M | **1,107 tok/s** | — | **713** | 7 / 670 |
| gate-kv90 | 14 | 45 | 2374.6 s | 2.357 M | 993 tok/s | -10.3% | 733 | 18 / 705 |
| gate-kv70 | 48 | 51 | 2116.8 s | 2.182 M | 1,031 tok/s | -6.9% | 743 | 14 / 607 |
| gate-kv50 | 44 | 89 | 1979.0 s | 2.225 M | 1,124 tok/s | +1.6% | 746 | 3 / 663 |

Turns per trajectory were 42.9-43.6 across all arms, so the arms did
comparable work.

## Analysis

**A. The gate engaged.** Parks and drops are non-zero in every gated arm, and
rise with aggressiveness (45 -> 51 -> 89 drops). This is "engaged and did not
help", not "never engaged" -- a distinction earlier attempts could not make.

**B. Preemptions never fall.** 713 -> 733 -> 743 -> 746, monotonically
*increasing* with gate aggressiveness. This is the cleanest signal here and it
runs opposite to the feature's purpose.

**C. Throughput shows no consistent effect.** The deltas are non-monotonic:
the most aggressive setting (kv 0.50) is the *best* of the three. If gating
cost throughput, kv 0.50 would be worst. With one run per arm, and a baseline
that varied 33% between identical earlier runs, +-10% is inside noise.

**D. The gate does control pressure.** kv 0.50 filtered hardest (89 drops) and
produced the fewest saturated samples (3/670 vs 7/670 baseline). The mechanism
works; the outcome does not follow.

**E. Why it cannot help here.** The fleet is idle at **median kv 0.08**, p90
0.33, with brief spikes to 1.00 -- SWE trajectories spend most of their life in
sandbox tool execution, not generation. A threshold gate reacts to *current*
KV, so when a burst trips the threshold the requests causing it are already
resident and growing; parking new arrivals cannot unwind that, it only defers
work the engines would have absorbed during the long idle stretches.

## Scrutiny

- One run per arm. Per-arm differences of this size are not resolvable; only
  the preemption direction (consistent across three arms) is.
- Sandbox pool is fixed at 4 nodes with no autoscaling. Every arm resets the
  pool first, because leaked sandboxes silently starve rollouts (see below).
- Preemption counters are pod-wide prometheus multiprocess aggregates: every
  engine port returns the same number, so they are fleet totals, not per-engine.

## Measurement failures corrected along the way

Recorded because each produced a confident but wrong result first.

| Failure | Symptom | Fix |
|---|---|---|
| Selection bias | "saturation eliminated, 54 -> 0" | per-decision stats describe only the SELECTED endpoint, and a filtering gate removes saturated engines from its own sample; added fleet-wide `FLEET` snapshots |
| Invisible signals | `parks=0` in every run | Ray captures actor stdout, not Python logging; park/drop/FLEET now `print()` |
| Sandbox starvation | `num_turns 2.0`, `response_length ~60`, job still SUCCEEDED | 761 leaked sandboxes against a fixed 4-node pool; every arm now resets first |
| Pooled counters | incoherent "71 vs 112 preemptions" | counter is a pod-wide aggregate read four times, not four engines |
| Wrong baseline | attributed the 2-turn collapse to the missing hook | it was starvation; the hook was not the cause |
