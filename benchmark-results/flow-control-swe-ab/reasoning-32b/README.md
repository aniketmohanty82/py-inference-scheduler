# Flow control at reasoning scale: Qwen3-32B, sustained saturation

The regime the 7B work pointed at. At 7B the gate reliably cut preemptions
(-30% to -46% across five pairs) but each preemption was too cheap to matter:
re-prefill is ~10x cheaper than decode, so all 509 events in the storm grid
wasted only ~5% of compute and any saving vanished under noise. A 32B model
with long reasoning contexts makes each eviction ~5x costlier. Two clean
pairs were run; the second adds engine-side performance instrumentation.

## TLDR

At genuine sustained saturation (fleet KV p90 0.98, hard-ceiling time
3-6%), the gate cuts preemptions **-34% and -40% per generated Mtok across
two independent pairs** (p ~= 5e-4 and 1e-5) and cuts **mean engine queue
wait 3.8 s -> 1.0 s (-74%)**. It does this by holding requests at the router
(13,670 parks) instead of letting engines overshoot into eviction. The cost,
now measurable for the first time: **decode throughput per busy
engine-second falls ~8.5%** and the **prefix-cache hit rate falls 30.7% ->
24.2%**, because parked trajectories return to find their prefix blocks
evicted. Net compute is close to a wash -- the gate trades preemption
recompute for cache-miss recompute of similar size -- so the win is
stability and engine-side latency, not tokens per GPU-hour.

## Setup (both pairs)

| | |
|---|---|
| Model | Qwen3-32B (reasoning) + LoRA r32/a32, `load_format=safetensors` |
| Hardware | 2 x 8xH100 80GB (16 GPUs), tp=4 -> **4 replicas**, `nnodes=2` |
| Image | swe13 (`sha256:8cdf4fb8...`): verl 0.9.0 **and** flash-attn 2.8.3 |
| Workload | R2E-Gym SWE agent loop, batch 64 x n4 = 256 trajectories, 25 turns, 20k-char observations, 4096 prompt / 28672 response, 1 step/arm |
| Pressure | gmu 0.35 -> ~16GB engine weights + ~10GB KV per GPU (~155k tokens/replica). Saturation comes from the model's footprint and context length, NOT a starved pool |
| Arms | baseline (no gate) vs `simple_backpressure` kv 0.90 / waiting 4; gate is the only difference |

Why this stack: verl 0.8 could not load 32B here (its weight sync
materialises the full 64GB model per rank). verl 0.9 ships LoRA adapters only
(`base_sync_done`, engine_workers.py:667); tp=4 across two nodes halves both
the engine weight burden and the FSDP actor shard, taking the per-GPU total
from 79GB (OOM) to ~62GB.

## Instruments

| Quantity | Source | Notes |
|---|---|---|
| Preemptions | vLLM `num_preemptions_total`, one delta per POD (engines sharing a pod report the same aggregate); scraper and hook FLEET lines agree exactly | per-arm `PROMETHEUS_MULTIPROC_DIR`; first-of-attempt deltas when an arm retried |
| Decode throughput | `generation_tokens_total` delta / busy engine-seconds (samples with running > 0, 15 s each) | tool-time-free; the rollout wall is NOT a rate denominator on this workload |
| Engine queue wait | `request_queue_time_seconds` sum / count deltas | the quantity the gate moves directly |
| Prefix-cache hit rate | `prefix_cache_hits` / `prefix_cache_queries` deltas | |
| KV distribution | scraper (uniform 15 s sampling) | hook FLEET distributions are park-biased ~10x |
| Per-token latency | not exported by vLLM 0.29 under `time_per_output_token_seconds` | column empty |

## Results

### Pair 2 (09-23, fresh pods, engine-side instrumentation)

| Metric | baseline | gate kv 0.90 / w 4 | delta |
|---|---|---|---|
| **Preemptions** | **194** | **110** | **-43%** |
| Generated tokens (engine) | 1.722 M | 1.638 M | -4.9% |
| **Preemptions / Mtok (gen)** | **112.7** | **67.2** | **-40%**, z = 4.38, p = 1.2e-5 |
| **Mean engine queue wait** | **3.78 s** | **0.98 s** | **-74%** (1,883 vs 1,923 requests) |
| Decode throughput (gen tok / busy engine-s) | 897 | 821 | **-8.5%** |
| Total tokens (gen + prompt) / busy engine-s | 6,883 | 7,607 | +10.5% |
| Prompt tokens (prefill) | 11.49 M | 13.54 M | +17.8% |
| Prefix-cache hit rate | 30.7% | 24.2% | -6.5 pp |
| KV >= 0.99 (hard ceiling) share | 5.9% | 3.1% | -47% rel. |
| KV >= 0.90 share | 27.0% | 32.3% | held at threshold by design |
| Parks / drops | 0 / 0 | 13,670 / 1,256 | |
| Mean response length (verl) | 8,236 | 8,927 | +8.4% |
| Rollout wall (`timing_s/gen`) | 535 s | 568 s | +6.0% (descriptive only) |

A second baseline rollout (an attempt whose rollout completed before an OOM
in the update) gives the baseline's own spread: 929 vs 897 tok/busy-s
(3.5%), 4.35 vs 3.78 s queue wait, 111.3 vs 112.7 preempt/Mtok, 33.4% vs
30.7% hit rate.

### Pair 1 (09-22, counters clean, no performance scraper)

| Metric | baseline | gate | delta |
|---|---|---|---|
| Preemptions | 164 | 122 | -26% |
| Tokens (verl response accounting) | 2.076 M | 2.328 M | +12% |
| Preemptions / Mtok | 79.0 | 52.4 | -34%, z ~= 3.46, p ~= 5e-4 |
| Parks / drops | 0 / 0 | 13,063 / 1,141 | |

## Analysis

**A. The preemption effect is replicated and large.** -34% then -40% per
Mtok on two pairs run on different pod generations, each individually
significant, consistent with the -30..-46% band from five 7B pairs. The
mechanism is visible in the ceiling-time column: the gate roughly halves the
time engines spend at KV >= 0.99, while holding them near 0.90 (the share at
>= 0.90 is slightly higher in the gate arm, which is the threshold doing its
job, not a regression).

**B. Engine queue wait collapses.** 3.78 s -> 0.98 s. Requests that reach an
engine under the gate are admitted almost immediately; the waiting has moved
to the router (13,670 parks). This is the first direct measurement of what
the gate does to engine-side latency, and it is the metric a serving
deployment with tail-latency SLOs would care about.

**C. The cost is real and now measured: ~8.5% decode throughput and 6.5 pp
of prefix-cache hit rate.** Baseline-to-baseline spread is 3.5% on
throughput, so -8.5% is suggestive at n = 1 for the gate arm, not
conclusive. The hit-rate drop has a clean mechanism: a parked trajectory's
next turn arrives late, and under saturation its prefix blocks have been
evicted by then, so it re-prefills. Prompt tokens rose 17.8% (about half
attributable to the 8.4% longer responses, the rest to cache misses).

**D. Net compute is close to a wash.** The 84 avoided preemptions save
roughly 0.8 M tokens of re-prefill; the cache misses cost roughly 1.1 M
extra prompt tokens. Total tokens processed per busy engine-second is +10.5%
for the gate arm, decode-only is -8.5%; which number is "throughput" depends
on how prefill is priced. The defensible statement: on this workload the
gate converts preemption recompute into cache-miss recompute of similar
magnitude, and buys lower ceiling time and a 4x reduction in engine queue
wait for it.

**E. Where this leaves flow control.** As a *throughput* optimisation for
RL rollouts, the evidence across 7B and 32B says no measurable gain. As a
*stability and latency* control -- fewer evictions, no ceiling thrash,
predictable engine queues -- the effect is large, replicated, and
significant. That is the design goal ("do not force replicas into
preemption"), delivered, with its price tag attached.

## Scrutiny

- One step per arm; the gate arm is n = 1 for the performance metrics.
  Preemption and queue-wait deltas dwarf the baseline's own spread;
  throughput and hit-rate deltas do not, and are stated as suggestive.
- verl 0.9's legacy trainer (`main_ppo_v0`, required because the V1 trainer
  needs the absent `transfer_queue` package) emits no per-trajectory
  timings, so the 7B audit's tool-independent instruments could not be
  computed; the engine-side counters replace them and are strictly better
  (no sandbox time in any denominator).
- Both arms' first attempts in this pair's history hit a 9 GiB CUDA OOM in
  the update (the LM-head logits at a 32k token budget): the configuration
  sits ~0.5 GB from the card ceiling and passes or fails on packing luck.
  Retries within an arm re-use the arm's prometheus dir, so preemptions are
  first-of-attempt deltas, cross-checked against the scraper.
- An earlier rerun wedged twice in the cross-node update with all ranks in
  `epoll_wait` and no NCCL watchdog -- a bootstrap-handshake hang. The
  scraper at the time swept every listening port on the pod with HTTP GETs,
  including the actors' NCCL listeners; it now probes only ports owned by
  vLLM server processes and the worker pods were recreated (they carried
  ~900 zombie/orphan processes from earlier failed jobs). Both fixes were
  applied together, so the hang's cause is a strong hypothesis, not proven.
- Configuration departures from main's SWE script: LoRA r32, `use_v1=False`,
  tp=4 / nnodes=2, gmu 0.35, 1 step, token budgets raised to 32768, per-arm
  `PROMETHEUS_MULTIPROC_DIR`. Raw logs and scraper series are archived
  off-pod; `perf_window.py` and `engine_scraper.sh` in this directory
  reproduce every engine-side number.
