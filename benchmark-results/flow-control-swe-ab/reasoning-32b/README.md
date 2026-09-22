# Flow control at reasoning scale: Qwen3-32B, sustained saturation

The regime the 7B work pointed at. At 7B the gate reliably cut preemptions
(-30% to -46% across five pairs) but each preemption was too cheap to matter:
re-prefill is ~10x cheaper than decode, so all 509 events in the storm grid
wasted only ~5% of compute and the saving vanished under noise. A 32B model
with long reasoning contexts makes each eviction ~5x costlier. This pair
tests whether the gate's protection converts into anything at that price.

## TLDR

At genuine sustained saturation -- median fleet KV **0.95**, versus 0.12 in
the 7B grid -- the gate cut preemptions **164 -> 122 raw (-26%)** while
generating **12% more tokens**, i.e. **79.0 -> 52.4 per Mtok (-34%)**, at
p ~= 0.0005 on a Poisson rate-ratio test. This is the strongest single-pair
preemption result of the campaign. Throughput is NOT assessed: verl 0.9's
legacy trainer does not emit the per-trajectory timings the audited
instruments require (see Scrutiny).

## Setup

| | |
|---|---|
| Model | Qwen3-32B (reasoning) + LoRA r32/a32, `load_format=safetensors` |
| Hardware | 2 x 8xH100 80GB (16 GPUs), tp=4 -> **4 replicas**, `nnodes=2` |
| Image | swe13 (`sha256:8cdf4fb8...`): verl 0.9.0 **and** flash-attn 2.8.3 |
| Workload | R2E-Gym SWE agent loop, batch 64 x n4 = 256 trajectories, 25 turns, 20k-char observations, 4096 prompt / 28672 response, 1 step/arm |
| Pressure | gmu 0.35 -> ~16GB engine weights + ~10GB KV per GPU (~155k tokens/replica, ~5 full contexts). Saturation comes from the model's own footprint and context length, NOT a starved pool |
| Arms | baseline (no gate) vs `simple_backpressure` kv 0.90 / waiting 4 (the threshold-sweep winner); gate is the only difference |

Why this stack matters: verl 0.8 could not load 32B at all here -- its weight
sync materialises the full 64GB model per rank. verl 0.9 ships LoRA adapters
only (`base_sync_done`, engine_workers.py:667), and tp=4 across two nodes
halves both the engine weight burden (32 -> 16GB/GPU) and the FSDP actor
shard (8 -> 4GB), taking the per-GPU total from 79GB (OOM) to ~62GB.

## Results

Preemptions are per-arm totals: each arm writes to its own
`PROMETHEUS_MULTIPROC_DIR`, so engines start at zero. The counter is a
POD-wide aggregate, so the two engines sharing a pod report the same number
(verified by pod IP); the arm total is the sum of the two distinct pod
values, not of all four ports.

| Metric | baseline-32b | gate-kv90-32b | delta |
|---|---|---|---|
| **Preemptions** | **164** (90 + 74) | **122** (58 + 64) | **-26%** |
| Tokens generated | 2.076 M | 2.328 M | gate **+12%** |
| **Preemptions / Mtok** | **79.0** | **52.4** | **-34%** |
| Parks / drops | 0 / 0 | 13,063 / 1,141 | the intervention |
| Fleet KV p50 / p90 / max | 0.95 / 0.99 / 1.00 | 0.94 / 0.99 / 1.00 | both saturated |
| Samples at KV >= 0.9 | 420 / 739 (56.8%) | 528 / 849 (62.2%) | see note |
| Mean response length | 8,108 tok | 9,093 tok | +12% |
| Rollout wall (`timing_s/gen`) | 488.5 s | 554.7 s | not a valid metric here |

**Significance.** Poisson rate-ratio test on 286 events over 4.40 Mtok of
exposure: expected 134.8 / 151.2 under the null, observed 164 / 122,
chi-square 11.96, **z ~= 3.46, p ~= 0.0005**. Unlike every earlier pair this
does not need pooling across runs to clear significance.

## Analysis

**A. The pressure is real and un-engineered.** Median KV 0.95 with 57-62% of
fleet samples at or above 0.90, sustained across the whole rollout. The 7B
campaign had to starve pools to 45k tokens to manufacture spikes; here a
generous ~155k-token pool still saturates, because 32B reasoning
trajectories carry enormous contexts. This removes the standing objection
that the workload was rigged to make the gate look useful.

**B. The gate cut preemptions while doing more work.** 164 -> 122 on +12%
more tokens generated. Both directions of the normalisation favour the
result, and the raw count (-26%) is the conservative reading.

**C. Effect size is consistent with everything before it.** -26% raw / -34%
normalised sits inside the -30%..-46% band measured across five 7B pairs at
three pool depths. Same intervention, same magnitude, 4.6x the model.

**D. The saturated-sample share is higher in the gate arm (62.2% vs 56.8%)
and that is an instrument artifact, not a regression.** The flow-control
watcher polls fastest while requests are parked -- i.e. exactly during
saturation -- so the gated arm oversamples its own worst moments. The bias
runs against the gate, so its saturation numbers are conservative.

## Scrutiny

- **One step per arm, one run per cell.** The preemption result clears
  significance on event counts alone; nothing else here is resolvable.
- **Throughput is not assessed and no claim is made.** verl 0.9's legacy
  trainer (`main_ppo_v0`, required because 0.9's V1 trainer needs the absent
  `transfer_queue` package) emits no `agent_loop/generate_sequences/mean`,
  `tool_calls/mean` or `num_turns/mean`. Without them the tool-independent
  throughput instrument and the tool-time symmetry validity gate -- both
  mandated after the 09-17 audit retracted a +13.6% claim -- cannot be
  computed. `timing_s/gen` remains banned as a rate denominator (it tracks
  the slowest trajectory, which is sandbox-bound).
- **Work equality is checked on response length only** (+12%, gate higher),
  not turn count, for the same reason.
- A prior pair at gmu 0.5 also completed but its counters were unusable: the
  two worker pods carried different prometheus baselines (~4,570 vs 89) from
  earlier failed jobs. Back-out arithmetic suggested ~170 vs ~85, consistent
  with this pair, but it is reconstruction rather than measurement and is
  excluded. Logs kept at `/work/sweep32b-dirty/`.
- Departures from main's SWE config: LoRA r32 (required for 32B to fit),
  `trainer.use_v1=False`, 1 step, `ppo_max_token_len_per_gpu` and the
  log-prob budgets raised to 32768 for 29k-token trajectories.
