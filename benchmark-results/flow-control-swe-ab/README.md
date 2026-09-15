# Flow control A/B on verl's SWE agent loop

`simple_backpressure` admission gating vs no gating, on verl 0.8.0's native SWE
agent loop. Both arms install the scheduler hook, so routing policy is held
constant and the gate is the only variable.

## TLDR

The gate does exactly what it is designed to do and it does not make sampling
faster. Engine-side saturation was eliminated outright — zero routing decisions
saw an engine at or above the threshold, against 54 in the baseline, and the
deepest engine queue fell from 47 to 3 — while total generation time moved
0.1%, which is far inside the step-to-step noise.

## Setup

| | |
|---|---|
| Workload | `run_swe.sh` (main), R2E-Gym SWE agent loop, batch 64 x n8, 4096-token prompts, 28672-token responses, 32 turns |
| Model | Qwen2.5-7B-Instruct, tp=2, 4 engines on 8xH100 |
| Pressure | `gpu_memory_utilization=0.22` (only departure from `run_swe.sh`) |
| Arms | `fcoff` = hook, no `flow_control`; `fcon` = hook + `simple_backpressure` (kv 0.90 / waiting 4) |
| Steps | 2 per arm, 64 tasks reused each step so both arms see identical work |
| Dataset | 64 R2E-Gym tasks, images prewarmed on all sandbox nodes before either arm |

Pressure was verified before the pair ran: the baseline saturates at kv 0.9999
with 75 concurrent requests per engine, against kv 0.375 at `run_swe.sh`'s
default gmu 0.5 — at the default the gate could never have fired.

## Results

| Metric | fcoff (baseline) | fcon (gate) | Delta |
|---|---|---|---|
| Generation time, step 1 | 855.2 s | 1084.7 s | +26.8% |
| Generation time, step 2 | 1132.6 s | 905.7 s | -20.0% |
| **Generation time, total** | **1987.8 s** | **1990.4 s** | **+0.1%** |
| Peak KV utilization | 0.9999 | 0.8994 | capped at threshold |
| Decisions seeing kv >= 0.90 | 54 / 282 | **0 / 281** | eliminated |
| Deepest engine queue | 47 | **3** | -94% |
| Peak concurrent reqs/engine | 75 | 51 | -32% |
| Requests parked by the gate | 0 | 0 | see below |
| Response length / turns (mean) | 8078 / 42.7, 8871 / 42.1 | 8828 / 43.3, 9105 / 44.6 | slightly more work done |

## Analysis

**A. The gate held the engines exactly at its threshold.** Peak KV of 0.8994
against a 0.90 setting, and not one of 281 routing decisions observed an engine
at or above it. The baseline crossed that line 54 times and ran its deepest
queue 15x deeper.

**B. It achieved that without ever parking a request.** Both arms report zero
parks. With four engines the filter half of the gate always found an
unsaturated engine to steer to, so admission never had to block — the
gate-plus-filter semantics behaving as specified, with queueing as the unused
fallback. A single-engine fleet, or a threshold low enough to shut all four at
once, is what would exercise the parking path.

**C. No wall-time effect.** Totals differ by 0.1%, and the per-step numbers
disagree in sign (+26.8% then -20.0%), so step-to-step variance dominates at
n=2. Nothing here supports a throughput claim in either direction.

**D. The gated arm did marginally more work in the same time** — longer
responses and more turns per trajectory — so per-token throughput is slightly
favourable, but well inside noise and not a result.

**E. Preemption counts are unavailable.** verl reports `num_preempted: -1` on
this stack, so engine-side preemption has to come from scraping engine
`/metrics`, which this pair did not collect. KV utilization and queue depth are
the pressure evidence here.

## Scrutiny

- n=2 steps per arm. Per-step generation time varies by ~25%, so only the
  aggregate is meaningful and even it cannot resolve small effects.
- Sandbox/tool time dominates SWE rollout wall time (92-97% in the store A/B on
  the same harness), which bounds how much any LLM-side change can move the
  total.
- The baseline needed 2 attempts (first died in startup); the retry is recorded
  in `progress.txt`. Both counted arms ran to completion on the same cluster,
  same node, same warm image cache.
