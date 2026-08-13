# Vanilla 32B smoke (arm-A preview): matched A-vs-B throughput, 2026-08-13

RayJob `rllm-32b-vanilla-smoke` SUCCEEDED (branch d933411). Byte-identical
config to `32b-policy-smoke` minus the routing_policy override: Qwen3-32B +
LoRA r32, TP=2 x 4 replicas, one 8xH200 spot node, same 8 tasks/seed/image
lineage. Full metrics: submitter.log.

## Single-step comparison (n=1 each - indicative, not final)

| metric | A: vanilla (StickyLeastLoaded) | B: scheduler (pre-champion profile) | delta |
|---|---|---|---|
| training throughput (tok/s) | 98.5 | 118.5 | +20% |
| step wall-clock (s) | 2060 | 1747 | -15% |
| rollout phase wall (s) | 168.4 | 160.7 | -5% |
| LLM wall in rollout (s) | 97.5 | 91.2 | -6% |
| tokens in step | 1.62M | 1.66M | ~equal |
| MFU | 0.359 | 0.371 | +3% rel |
| weight sync (s) | 6.14 | 6.19 | equal |

## Read

- **No scheduler overhead penalty visible** - the policy arm was faster on
  every timing metric. The rollout-phase delta (-5%) is the routing-relevant
  signal; the larger step-time delta also folds in training-phase variance
  from stochastically different trajectories (token counts differ ~2%).
- n=1 per arm on a spot node: treat direction as meaningful (scheduler >=
  vanilla), magnitude as noise-bounded. The 30-50-step arms produce the
  real curves.
- B above ran the pre-champion profile (sticky/queue/kv only). A rerun on
  the FULL champion profile (prefix_cache + saturation, image 5a314c6e)
  launched immediately after this run on the same warm node.
