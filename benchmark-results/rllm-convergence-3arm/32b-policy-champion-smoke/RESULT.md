# 32B champion-profile smoke (arm-B final shape), 2026-08-13

Same config as the other 32B smokes; scheduler with the FULL champion
profile (saturation + prefix_cache 4.0 + waiting 1.0 + kv 0.5 + sticky 4.0),
image 5a314c6e with the body-threading gateway. PrefixCacheScorer active in
1,058 decisions. Full metrics: submitter.log.

## Three-run comparison (single step each, n=1 - directional only)

| metric | A: vanilla | B: old profile | B: champion profile |
|---|---|---|---|
| rollout phase wall (s) | 168.4 | 160.7 | **136.0** |
| LLM wall in rollout (s) | 97.5 | 91.2 | **70.2** |
| step wall-clock (s) | 2060 | 1747 | 2511 |
| tokens in step | 1.62M | 1.66M | 1.40M |
| training throughput (tok/s) | 98.5 | 118.5 | 69.8 |
| MFU | 0.359 | 0.371 | 0.371 |

## Read (which numbers to trust)

- **Rollout/inference phase is the scheduler-attributable signal**, and it
  improves monotonically: LLM wall -28% vs vanilla with the champion
  profile - consistent with prefix affinity raising vLLM prefix-cache hits
  across the 4 replicas. This is the metric class our layer can influence.
- **Step-level tok/s is NOT comparable across these runs**: trajectory
  lengths are stochastic (1.40-1.66M tokens per batch) and the
  training/update phase dominates step time, so per-step throughput mixes
  batch composition into the denominator. The champion run's lower tok/s
  reflects a smaller batch, not slower serving - its rollout phase was the
  fastest of the three.
- n=1 per configuration on a spot node. The 30-50-step arms with fixed data
  order are the real measurement; these smokes establish direction (no
  overhead penalty; prefix affinity helps the phase it touches) and that
  all three configurations run.
