# verl-native SWE A/B - gVisor + no-sleep regime (FIRST VALID PAIR)

Store-vs-recompute on verl 0.8.0's native agent loop, 2026-09-09.
Qwen2.5-7B-Instruct, 12 GRPO steps per arm, batch 16 x n 4 (64 trajectories
per rollout), seed 42, single 8xH200 node (both arms on the same worker pod),
tp=2 (4 engines), gmu 0.30, `free_cache_engine=False` BOTH arms (sleep mode
corrupts engines under the connector's RDMA registration - see
`../entropy-diagnosis/`), sandboxes as agent-sandbox CRs under gVisor in
`agents-system`, dataset baked into image `swe2` (256 train rows), 1TiB
store (8 x 128gb segments), arms differ by exactly the kv_transfer_config
flags. Gated by a 2-step store smoke requiring entropy < 1.0, >=5 turns,
and non-zero tool time per step.

## Validity (all recorded, all pass)

| gate | recompute | store |
|---|---|---|
| steps completed / rc | 12/12, rc=0 | 12/12, rc=0 |
| actor/entropy range | 0.236-0.305 | 0.251-0.303 |
| num_turns/mean range | 40.8-45.7 | 39.9-44.9 |
| rollout-error rewards | 0 | 0 |
| OOM / NCCL / ENGINE_DEAD | 0 | 0 |
| by_source identity | n/a | sums to prompt_tokens_total EXACTLY per engine |

## Results (per-step means over 12 steps; deltas store vs recompute)

| metric | recompute | store | delta | store better |
|---|---|---|---|---|
| timing_s/gen (rollout wall) | 759.4s | 730.2s | -3.8% | 8/12 |
| timing_s/step | 779.1s | 749.0s | -3.9% | 8/12 |
| traj LLM time mean | 11.91s | 11.21s | -5.9% | 10/12 |
| traj LLM time max (straggler) | 61.8s | 48.9s | -20.9% | 7/12 |
| traj tool time mean | 195.8s | 198.7s | +1.4% | 7/12 |
| response_length/mean | 7839 | 7360 | -6.1% | - |

Tool time is store-independent and lands at +1.4% with 7/12 - the pairing
symmetry check the invalid plain-pod pair failed (-35%, 12/12).

## Store-arm serving split (sidecar scrape, final counters, 4 engines)

| source | tokens | share |
|---|---|---|
| local_cache_hit | 54,657,968 | 93.77% |
| local_compute | 3,503,664 | 6.01% |
| external_kv_transfer | 129,360 | 0.22% |
| TOTAL (= prompt_tokens_total) | 58,290,992 | |

## Interpretation (S5)

The pair is valid; the REGIME is low-pressure for the store. Each
trajectory's turns stay on one engine, and at gmu 0.30 the local prefix
cache absorbs ~94% of all prompt tokens, so the external tier is exercised
for only 0.22% of tokens - about the analytic cross-step bound (fresh tasks
per step; only shared prefixes survive the per-step cache reset). Under
that load the store is measured HARMLESS-TO-MODESTLY-POSITIVE: every
LLM-side metric favors it (most consistently per-trajectory LLM time,
10/12), and quality metrics are indistinguishable. This regime does NOT
demonstrate the store's headline value; that requires eviction pressure
(smaller KV pool, larger batches) or cross-engine reuse (multi-node,
engine restarts), which is the next experiment axis.

## Measurement caveats

- Sandboxes have no network egress (recorded in-pod: DNS fails); model
  `pip install` attempts burn the 60s command timeout, inflating tool time
  in both arms equally. Straggler sandboxes measured idle (2-73 mCPU) -
  tool time is timeout-bound, not CPU-bound.
- The worker-local by_source snapshotter died with a pod replacement;
  the (netns-fixed) sidecar is the recorded source, and its log rotates -
  final cumulative counters are present, per-minute deltas only partially.
- Step-level by_source is not recorded, only run-level finals.
