# verl-native SWE A/B - upstream-scale pair (32B + LoRA, 256-wide)

Store-vs-recompute on verl 0.8.0's native SWE agent loop, 2026-09-10.
Qwen2.5-32B-Instruct + LoRA r32/a32 (fsdp2, dynamic-bsz 16384), **4 GRPO
steps per arm, batch 64 x n 4 = 256 trajectories per rollout** (upstream
PR-62 scale), seed 42, single 8xH200 node, tp=2 (4 engines), **gmu 0.45**,
`free_cache_engine=False` both arms, 32 assistant turns,
`SWE_OBS_MAX_CHARS=20000`, gVisor pool pinned at 21 nodes, dataset baked in
image `swe8` (256 rows). 4 x 64 consumes each task **exactly once**, so no
cross-epoch task repeats and no cross-epoch staleness caveat.

Arms differ by exactly the kv_transfer_config flags; both verified on the
live Hydra command line before running (recompute: 0 store flags; store:
kv_transfer_config + sha256_cbor). Mooncake master verified empty at pair
start; the recompute arm writes nothing, so the store arm began on a clean
tier.

## Validity (all recorded, all pass)

| gate | recompute | store |
|---|---|---|
| steps completed / rc | 4/4, rc=0 | 4/4, rc=0 |
| wall-clock audit (sum step times vs wall) | 5,909s vs 6,201s (292s init) | 5,637s vs 5,909s (272s init) |
| actor/entropy range | 0.188-0.202 | 0.198-0.775 (see DRIFT below) |
| num_turns/mean range | 46.2-48.8 | 45.9-48.1 |
| rollout-error / OOM / EngineDeadError | 0 / 0 / 0 | 0 / 0 / 0 |
| by_source identity | EXACT | EXACT |
| PRESSURE: num_preemptions_total | 57 | 30 |
| PRESSURE: kv_cache_usage_perc max | 0.9952 | 0.9954 |
| connector: force-fails / errors / dropped signals | n/a | 0 / 0 / 0 |

## Serving split (final cumulative counters, 4 engines, identity EXACT)

| source | recompute | share | store | share |
|---|---|---|---|---|
| local_compute (true prefill) | 58,065,234 | 38.26% | **14,619,750** | **9.76%** |
| local_cache_hit | 93,707,616 | 61.74% | 119,627,440 | 79.86% |
| external_kv_transfer | 0 | 0% | 15,545,040 | 10.38% |
| TOTAL (= prompt_tokens_total) | 151,772,850 | | 149,792,230 | |

Workloads matched to 1.3% on total prompt tokens. **True prefill compute:
58.1M -> 14.6M tokens = 3.97x reduction** - the headline result, and the
one that does not depend on timing noise.

Note the second-order effect: the store arm's LOCAL cache-hit share is also
higher (79.9% vs 61.7%) and its per-engine prefix-hit rates run 29-67% vs
21-27%. Restored KV is cached locally on arrival, so the external tier
raises local hit rate rather than competing with it. The store arm also
took **half the preemptions** (30 vs 57): recomputing less means holding
fewer blocks for less time.

Mooncake ops (store arm, zero failed keys): save_put 62,365 ops @ 12.9ms
mean, load_get 4,258 ops @ 100.0ms mean. Master evicted 36 times / 1.89TB
over the run (tier turns over at this scale) with no failures.

## Per-step results (4 steps; deltas store vs recompute)

| verl metric | recompute | store | delta | store lower |
|---|---|---|---|---|
| timing_s/agent_loop/generate_sequences/mean | 197.22s | 82.45s | **-58.2%** | 3/4 |
| timing_s/agent_loop/generate_sequences/max | 669.80s | 439.93s | -34.3% | 4/4 |
| timing_s/gen | 1177.37s | 1111.65s | -5.6% | 2/4 |
| timing_s/step | 1477.37s | 1409.26s | -4.6% | 2/4 |
| timing_s/agent_loop/tool_calls/mean | 131.75s | 138.58s | +5.2% | 2/4 |
| response_length/mean | 11,067 | 10,999 | -0.6% | 2/4 |
| num_turns/mean | 47.59 | 47.13 | -1.0% | 4/4 |
| critic/score/mean | 0.0508 | 0.0430 | -4.3% | 2/4 |

NOTE (read the -58.2% per step, not as a mean): step N draws the same 64
tasks in both arms (seed 42), so per-step pairs are apples-to-apples, but
task difficulty varies a lot across steps because each task appears once.

| step | rc generate_sequences/mean | store | ratio |
|---|---|---|---|
| 1 | 326.4s | 104.6s | 3.1x |
| 2 | 325.0s | 100.5s | 3.2x |
| 3 | 79.1s | 58.3s | 1.36x |
| 4 | 58.5s | 66.5s | 0.88x (store slower) |

The store's advantage tracks how loaded the step is: 3x on the two heavy
steps, ~par on the two light ones. That is the pressure-gated payoff the
earlier pairs predicted, now visible WITHIN one pair. It also means the
arithmetic mean overstates the typical case - report the per-step table.

Rollout wall (timing_s/gen) moves only -5.6% because makespan remains
sandbox-bound: per-step it equals the slowest trajectory's tool time to
within a few seconds, as established in `../pressure-pair/`.

## DRIFT: the store arm's entropy climbs; recompute's does not

| step | recompute entropy | store entropy | rc ppo_kl | store ppo_kl |
|---|---|---|---|---|
| 1 | 0.188 | 0.198 | -1.81e-05 | +3.53e-05 |
| 2 | 0.201 | 0.265 | -8.66e-06 | +2.96e-05 |
| 3 | 0.202 | **0.390** | -4.41e-05 | +3.60e-05 |
| 4 | 0.197 | **0.775** | -4.96e-05 | +1.50e-05 |

Recompute is flat across all four steps; the store arm rises monotonically
to ~3.9x its own step-1 value. `actor/ppo_kl` also flips sign consistently
(recompute negative, store positive) - tiny in magnitude (1e-5) but
systematic, and ppo_kl measures exactly the rollout-vs-training policy
divergence a stale-KV effect would produce.

Mechanism (documented in `../../integration/verl/swe/PINS.md` before this
run, now MEASURED): store keys contain no weight version. With LoRA the
merged rollout weights change every step, so KV written in steps 1-3 is
served back in step 4 under different weights. The step-1-self-baseline
tripwire agreed in the design phase is what this table implements.

Scope of the claim: 0.775 still clears the <1.0 validity gate and the
workload metrics (turns, response length, score) stay comparable, so the
pair is VALID. But N=4 with one pair cannot separate stale-KV from other
causes with certainty - what is recorded is (a) a flat recompute control on
identical tasks, (b) a monotonic store-only rise, (c) a consistent ppo_kl
sign flip, and (d) a mechanism that predicts exactly this. Extrapolating
the trend, entropy would breach the gate by roughly step 5-6. **A
longer-horizon store run needs per-step flush or weight-versioned keys
before it can be trusted for convergence work.** This is the first recorded
cost of the store in this program.

## Connector fix stack (this pair is the first run on it)

The prior regime could not complete a single step. Four ways the connector
dropped the block-release signal (and one sizing law) were found and fixed;
full fault tree in `../../integration/verl/CONNECTOR-INVESTIGATION.md`,
patch in `connector_v3.py` (19 unit tests run at image build time).

| symptom before | after (this pair) |
|---|---|
| silent wedges: engine pinned ~98% KV, 0 running, frozen hours | 0 occurrences |
| engine killed by `assert req_id in self.requests` | 0 |
| loads stranded by promotion starvation (vLLM only promotes on non-preempting steps) | 4,429 watchdog promotions, 0 force-fails |
| 1 key / 2MB per save put (~3.9ms RPC vs ~40us wire on RDMA) | 62,365 puts carrying ~21 keys each |
| admission starvation 71-77% of pressured samples | 0% |

**Sizing law (the actual blocker):** at gmu 0.317 the KV pool was 2,125
blocks = 34k tokens while a trajectory context reaches 28k, so one request
parked on an async pull held ~82% of the pool and all four engines
deadlocked; no in-flight cap can fix a pool below one working set. gmu 0.45
gives 11,645 blocks = 186k tokens (~6-7 contexts) and the deadlock
disappears while preemption pressure remains (KV peaks 0.995, 30-57
preemptions).

## Measurement caveats

- **Regime differs from `../pressure-pair/`** (gmu 0.317, 12x64, 25 turns,
  6k obs cap). Cross-pair deltas are not directly comparable: this pair is
  4x wider per rollout, has a 5.5x larger KV pool, longer turns and fatter
  observations (response ~11.1k vs ~7.5k). Compare mechanisms, not numbers.
- 4 steps gives N-of-4 consistency counts; weaker than the prior pair's
  N-of-12, but each step is a 256-trajectory sample (4x larger).
- The store arm ran immediately after recompute on the same node with the
  same warmed sandbox pool; tool-time symmetry is +5.2% at 2/4.
- Sandboxes have no network egress; `pip install` attempts burn the 60s
  command timeout in both arms and drive makespan variance.
- verl's `agent_loop/*/num_preempted` is -1 on this stack; preemption
  counts come from engine `/metrics`.
- Engine scrape probes the mooncake RDMA handshake listener, producing
  benign `SocketHandShakePlugin: malformed json` noise in both arms.

## Files

- `recompute_driver.log.gz`, `store_driver.log.gz` - full driver logs
  (verl per-step metric lines; source of the A/B and drift tables).
- `engine_scrapes.log.gz` - per-minute in-worker engine /metrics samples,
  SNAP-timestamped; arm windows recompute [1789036097,1789042298], store
  [1789042298,1789048207].
- `sidecar_final.log.gz` - netns-fixed sidecar archive tail (mooncake
  op-time histograms).
- `master_postrun.txt.gz` - mooncake master counters after the store arm.
- `connector_v3.py` - the exact connector patch both arms' image carried.
