# Overscheduling v0 vs stock verl on a multi-turn agentic RL rollout: smaller sampler pools, more work per GPU

Runs: 2026-10-09, 05:11–05:34 UTC. Code: `feat/overscheduling-v0` at `4fd565c`.

## TLDR

v0 lets a smaller sampler pool carry the same rollout. Its gate keeps each sampler under 94% KV, and Mooncake reloads KV evicted during tool waits instead of recomputing it.

- **v0 on 3 samplers:** 90% of stock's throughput on 75% of the GPUs. **+20% samples/s/GPU.**
- **v0 on 2 samplers:** 71% of stock's throughput on 50% of the GPUs. **+42% samples/s/GPU.**
- No preemptions in any arm.

Why it matters: agentic rollouts wait on tools, so a pool sized for peak load sits partly idle. More work per GPU frees GPUs for training or for more rollouts.

## Purpose

- RL rollouts with tool calls leave samplers idle while trajectories wait.
- Teams size the sampler pool for peak load and keep it.
- Question: with v0, can a smaller pool keep up?
- Baseline: stock verl 0.9.1 on 4 samplers. This is what verl users run today.

## How each arm decides

- **Stock verl (4 samplers):** verl 0.9.1's built-in load balancer routes every turn. No KV store.
- **v0 (3 or 2 samplers):** our gate admits a turn only where its KV fits under 94%; scorers keep each trajectory on its engine; Mooncake reloads evicted KV from host memory on both nodes.

Our scheduler profile, verbatim (`integration/verl/overscheduling/scheduler.yaml` at `4fd565c`):

```yaml
profile_handler:
  type: single_profile
profiles:
  overscheduling:
    flow_control:
      type: kv_saturation
      # The load runs fixed-length turns, so a first turn's output is known: keep in step with --output-tokens.
      default_osl: 256
      # Sticky routing places a first-turn burst unevenly, faster than engine metrics move; this cap
      # holds the busiest engine below preemption, allowing for the ledger counting tokens where
      # vLLM allocates whole blocks.
      budget_fraction: 0.94
    scorers:
      # Affinity first: each trajectory prefers the engine holding its prefix. When kv_saturation
      # finds that engine full, the backpressure scorers below pick among the engines with room.
      - type: sticky_session
        header_name: x-rls-session-id
        weight: 4.0
      - type: prefix_cache
        weight: 4.0
      - type: least_queue
        weight: 1.0
      - type: waiting_queue
        weight: 1.0
      - type: kv_cache
        weight: 1.0
    picker:
      type: max_score
```

## Methodology

### 1. Hardware and software

| Item | Value |
|---|---|
| Nodes | 2 × GKE `a3-ultragpu-8g`, 8 × NVIDIA H200 141 GB each, RDMA between nodes |
| Model | Qwen2.5-32B-Instruct |
| Sampler | vLLM 0.29.0, tensor parallel 4 (4 GPUs per sampler) |
| KV per sampler | `gpu_memory_utilization` 0.3: 25,120 blocks × 16 tokens = 401,920 tokens |
| RL framework | verl 0.9.1, rollout only (no training) |
| KV store (v0 only) | Mooncake 0.3.13.post1 over RDMA, 128 GB host memory per GPU process |
| Scheduler (v0 only) | py-inference-scheduler at `4fd565c` |

### 2. Workload

Each trajectory is a 6-turn agentic loop. The model writes 256 tokens, a tool runs, and its 128-token reply is appended. Tool time is random per turn but seeded, so every arm waits exactly the same. Prompts are unique.

| Dimension | Value |
|---|---|
| Trajectories per rollout | 192 |
| Turns per trajectory | 6 |
| Prompt | 6,000 words (6,034 tokens with the chat template), unique per trajectory |
| Model output per turn | 256 tokens, fixed (EOS ignored) |
| Tool reply per turn | 128 tokens |
| Tool time per turn | Lognormal, mean 3 s, σ 0.5, seeded per trajectory and turn |
| Peak context per trajectory | 6,034 + 5 × (256 + 128) + 256 = 8,210 tokens (514 KV blocks) |
| Repeats | 3 rollouts per arm, seeds 0, 1, 2, same in every arm |
| Warm-up | 1 unmeasured rollout per arm |
| Between rollouts | vLLM prefix cache and Mooncake store cleared |

### 3. Integration

| Arm | Samplers (per node) | Routing | KV store |
|---|---|---|---|
| Stock verl | 4 (2 + 2) | verl's built-in balancer | None |
| v0, 3 samplers | 3 (2 + 1) | verl hook to our scheduler, profile above | Mooncake via `RLPullPolicyConnector`, `save_decode_cache=false` |
| v0, 2 samplers | 2 (1 + 1) | Same as above | Same as above |

### 4. Metrics

| Metric | Definition | Better |
|---|---|---|
| Rollout time | Wall time for verl to finish all 192 trajectories | Lower |
| Samples/s/GPU | Trajectories ÷ rollout time ÷ sampler GPUs. v0 on 3: 192 ÷ 59.1 s ÷ 12 = 0.271 | Higher |
| Throughput vs stock | Stock rollout time ÷ arm rollout time. v0 on 3: 53.4 ÷ 59.1 = 90% | Higher |
| KV mean / peak | vLLM KV cache usage, scraped every 2 s, averaged over samplers / highest value | Mean: higher. Peak: below 1.0 |
| Prompt KV from store | Share of prompt tokens vLLM loaded from Mooncake | Context |
| Prompt tokens computed | Share of prompt tokens vLLM computed instead of reusing | Lower |
| Preemptions | Requests vLLM preempted | Lower |

## Results

Means of 3 rollouts. Percentages are relative to stock verl on 4 samplers.

| Arm | GPUs | Rollout time | Samples/s/GPU | Throughput vs stock | KV mean / peak | Prompt KV from store | Prompt tokens computed | Preemptions |
|---|---|---|---|---|---|---|---|---|
| Stock verl, 4 samplers | 16 | 53.4 s | 0.225 | 100% | 0.35 / 0.74 | 0% | 15.9% | 0 |
| v0, 3 samplers | 12 | 59.1 s (+11%) | **0.271 (+20%)** | 90% | 0.52 / 0.93 | 38.2% | 17.4% | 0 |
| v0, 2 samplers | 8 | 74.9 s (+40%) | **0.321 (+42%)** | 71% | 0.75 / 0.93 | 78.7% | 18.9% | 0 |

Per rollout:

| Seed | Stock verl, 4 samplers | v0, 3 samplers | v0, 2 samplers |
|---|---|---|---|
| 0 | 56.2 s, 0.214 | 61.7 s, 0.259 (+21.5%) | 75.7 s, 0.317 (+48.5%) |
| 1 | 53.3 s, 0.225 | 58.6 s, 0.273 (+21.2%) | 75.6 s, 0.318 (+40.9%) |
| 2 | 50.6 s, 0.237 | 57.1 s, 0.280 (+18.1%) | 73.3 s, 0.327 (+37.9%) |

Cells: rollout time, samples/s/GPU (change vs stock on the same seed).

## Run logs

Each log holds the per-rollout results, every routing decision (v0 arms), and engine counters scraped every 2 s.

| Arm | Ray job | verl log |
|---|---|---|
| Stock verl, 4 samplers | `os-final-n192-stock4` | `<verl_stock_4samplers_n192.log>` |
| v0, 3 samplers | `os-final-n192-v0x3` | `<verl_v0_3samplers_n192.log>` |
| v0, 2 samplers | `os-final-n192-v0x2` | `<verl_v0_2samplers_n192.log>` |

## Analysis

**(a) v0 on 3 samplers does 20% more work per GPU, at 90% of the throughput.**
- Stock's 4 samplers sit partly idle during tool waits: KV mean 0.35.
- v0 packs the same work into 3 samplers: KV mean 0.52.
- The store reloads 38% of prompt KV, evicted while trajectories waited on tools.

**(b) v0 on 2 samplers does 42% more work per GPU, at 71% of the throughput.**
- Two samplers hold only about half of the waiting trajectories' contexts.
- The store reloads 79% of prompt KV.
- Computed prompt tokens stay at 18.9%, close to stock's 15.9%. Stock's figure is the floor: first-turn prompts plus tool replies.
- About half the reloaded KV comes from the other node, estimated from a sample of loads.

**(c) The cost is rollout time.**
- +11% on 3 samplers, +40% on 2.
- "Similar throughput" holds for 3 samplers.
- 2 samplers trade more time for more work per GPU.

**(d) No preemptions in any arm.**
- v0's KV peaks at 0.93, under the 0.94 cap.

**(e) Results hold across seeds.**
- v0 on 3: +18% to +22% per GPU.
- v0 on 2: +38% to +49% per GPU.

**(f) Limits.**
- Per-GPU numbers exclude the store's host memory: 128 GB per GPU process, 1.5 TB for v0 on 3 and 1 TB for v0 on 2.
- `gpu_memory_utilization` 0.3 keeps KV scarce on purpose. Real deployments need longer contexts or more trajectories to see the same pressure.
- Tool time is synthetic: lognormal, mean 3 s.
