# Overscheduling v0 vs stock verl on an agentic RL rollout

Runs: 2026-10-09, 05:11–05:34 UTC. Code: `feat/overscheduling-v0` at `4fd565c`.

## TLDR

Overscheduling v0, using cross-node KV transfer, prefill-based flow control and KV offload, results in **20% better samples/sec/GPU** when running on 3 samplers instead of 4, and **42% better samples/sec/GPU** when running on 2 samplers instead of 4.

## Problem

- Agentic RL rollouts alternate model turns with tool calls.
- While a trajectory waits on a tool, its sampler has less to do.
- Teams size the sampler pool for peak load, so GPUs sit partly idle.
- Shrinking the pool is risky. The waiting trajectories' KV no longer fits in GPU memory. It gets evicted and recomputed, or requests get preempted.

## Overscheduling v0

v0 lets a smaller sampler pool carry the same rollout. It has four parts.

| Part | What it does |
|---|---|
| Prefill-based flow control | A gate admits a turn only to a sampler whose KV can hold the turn's prompt plus the trajectory's last output length. A first turn uses a configured length instead (256 tokens here). Each sampler is capped at 94% of its KV (377,804 of 401,920 tokens), so vLLM never needs to preempt. |
| KV offload | Mooncake saves each turn's prompt KV to host memory. When a waiting trajectory's KV is evicted from GPU memory, its next turn reloads it instead of recomputing it. |
| Cross-node KV transfer | Mooncake spreads that host memory across both nodes over RDMA. Any sampler can reload KV saved on either node. |
| Affinity routing | Scorers keep each trajectory on the sampler that holds its prefix. When that sampler is full, backpressure scorers pick the least busy sampler with room. |

## Methodology

### 1. Hardware and software

| Item | Value |
|---|---|
| Nodes | 2 × GKE `a3-ultragpu-8g`, 8 × NVIDIA H200 141 GB each, RDMA between nodes |
| Model | Qwen2.5-32B-Instruct |
| Sampler | vLLM 0.29.0, tensor parallel 4 (4 GPUs per sampler) |
| KV per sampler | 401,920 tokens (`gpu_memory_utilization` 0.3) |
| RL framework | verl 0.9.1, rollout only (no training) |
| KV store (v0 only) | Mooncake 0.3.13.post1 over RDMA, 128 GB host memory per GPU process |
| Scheduler (v0 only) | py-inference-scheduler at `4fd565c` |

### 2. Workload

Each trajectory is a 6-turn agentic loop. The model writes 256 tokens, a tool runs, and its 128-token reply is appended. Tool time is random per turn but fixed in advance, so every arm waits exactly the same. Prompts are unique.

| Dimension | Value |
|---|---|
| Trajectories per rollout | 192 |
| Turns per trajectory | 6 |
| Prompt | 6,034 tokens, unique per trajectory |
| Model output per turn | 256 tokens, fixed (EOS ignored) |
| Tool reply per turn | 128 tokens |
| Tool time per turn | Lognormal, mean 3 s, σ 0.5, fixed per trajectory and turn |
| Peak context per trajectory | 6,034 + 5 × (256 + 128) + 256 = 8,210 tokens |
| Repeats | 3 rollouts per arm. Rollouts 1, 2 and 3 use the same prompts and tool times in every arm. |
| Warm-up | 1 unmeasured rollout per arm |
| Between rollouts | vLLM prefix cache and Mooncake store cleared |

### 3. Arms

| Arm | Samplers (per node) | Routing | KV store |
|---|---|---|---|
| Stock verl | 4 (2 + 2) | verl 0.9.1's built-in load balancer | None |
| v0, 3 samplers | 3 (2 + 1) | Our scheduler through the verl hook. A gate caps each sampler at 94% of its KV. Routing uses sticky session and prefix cache scoring, followed by backpressure scoring. | Mooncake, saving prompt KV each turn |
| v0, 2 samplers | 2 (1 + 1) | Same as v0 on 3 samplers | Same as v0 on 3 samplers |

## Results

Means of 3 rollouts. Percentages are relative to stock verl on 4 samplers. Per-rollout numbers are in the verl logs.

| Arm | GPUs | Rollout time | Samples/s/GPU | Throughput | KV mean / peak | Preemptions |
|---|---|---|---|---|---|---|
| Stock verl, 4 samplers | 16 | 53.4 s | 0.225 | 3.61 samples/s | 0.35 / 0.74 | 0 |
| v0, 3 samplers | 12 | 59.1 s (+11%) | **0.271 (+20%)** | 3.25 samples/s (−10%) | 0.52 / 0.93 | 0 |
| v0, 2 samplers | 8 | 74.9 s (+40%) | **0.321 (+42%)** | 2.57 samples/s (−29%) | 0.75 / 0.93 | 0 |

> [!NOTE]
> - **Rollout time:** wall time for verl to finish all 192 trajectories.
> - **Samples/s/GPU:** trajectories ÷ rollout time ÷ sampler GPUs. For v0 on 3 samplers: 192 ÷ 59.1 s ÷ 12 = 0.271.
> - **Throughput:** trajectories ÷ rollout time. For v0 on 3 samplers: 192 ÷ 59.1 s = 3.25 samples/s.
> - **KV mean / peak:** vLLM's KV cache usage, sampled every 2 s. Mean is averaged over samplers. Peak is the highest sample.
> - **Preemptions:** requests vLLM preempted.

Where prompt KV came from, per trajectory:

| Arm | From GPU cache | From Mooncake | Computed |
|---|---|---|---|
| Stock verl, 4 samplers | 35,296 tokens (84.1%) | 0 | 6,671 tokens (15.9%) |
| v0, 3 samplers | 18,628 tokens (44.4%) | 16,026 tokens (38.2%) | 7,313 tokens (17.4%) |
| v0, 2 samplers | 1,032 tokens (2.5%) | 33,013 tokens (78.7%) | 7,921 tokens (18.9%) |

Every turn's prompt is the conversation so far. vLLM gets its KV from the sampler's GPU cache, loads it from Mooncake, or computes it.

Run logs:

| Arm | verl log |
|---|---|
| Stock verl, 4 samplers | `<verl_stock_4samplers_n192.log>` |
| v0, 3 samplers | `<verl_v0_3samplers_n192.log>` |
| v0, 2 samplers | `<verl_v0_2samplers_n192.log>` |

## Analysis

**(a) Samples/s/GPU rises because rollout time grows less than the GPU count falls.**
- 3 samplers use 25% fewer GPUs, which is worth +33% per GPU. The rollout runs 11% longer, which takes back 13 points. Net: +20% (16/12 × 53.4/59.1 = 1.20).
- 2 samplers use half the GPUs, which is worth +100% per GPU. The rollout runs 40% longer, which takes back 58 points. Net: +42% (16/8 × 53.4/74.9 = 1.43).

**(b) Tool waits are the same in every arm, so a smaller pool only stretches the model time.**
- In each rollout, the slowest trajectory spends 24–31 s waiting on tools. That part does not change with pool size.
- The rest is model time: about 26 s on stock, 32 s on v0 with 3 samplers, and 48 s on v0 with 2.
- Each v0 sampler serves 33% more trajectories with 3 samplers and 100% more with 2. Model time grows only 22% and 82%, because fuller samplers do more work per step. KV mean rises from 0.35 on stock to 0.52 and 0.75.

**(c) KV offload keeps the smaller pools from recomputing.**
- At peak, the 192 trajectories need 1.58M tokens of KV (192 × 8,210). Two samplers hold 0.80M (2 × 401,920), so about half of the waiting trajectories' KV cannot stay on GPU.
- On 2 samplers, Mooncake supplies 78.7% of prompt KV. The GPU cache supplies only 2.5%.
- v0 on 2 samplers computes 18.9% of prompt KV, close to stock's 15.9%.
- Without the store, that 78.7% would be computed again.

**(d) Cross-node transfer carries about half of those reloads.**
- Mooncake spreads saved KV across both nodes' host memory.
- 44% (3 samplers) and 50% (2 samplers) of reloaded KV comes from the other node. These shares are estimated from a sample of loads.

**(e) Prefill-based flow control prevents preemptions.**
- The gate caps each sampler at 94% of its KV. KV peaked at 0.93.
- No arm preempted a single request.

**(f) Limits.**
- Per-GPU numbers exclude the store's host memory: 128 GB per GPU process.
- `gpu_memory_utilization` 0.3 keeps KV scarce on purpose.
- Tool time is synthetic: lognormal, mean 3 s.
