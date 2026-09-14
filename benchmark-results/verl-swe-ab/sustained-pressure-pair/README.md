# Shared KV tier vs local-only recompute — SWE-bench agent RL, Qwen2.5-32B

## TLDR

Adding a shared RDMA KV tier to an agentic RL rollout cut **sampling time per
trajectory by 35%**. The slowest trajectory in each rollout, which is the one a
step actually waits on, improved by 23%.

| | local-only | shared KV tier | change |
|---|---|---|---|
| sampling time per trajectory | 348.2 s | 226.2 s | **-35.1%** |
| sampling time, slowest trajectory | 1,146.7 s | 883.8 s | **-22.9%** |
| prompt tokens recomputed on GPU | 148.6M | 67.0M | **2.22x less** |

---

## What we tested

We theorize that an external KV tier that HBM can offload to is better than always recomputing requests. This is most useful when the local prefix cache
is not enough to save on prefill. We have seen this to be a normal condition for
long-context agentic workloads. They present a large, additive context on
every turn, and they run enough trajectories at once that the HBM memory pool cannot hold them all.

The goal is to see whether enabling KV offload (through Mooncake) improves sampling performance for long-context, agentic, heavily preemptive RL workloads.

---

## Setup

### Stack

| | |
|---|---|
| nodes | 1 |
| machine type | a3-ultragpu-8g |
| GPUs | 8 x NVIDIA H200, 143 GB each |
| inference engines | 4, at tp=2 (8 GPUs / 2) |
| model | Qwen2.5-32B-Instruct + LoRA r32/a32 |
| vLLM / verl | 0.22.1 / 0.8.0 |
| KV tier | Mooncake over RDMA, 8 x 128 GB host segments |
| sandbox pool | 21 x e2-standard-16, gVisor |

This applies for both arms of the test. They differ
only by the `kv_transfer_config` flags.

### Workload

Each training step samples 128 SWE-bench tasks with 4 generations each, so
**512 trajectories per rollout**. A trajectory is a multi-turn agent episode.
The model writes a bash command. The harness runs it in a sandbox. The output
comes back as an observation and the loop repeats, until the model submits or
hits 32 turns.

| | |
|---|---|
| trajectories per rollout | 512 |
| steps per arm | 4 |
| turns per trajectory | ~29 |
| tokens presented per turn | ~2,925 |
| KV pool | 424k tokens across 4 engines |

Two things about this workload drive KV reuse.

**Every turn is a separate engine request.** It carries the whole conversation
so far. Turn 20 re-presents everything from turns 1 to 19. So nearly all the
prompt tokens an engine sees are tokens it has already seen.

**Between turns a trajectory holds no GPU memory.** It is away running a shell
command. Its KV sits in the evictable part of the pool. Whether it survives
until the trajectory returns decides if the next turn is cheap or expensive.

### Making the local cache fall short

We wanted the baseline short of local cache for the whole run, not just at the
start. Three settings did that.

| setting | value | reason |
|---|---|---|
| batch size | 128, x4 generations = 512 trajectories | enough concurrent trajectories that their combined context exceeds the pool |
| `gpu_memory_utilization` | 0.38 | ~106k tokens per engine |
| shell command timeout | 15 s | a command running longer than this is hung. Long hangs leave the engine idle and spread the trajectories out, which removes the pressure we are trying to create |

It worked. The baseline served only **14.3%** of its prompt tokens from local
cache, and was preempted 317 times.

---

## How the tier decides to fetch

vLLM already hashes every KV block, chaining each hash into the next, to drive
its own prefix cache. The connector reuses those same hashes as store keys. So
the local cache and the remote tier are keyed identically, and a block pulled
from the tier is an ordinary cached block once it lands.

A fetch happens only when all five of these hold.

| | condition | why |
|---|---|---|
| 1 | the local cache does not already cover the prompt | if it does, skip the lookup. The lookup is a blocking network call inside the scheduler loop |
| 2 | the tier has an unbroken run of blocks from the start | it is a prefix match. A gap truncates it |
| 3 | no tier wipe is pending | otherwise we match keys that are about to be deleted |
| 4 | the match is over 1,024 tokens | below that, local prefill beats a round trip |
| 5 | fewer than 1 fetch already in flight | a waiting request holds its full context in GPU memory until its data lands |

On a fetch the request reserves its blocks and pauses. The transfer is issued
after the model forward launches, so it overlaps other requests' compute. A
background thread writes over RDMA straight into the paged cache, with no host
copy. The request resumes a step or two later and prefills only the part the
tier did not cover.

If a fetch fails the request just recomputes. That is why declining at any of
the five gates is always safe.

---

## Results

4-step means. Change is the shared tier relative to local-only.

| | local-only | shared KV tier | change |
|---|---|---|---|
| **sampling time per trajectory** | **348.2 s** | **226.2 s** | **-35.1%** |
| decode tokens produced | 3,034,780 | 3,096,673 | +2.0% |
| prompt tokens recomputed | 148,596,234 | 67,037,475 | **-54.9%** |
| **sampling time, slowest trajectory** | **1,146.7 s** | **883.8 s** | **-22.9%** |
| tool-call time per trajectory | 57.1 s | 59.7 s | +4.5% |
| KV pool occupancy while busy | 0.751 | 0.659 | -12.3% |
| peak requests running | 112.8 | 93.3 | -17.3% |
| preemptions per step | 79.3 | 81.3 | +2.5% |
| `perf/throughput` from verl, for reference | 225.5 | 328.2 | +45.6% |
| host memory used | 164.9 GB | 1,221.9 GB | **+641%** |

Workloads matched: total prompt tokens within 2.4%, batch tokens and response
length within 0.03%, and prompt length per step identical between the arms.

> **NOTE — the two headline metrics.** Both are recorded by verl, not derived.
> *Sampling time per trajectory* is
> `timing_s/agent_loop/generate_sequences/mean`: the time one trajectory spends
> inside generate calls, summed over its turns and averaged across all 512
> trajectories. It includes time queued behind other requests, which is the
> point. *Sampling time, slowest trajectory* is the same quantity's maximum
> across the 512.
>
> A maximum over 512 samples can be noisy, so we checked this one. Its spread
> is mild, at 3.3-3.9x the mean, and the per-step ratios between arms are 0.75,
> 0.74, 0.87 and 0.73. That is a consistent effect rather than a lucky draw. By
> contrast the maximum *tool-call* time has a 9-17x spread and ratios swinging
> from 0.40 to 1.05, so we do not report it as a result.

> **NOTE — why we do not lead with verl's own `perf/throughput`.** It is the
> obvious metric to reach for, and it agrees with us in direction, but it is not
> an independent measurement. verl computes it as
> `total_num_tokens / (timing_s/step x n_gpus)`. The numerator is tokens in the
> training batch, which is identical across the arms by construction, at 0.032%.
> So the metric is a constant divided by step wall clock. We checked: it
> reproduces that formula to four decimal places in all 8 arm-steps, and its
> ratio (1.456x) simply inverts the step-time ratio (1.446x).
>
> That matters because step wall clock is a makespan. A rollout does not finish
> until its slowest single trajectory does, and we confirmed rollout wall clock
> equals the slowest trajectory's generate plus tool time to within 3% in every
> step of both arms. So `perf/throughput` and `timing_s/gen` track 1 trajectory
> out of 512, and most of their apparent gap here comes from that trajectory's
> shell commands rather than from the KV tier. The two headline metrics are
> averages over all 512 and do not have this problem.

> **NOTE — host memory is a standing cost.** The 1.2 TB is the Mooncake
> segments. It is paid whether or not the tier is being hit.

> **NOTE — preemptions did not fall.** 317 against 325. The tier relieved
> pressure as occupancy, not as fewer evictions. We would not claim from this
> run that the tier prevents thrashing.

---

## Where the prompt tokens came from

Every prompt token an engine processes comes from one of three places. It is
recomputed on the GPU, served from the local prefix cache, or fetched from the
tier. The three always sum to the total.

**Share of all prompt tokens:**

| | recomputed on GPU | local prefix cache | fetched from tier |
|---|---|---|---|
| local-only | 85.7% | 14.3% | — |
| shared KV tier | 37.7% | 18.5% | 43.8% |

**The same split per turn, in tokens:**

| | presented | recomputed | local cache | tier |
|---|---|---|---|---|
| local-only | 2,925 | 2,506 | 419 | 0 |
| shared KV tier | 2,926 | 1,104 | 541 | 1,280 |

Both arms are shown the same ~2,925 tokens per turn. Only the split moves.

The useful lesson is in the middle column. **The tier did not take work away
from the local cache. It gave it more**, from 419 to 541 tokens per turn. A
fetched block becomes an ordinary cached block once it lands, so it can serve a
local hit on that trajectory's next turn. The two tiers add up rather than
compete. This is also why the tier is only ever consulted on a local miss.

---

## Why sampling got faster

Saving prefill work does not obviously save this much time, so the size is
worth being explicit about.

Per turn the tier avoids 1,399 tokens of prefill and saves 4.40 s of sampling
time. Those tokens are worth about **0.11 s** of GPU compute. For a 32B dense
model at tp=2 on H200s prefill costs roughly 0.08 ms per token, so 1,399 tokens
is 0.113 s.

The arithmetic we skipped therefore accounts for about 2.6% of the time saved.

Our explanation for the rest is queueing. Prefill and decode run on the same
GPUs. In the baseline 86% of every turn's context is being recomputed, and up
to 299 requests are waiting to be scheduled at once. At that load the engine is
past the point where adding work costs only its own compute time. Work removed
from one request shortens the wait for every other request in the queue.

This is a theory that fits the numbers, not something we measured directly.
What the measurement says is that the time saved is roughly 39x the compute
saved, so the saving is not coming from the arithmetic.

It also means the result depends on load. At low contention we would expect
most of this to disappear, and in an earlier low-pressure run of ours it did.

---

## Cost of an avoided prefill token

This puts a price on one token of avoided prefill, from measured values only.

The two inputs are the sampling-time difference and the prefill-token
difference, both per trajectory:

```
sampling time     348.2 s  -  226.2 s  =  122.1 s saved per trajectory
prefill tokens     72,557  -   32,733  =  39,824 tokens avoided per trajectory

122.1 s / 39,824 tokens  =  3.07 ms per avoided prefill token
```

Sampling time is the headline metric above. The per-trajectory token counts are
the engine's own by-source counters divided by the trajectories in the arm,
512 x 4 steps = 2,048. As a check, the same division per *turn* instead of per
trajectory gives the same answer, because turn counts match between arms.

**What the number is.** The marginal cost of a prefill token *to this system at
this load*. It is not the hardware cost of prefilling a token, which is about
0.08 ms. The 38x gap between them is the queueing effect above.

**What it is for.** Capacity questions. "If I remove a million tokens of prefill
from this workload, what do I get back."

**What to be careful about.** It credits the whole time saving to avoided
prefill, and the arms differ in scheduling too. It rises with load, so it should
be recalculated per setup rather than carried across.

---

## What we cannot explain yet

**The tier arm runs at about twice the baseline's policy entropy**, 0.30-0.40
against 0.19-0.22. It is flat across all four steps and well inside the usual
validity limit of 1.0. It is present from step 1, so it is a constant offset
rather than something that grows.

It cannot be stale KV, because step 1 has no earlier weights to be stale
against. It appeared alongside the jump to a 43.8% fetch share. Our theory is
numerical: KV recomputed locally and KV fetched from the tier may not be
bit-identical, because a prefix assembled from fetched blocks hits different
attention chunk boundaries. At a 43.8% share that could stop being negligible.

The alternative is that the two arms sampled different trajectories and saw
different states. Task outcomes are near identical between the arms, which
argues they are comparable, but does not settle it.

This should be characterised before the tier is used for convergence work. The
cheap test is a single step comparing logprobs of fetched versus recomputed
blocks on identical prompts.

---

## Files

| file | contents |
|---|---|
| `metrics.csv` | every recorded number. 93 metrics x 4 steps x 2 arms, with means and deltas |
| `engine_timeseries.csv.gz` | engine metrics over time, one row per sample, stamped with step and phase |
| `recompute_driver.log.gz`, `store_driver.log.gz` | raw verl logs. Source of every training-loop metric |
| `recompute_scrape.log.gz`, `store_scrape.log.gz` | raw vLLM `/metrics`, sampled every 60 s. Source of every engine metric |
| `recompute.md`, `store.md` | per-step detail for each arm |
| `harvest34.py` | regenerates every table here from the logs |
| `p34_arm.sh` | the exact run script |
