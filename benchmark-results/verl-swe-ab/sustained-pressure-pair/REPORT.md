# Shared KV tier vs local-only recompute — SWE-bench agent RL, Qwen2.5-32B

## TLDR

Adding a shared RDMA KV tier to an agentic RL rollout cut **sampling time per
trajectory by 35%** and raised **generation throughput 1.61x**. It did this
while producing 4.6% more tokens.

| | local-only | shared KV tier | change |
|---|---|---|---|
| sampling time per trajectory | 348.2 s | 226.2 s | **-35.1%** |
| generation throughput | 4.06 tok/traj-s | 6.54 tok/traj-s | **1.61x** |
| prompt tokens recomputed on GPU | 148.6M | 67.0M | **2.22x less** |

The prefill saving is the expected benefit of KV offload. The sampling-time
saving is the result worth attention, because it is far larger than the prefill
saving alone can explain. See "Why sampling got faster".

---

## What we tested

Production RL rollouts run local-only today. That means vLLM's paged prefix
cache, with a full recompute whenever a block is evicted. This is the baseline.

We theorise that an external KV tier is most useful when the local prefix cache
is not enough to save on prefill. That should be the normal condition for
long-context agentic workloads. They present a large accumulated context on
every turn, and they run enough trajectories at once that the pool cannot hold
them all.

So the test needs a setup where the baseline is genuinely short of local cache.
If the local cache is doing its job there is nothing for a second tier to
rescue, and earlier runs of ours found exactly that.

---

## Setup

### Stack

| | |
|---|---|
| node | 1 x a3-ultragpu-8g, 8 x NVIDIA H200 |
| engines | 4 vLLM engines, tp=2 |
| model | Qwen2.5-32B-Instruct + LoRA r32/a32 |
| vLLM / verl | 0.22.1 / 0.8.0 |
| KV tier | Mooncake over RDMA, 8 x 128 GB host segments |
| sandboxes | gVisor pool, 21 x e2-standard-16 |

Both arms run the same engine, scheduler and local prefix cache. They differ
only by the `kv_transfer_config` flags, checked on the live command line before
each arm started.

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

One thing to watch if you repeat this. Batch size must divide the dataset. We
used 4 steps x 128 against 256 rows, which is exactly 2 epochs. A batch that
does not divide it makes verl drop the remainder and reshuffle, so the task set
changes between steps and nothing warns you.

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
| **generation throughput** | **4.06 tok/traj-s** | **6.54 tok/traj-s** | **1.61x** |
| decode tokens produced | 2,897,146 | 3,030,513 | +4.6% |
| prompt tokens recomputed | 148,596,234 | 67,037,475 | **-54.9%** |
| slowest trajectory's sampling time | 1,146.7 s | 883.8 s | -22.9% |
| tool-call time per trajectory | 57.1 s | 59.7 s | +4.5% |
| KV pool occupancy while busy | 0.751 | 0.659 | -12.3% |
| peak requests running | 112.8 | 93.3 | -17.3% |
| preemptions per step | 79.3 | 81.3 | +2.5% |
| host memory used | 164.9 GB | 1,221.9 GB | **+641%** |

Workloads matched: total prompt tokens within 2.4%, batch tokens and response
length within 0.03%, and prompt length per step identical between the arms.

> **NOTE — the two headline definitions.** *Sampling time per trajectory* is the
> time one trajectory spends inside generate calls, summed over its turns and
> averaged across all 512 trajectories. It includes time queued behind other
> requests, which is the point. *Generation throughput* is decode tokens divided
> by that same generation time, aggregated over the arm. It answers "how many
> tokens does a second of generation buy", and it does not move with how many
> tokens a given step happened to present.

> **NOTE — ignore `timing_s/gen` and `perf/throughput` in the raw logs.** Both
> divide by a rollout's wall clock, and a rollout does not finish until its
> slowest single trajectory does. We checked: rollout wall clock equals the
> slowest trajectory's generate plus tool time to within 3%, in every step of
> both arms. Those metrics therefore track 1 trajectory out of 512, and most of
> their apparent gap here comes from that trajectory's shell commands rather
> than from the KV tier. The two metrics above are averages over all 512 and do
> not have this problem.

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
