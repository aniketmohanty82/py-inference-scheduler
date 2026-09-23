# Shared KV tier vs local-only recompute, and our connector vs vLLM's native one — SWE-bench agent RL, Qwen2.5-32B, 12 steps

## TLDR

Adding a shared RDMA KV tier to an agentic RL rollout cut **sampling time per
trajectory by 65-67%** over 12 training steps. The slowest trajectory in each
rollout, which is the one a step actually waits on, improved by 41-46%.

vLLM 0.29 ships that tier natively. Our connector, which existed to add a
weight-update flush and decode-KV saving on top of it, now measures the same as
the native one: sampling time within 5%, coin-flip sign across the 12 steps,
identical fidelity. **On this workload our connector has no measurable edge
over vLLM native.**

| | local-only | vLLM native tier | our tier | native vs local-only | ours vs native |
|---|---|---|---|---|---|
| sampling time per trajectory | 169.3 s | 59.5 s | 56.3 s | **-64.9%** | -5.3% |
| sampling time, slowest trajectory | 811.7 s | 477.3 s | 440.6 s | **-41.2%** | -7.7% |
| prompt tokens recomputed on GPU | 279.3M | 24.2M | 27.2M | **11.5x less** | +12% |

---

## What we tested

We theorize that an external KV tier that HBM can offload to is better than
always recomputing requests. This is most useful when the local prefix cache
is not enough to save on prefill. We have seen this to be a normal condition for
long-context agentic workloads. They present a large, additive context on
every turn, and they run enough trajectories at once that the HBM memory pool
cannot hold them all.

Two questions, one run.

**Does the tier help?** Same question as the earlier 4-step pair, over 12
steps instead of 4 so the answer does not rest on the cold first step.

**Does our connector still earn its place?** vLLM 0.29 added the two things
our connector was built for: a wipe of the tier at every weight update
(`reset_cache`), and saving decode-generated KV (`save_decode_cache`). What
remains ours is one admission rule: decline a tier fetch under 1,024 tokens
and recompute locally instead. So the third arm is the untouched upstream
connector, and the comparison between it and ours isolates that one rule.

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
| vLLM / verl | 0.29.0 / 0.9.0 |
| KV tier | Mooncake over RDMA, 8 x 128 GB host segments |
| sandbox pool | 21 x e2-standard-16, gVisor |

This applies for all three arms. They differ only by the `kv_transfer_config`
flags.

| arm | `kv_connector` | what it is |
|---|---|---|
| local-only | none | every prompt token is either a local prefix-cache hit or recomputed |
| vLLM native tier | `MooncakeStoreConnector` | upstream, unmodified, with `save_decode_cache: true` |
| our tier | `RLPullPolicyConnector` | the upstream class plus the 1,024-token fetch floor, with `save_decode_cache: true` |

vLLM 0.29 needed three things to run under verl at all, applied identically
to every arm: flash-attention compiled from source (no prebuilt wheel exists
for its torch), a 15-line fix to vLLM's LoRA loading (upstream PR #51423, open
at the time), and verl's legacy trainer path. None of them touch the KV tier.

### Workload

Each training step samples 128 SWE-bench tasks with 4 generations each, so
**512 trajectories per rollout**. A trajectory is a multi-turn agent episode.
The model writes a bash command. The harness runs it in a sandbox. The output
comes back as an observation and the loop repeats, until the model submits or
hits 32 turns.

| | |
|---|---|
| trajectories per rollout | 512 |
| steps per arm | 12 |
| turns per trajectory | ~24 |
| tokens presented per turn | ~2,800 |
| KV pool | 424k tokens across 4 engines |

Two things about this workload drive KV reuse.

**Every turn is a separate engine request.** It carries the whole conversation
so far. Turn 20 re-presents everything from turns 1 to 19. So nearly all the
prompt tokens an engine sees are tokens it has already seen.

**Between turns a trajectory holds no GPU memory.** It is away running a shell
command. Its KV sits in the evictable part of the pool. Whether it survives
until the trajectory returns decides if the next turn is cheap or expensive.

### Making the local cache fall short

Same three settings as the 4-step pair, unchanged.

| setting | value | reason |
|---|---|---|
| batch size | 128, x4 generations = 512 trajectories | enough concurrent trajectories that their combined context exceeds the pool |
| `gpu_memory_utilization` | 0.38 | ~106k tokens per engine |
| shell command timeout | 15 s | a command running longer than this is hung. Long hangs leave the engine idle and spread the trajectories out, which removes the pressure we are trying to create |

The local-only arm served **35.7%** of its prompt tokens from local cache over
the 12 steps and recomputed the rest.

### How the arms were run

All three ran on the same node and the same pods, one after another, each
against a freshly emptied tier. Local-only and vLLM native ran back to back.
Our arm ran about ten hours later, because its first attempt crashed an engine
at step 9 (see "Where our connector went wrong"). It was rerun with the fix on
the same pods. That gap matters for one set of numbers below and is called out
where it does.

---

## How the tier decides to fetch

vLLM already hashes every KV block, chaining each hash into the next, to drive
its own prefix cache. The connector reuses those same hashes as store keys. So
the local cache and the remote tier are keyed identically, and a block pulled
from the tier is an ordinary cached block once it lands.

In vLLM native a fetch happens when the first three of these hold. Our
connector adds the fourth.

| | condition | why | native | ours |
|---|---|---|---|---|
| 1 | the local cache does not already cover the prompt | if it does, skip the lookup. The lookup is a blocking network call inside the scheduler loop | yes | yes |
| 2 | the tier has an unbroken run of blocks from the start | it is a prefix match. A gap truncates it | yes | yes |
| 3 | no tier wipe is pending | otherwise we match keys that are about to be deleted. the tier is wiped at every weight update | yes | yes |
| 4 | the match is over 1,024 tokens | below that, local prefill beats a round trip | no | yes |

The earlier pair also capped fetches in flight at one. That cap is gone. At
this pool size it cost more than it saved: it serialised fetches and cut the
tier's share of prompt tokens from 73% to 44% in a 4-step check we ran before
this benchmark.

On a fetch the request reserves its blocks and pauses. A background thread
writes over RDMA straight into the engine's KV blocks, with no copy through
host memory. The request resumes one or two scheduling passes later and
prefills only the part the tier did not cover.

If a fetch fails the request just recomputes. That is why declining at any of
the gates is always safe.

**The wipe works in vLLM native.** We checked at the store itself rather than
in the logs. At the first weight update the master's key count went from
495,458 to 8,972 in one 30-second sample, with a matching burst of delete
calls, then rebuilt from the next step's writes. That is verl's boundary call
reaching the tier. Our connector inherits this code unchanged.

---

## Results

12-step means.

| | local-only | vLLM native tier | our tier | native vs local-only | ours vs native |
|---|---|---|---|---|---|
| **sampling time per trajectory** | **169.3 s** | **59.5 s** | **56.3 s** | 🟢 **-64.9%** | ⚪ -5.3% |
| **sampling time, slowest trajectory** | **811.7 s** | **477.3 s** | **440.6 s** | 🟢 **-41.2%** | ⚪ -7.7% |
| prompt tokens recomputed | 279,253,366 | 24,247,030 | 27,224,236 | 🟢 **-91.3%** | 🔴 +12.3% |
| tool-call time per trajectory | 45.9 s | 47.0 s | 50.7 s | 🔴 +2.4% | 🔴 +7.8% |
| `timing_s/gen` from verl, for reference | 1,095.8 s | 744.0 s | 853.8 s | 🟢 -32.1% | 🔴 +14.8% |
| `perf/throughput` from verl, for reference | 270.1 | 362.1 | 337.5 | 🟢 +34.1% | 🔴 -6.8% |
| preemptions over the run | 408 | 2,146 | 1,756 | 🔴 +426% | 🟢 -18.2% |
| host memory used | 131.2 GB | 1,191.2 GB | 1,191.5 GB | 🔴 **+808%** | ⚪ +0.0% |
| policy entropy | 0.2037 | 0.1972 | 0.2010 | ⚪ -3.2% | ⚪ +1.9% |
| decode tokens produced | 7,400,972 | 7,319,259 | 7,330,035 | ⚪ -1.1% | ⚪ +0.1% |

🟢 the right-hand arm did better · 🔴 it did worse · ⚪ neither: inside noise, or
describing the workload rather than scoring it. Direction is not always "lower
is better": throughput and decode tokens are better higher, everything else
better lower.

Workloads matched: total prompt tokens within 4.8%, batch tokens within 1.7%,
response length within 1.9%, turns within 1.2%, and prompt length per step
identical across all three arms.

> **NOTE — the two headline metrics.** Both are recorded by verl, not derived.
> *Sampling time per trajectory* is
> `timing_s/agent_loop/generate_sequences/mean`: the time one trajectory spends
> inside generate calls, summed over its turns and averaged across all 512
> trajectories. It includes time queued behind other requests, which is the
> point. *Sampling time, slowest trajectory* is the same quantity's maximum
> across the 512.
>
> A maximum over 512 samples is noisy, and here it is noisier than in the
> 4-step pair: the spread is 6x the mean in local-only and 11x in both tier
> arms, and the per-step ratio of ours to native swings from 0.29 to 2.23. We
> report it because it is what a step waits on, but the per-trajectory mean is
> the number to trust.

> **NOTE — what "ours vs native" can and cannot say.** The test was set up
> before the run to call the two arms the same if sampling time per trajectory
> landed within ±7% without significance. It landed at -5.3%, in our favour in
> 6 of 12 steps, with a 95% interval of -17% to +7% on the paired difference.
> That is a tie. It is not evidence that our connector is slower, and it is not
> evidence that it is faster.

> **NOTE — why we do not lead with verl's own `perf/throughput`.** It is the
> obvious metric to reach for, but it is not an independent measurement. verl
> computes it as `total_num_tokens / (timing_s/step x n_gpus)`. The numerator
> is tokens in the training batch, which is matched across the arms within
> 1.7%. So the metric is a near-constant divided by step wall clock. We
> checked: it reproduces that formula exactly in all 36 arm-steps.
>
> That matters because a rollout does not finish until its slowest single
> trajectory does. We confirmed this: rollout wall clock equals the slowest
> trajectory's generate plus tool time to within 7%, in every step of all
> three arms. So `perf/throughput` and `timing_s/gen` track 1 trajectory out of
> 512. The two headline metrics are averages over all 512 and do not have this
> problem.

> **NOTE — the rows where ours reads worse are sandbox time, not the tier.** In
> `timing_s/gen`, `perf/throughput` and tool-call time, our arm looks 7-15%
> behind native. All three are the same effect. The slowest trajectory in each
> of our steps *generated* 139 s faster than native's on average and spent
> 249 s longer in shell commands, which nets to the +110 s on `timing_s/gen`
> exactly. And it was not only the slowest trajectory: every sandbox call in
> our arm ran about 8% slower, in 10 of 12 steps, on the same 21 nodes with no
> scheduling events. The KV connector does not touch the sandbox. The sandbox
> nodes are shared, and our arm ran ten hours after the other two. We read
> those three rows as drift in the sandbox tier over that gap.

> **NOTE — host memory is a standing cost.** The 1.2 TB is the Mooncake
> segments. It is paid whether or not the tier is being hit.

> **NOTE — preemptions went up, a lot.** 408 in local-only against ~2,000 in
> the tier arms. That is a change of regime, not a failure. In local-only the
> pool is full of contexts that are slow to rebuild, so the engine preempts
> rarely and each preemption is expensive. In the tier arms a preempted context
> comes back from the tier in milliseconds, so the engine is willing to preempt
> five times as often and still runs 65% faster. Our arm preempted 18% less
> than native with no visible timing benefit.

---

## Where the prompt tokens came from

Every prompt token an engine processes comes from one of three places. It is
recomputed on the GPU, served from the local prefix cache, or fetched from the
tier. The three always sum to the total.

**Share of all prompt tokens over 12 steps:**

| | recomputed on GPU | local prefix cache | fetched from tier |
|---|---|---|---|
| local-only | 64.3% | 35.7% | — |
| vLLM native tier | 5.9% | 49.3% | 44.9% |
| our tier | 6.5% | 52.2% | 41.3% |

**The same split per turn, in tokens:**

| | presented | recomputed | local cache | tier |
|---|---|---|---|---|
| local-only | 2,883 | 1,855 | 1,029 | 0 |
| vLLM native tier | 2,784 | 163 | 1,371 | 1,249 |
| our tier | 2,814 | 182 | 1,469 | 1,162 |

All arms are shown roughly the same ~2,800 tokens per turn. Only the split
moves.

The useful lesson is in the middle column, and it held from the 4-step pair.
**The tier did not take work away from the local cache. It gave it more**,
from 1,029 to ~1,400 tokens per turn. A fetched block becomes an ordinary
cached block once it lands, so it can serve a local hit on that trajectory's
next turn. The two tiers add up rather than compete.

The last column is where our one remaining rule shows. Declining fetches under
1,024 tokens moved 87 tokens per turn from the tier column to the recompute
column and 98 into the local cache. The engine time it saved is inside the
noise. That is the whole measured effect of our connector.

---

## Why sampling got faster

Saving prefill work does not obviously save this much time, so the size is
worth being explicit about.

Per turn the native tier avoids 1,714 tokens of prefill and saves 4.53 s of
sampling time. Those tokens are worth about **0.14 s** of GPU compute. For a
32B dense model at tp=2 on H200s prefill costs roughly 0.08 ms per token, so
1,714 tokens is 0.137 s.

The arithmetic we skipped therefore accounts for about 3% of the time saved.

Our explanation for the rest is queueing. Prefill and decode run on the same
GPUs. In local-only 64% of every turn's context is being recomputed, and up to
264 requests are waiting to be scheduled at once. At that load the engine is
past the point where adding work costs only its own compute time. Work removed
from one request shortens the wait for every other request in the queue.

This is a theory that fits the numbers, not something we measured directly.
What the measurement says is that the time saved is roughly 33x the compute
saved, so the saving is not coming from the arithmetic.

It also means the result depends on load. At low contention we would expect
most of this to disappear, and in an earlier low-pressure run of ours it did.

---

## Cost of an avoided prefill token

This puts a price on one token of avoided prefill, from measured values only,
for the native tier against local-only.

```
sampling time     169.3 s  -   59.5 s  =  109.8 s saved per trajectory
prefill tokens     45,451  -    3,946  =  41,505 tokens avoided per trajectory

109.8 s / 41,505 tokens  =  2.65 ms per avoided prefill token
```

Sampling time is the headline metric above. The per-trajectory token counts are
the engine's own counters for where each prompt token came from, divided by
the trajectories in the arm, 512 x 12 steps = 6,144. Our arm gives 2.75 ms by
the same arithmetic.

**What the number is.** The marginal cost of a prefill token *to this system at
this load*. It is not the hardware cost of prefilling a token, which is about
0.08 ms. The 33x gap between them is the queueing effect above. The 4-step pair
measured 3.07 ms under the same settings; the difference is within what
run-to-run variation on this rig looks like.

**What it is for.** Capacity questions. "If I remove a million tokens of prefill
from this workload, what do I get back."

**What to be careful about.** It credits the whole time saving to avoided
prefill, and the arms differ in scheduling too. It rises with load, so it should
be recalculated per setup rather than carried across.

---

## Where our connector went wrong

Our arm's first attempt crashed one of the four engines at step 9. The cause
was ours. The upstream scheduler records a pending fetch the moment it finds a
match, before our rule gets to decline it. When we declined, the record was
left behind, and nine steps later the engine tripped an assertion on it. The
fix is one line, drop the record when declining, and the rerun completed all 12
steps clean.

The crash is worth recording for what happened next. The engine's frontend
stayed up after the engine died, so it kept reporting the last numbers it had.
verl pins each trajectory to one engine for its whole life and never moves it.
113 trajectories were pinned to the dead engine. The other three engines
finished their share and sat idle, and the step waited five and a half hours
for the 113 that could never come back. That is a property of verl's routing,
not of the tier, and it is the same blindness that leaves engines idle while
others queue in a healthy run.

---

## What changed since the 4-step pair

**The entropy offset is gone.** The 4-step pair's tier arm ran at twice the
baseline's policy entropy from step 1, and we could not explain it. Here all
three arms sit at 0.20 across all 12 steps. The connector in that pair carried
its own decode-KV saving code, which reached into the scheduler's internals;
it was retired for this run in favour of upstream's `save_decode_cache`. That
is consistent with our old code having caused the offset, and it is the only
thing that changed on that path. It is not proof.

**The result is bigger.** The 4-step pair measured -35% on sampling time; this
run measures -65%. Two things moved. The fetch cap that serialised transfers is
gone, and the cold first step is one of 12 rather than one of 4. The 4-step
pair's own step 2-4 numbers were already near -55%.

---

## Files

| file | contents |
|---|---|
| `recompute_driver.log.gz`, `native_driver.log.gz`, `store_driver.log.gz` | raw verl logs, one per arm. Source of every training-loop metric |
| `recompute_scrape.log.gz`, `native_scrape.log.gz`, `store_scrape.log.gz` | raw vLLM `/metrics`, one sweep of all four engines every ~7.5 min. Source of every engine metric |
| `store_attempt1_crashed_driver.log.gz` | the first store attempt, kept for the crash record. Not used in any table |
| `recompute.md`, `native.md`, `store.md` | per-step detail for each arm |
| `harvest12.py` | the parser behind every table here |
| `p40_arm.sh` | the exact run script |
