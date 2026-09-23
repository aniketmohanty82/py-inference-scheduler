# Router-side flow control vs unthrottled admission — SWE-bench agent RL, Qwen3-32B

## TLDR

Holding requests at the router until an engine reads below a KV threshold cut
**preemptions by 40% per generated token** and **engine queue wait by 74%**,
on a fleet running at a median KV occupancy of 0.95. It cost 8.5% of decode
throughput and 6.5 points of prefix-cache hit rate, because a parked
trajectory comes back to find its prefix evicted. Net GPU work is close to a
wash. The gate buys stability and engine-side latency, not tokens per
GPU-hour.

| | unthrottled | flow control | change |
|---|---|---|---|
| preemptions per generated Mtok | 112.7 | 67.2 | **-40.4%** |
| mean engine queue wait | 3.78 s | 0.98 s | **-74.0%** |
| decode throughput per busy engine-second | 897 tok/s | 821 tok/s | **-8.5%** |

A second, earlier pair on a different pod generation measured -34% on
preemptions. Both pairs are individually significant.

---

## What we tested

We theorize that an RL rollout oversubscribes its inference engines in a
specific way: the trainer composes a whole batch of trajectories and admits
every generate call the moment its sandbox returns, so the engines' KV pools
are pushed past capacity and vLLM responds by preempting, which evicts a
running request's KV and re-prefills it later. `simple_backpressure` holds a
request at the router while every engine reads above a KV or queue-depth
threshold, and releases it to the least-loaded engine once one drops under.

The goal is to see whether that admission gate reduces preemptions on a
long-context, heavily preemptive agentic RL workload, and what it costs. An
earlier campaign at 7B established that the gate cuts preemptions 30-46% but
that each preemption there was too cheap to matter. At 32B with reasoning-
length contexts every eviction re-prefills ~5x the compute.

---

## Setup

### Stack

| | |
|---|---|
| nodes | 2 |
| machine type | a3-highgpu-8g (spot) |
| GPUs | 16 x NVIDIA H100, 80 GB each |
| inference engines | 4, at tp=4 (16 GPUs / 4) |
| model | Qwen3-32B (thinking) + LoRA r32/a32 |
| vLLM / verl | 0.29.0 / 0.9.0 (image `rllm-verl-mooncake@sha256:8cdf4fb8…`, pinned by digest) |
| router | `integration/verl/verl_hook.py`, `least_queue` scorer, shared-inflight ledger |
| sandbox pool | 4 x e2-standard-32, gVisor |

This applies for both arms of the test. They differ only by the presence of
the `flow_control` block in the router config (`lq-off.yaml` vs
`lq-kv90.yaml`: kv threshold 0.90, waiting threshold 4).

### Workload

Each training step samples 64 SWE-bench tasks with 4 generations each, so
**256 trajectories per rollout**. A trajectory is a multi-turn agent episode.
The model writes a bash command. The harness runs it in a sandbox. The output
comes back as an observation and the loop repeats, until the model submits or
hits 25 turns.

| | |
|---|---|
| trajectories per rollout | 256 |
| steps per arm | 1 |
| turns per trajectory | up to 25 |
| response tokens per trajectory | ~8,200-8,900 |
| prompt tokens presented per generated token | 6.7-8.3 |
| KV pool | ~155k tokens per engine, ~620k across 4 |

Two things about this workload drive preemption.

**Every turn is a separate engine request carrying the whole conversation.**
Prompt tokens outnumber generated tokens roughly 7 to 1. A preemption is
therefore expensive twice: the evicted request's KV is gone, and when it
resumes it re-prefills a context that is mostly repeat.

**Between turns a trajectory holds no GPU memory.** It is away running a
shell command. When it returns, whether its prefix is still cached decides
whether the next turn is cheap or expensive. Anything that delays its return
-- including a router that parks it -- makes eviction more likely.

### Making the engines saturate

We wanted pressure to come from the model's own footprint, not from a pool
starved below one request's working set (the 7B campaign had to do that,
and it invites the objection that the workload was rigged).

| setting | value | reason |
|---|---|---|
| `gpu_memory_utilization` | 0.35 | 16 GB of tp=4 weights + ~10 GB KV per GPU; ~155k tokens per engine, about 5 full contexts, well above the one-context deadlock floor |
| trajectories | 64 tasks x 4 generations | 64 per engine against ~5 contexts of pool: ~12x oversubscribed |
| observation cap / turns | 20,000 chars / 25 | upstream regime; long contexts are the point |

It worked. The unthrottled arm ran at a **median KV occupancy of 0.95**
(engine-side, sampled uniformly: p90 0.98), spent 5.9% of samples at the
hard ceiling of 0.99+, and was preempted **194 times in a single rollout** --
the entire 7B storm grid produced 509 across four.

---

## How the gate decides to park

The router refreshes every engine's stats before each admission (KV
occupancy, queue depth, running requests) and runs the gate over the
candidate list. An engine is excluded when either threshold is met. If every
engine is excluded, the request parks.

| | condition | why |
|---|---|---|
| 1 | engine KV occupancy >= 0.90 | the next request's prefill would push it into eviction |
| 2 | engine waiting queue >= 4 | queued requests convert into KV growth the moment blocks free |
| 3 | all engines excluded | park; otherwise route to the least-loaded survivor |
| 4 | parked: a watcher re-polls every 100 ms | releases one waiter per tick at an AIMD-paced rate, re-gated on the fresh snapshot |
| 5 | inflight counts are fleet-wide | verl fans the batch across 8 loop workers; a shared ledger keeps each one's view of "least loaded" honest |

Parking costs nothing on the engine. What it costs is time: a parked
trajectory's prefix keeps ageing in a cache under eviction pressure.

---

## Results

Single-step values. Change is flow control relative to unthrottled.

| | unthrottled | flow control | change |
|---|---|---|---|
| **preemptions per generated Mtok** | **112.7** | **67.2** | 🟢 **-40.4%** |
| **preemptions** | **194** | **110** | 🟢 **-43.3%** |
| **mean engine queue wait** | **3.78 s** | **0.98 s** | 🟢 **-74.0%** |
| samples at the hard KV ceiling (>= 0.99) | 5.9% | 3.1% | 🟢 -47.5% |
| decode throughput per busy engine-second | 896.9 tok/s | 821.0 tok/s | 🔴 **-8.5%** |
| prefix-cache hit rate | 30.7% | 24.2% | 🔴 **-6.5 pp** |
| prompt tokens prefilled | 11,493,364 | 13,537,253 | 🔴 +17.8% |
| all tokens processed per busy engine-second | 6,883 | 7,607 | ⚪ +10.5% |
| samples above the gate threshold (>= 0.90) | 27.0% | 32.3% | ⚪ +5.3 pp |
| decode tokens produced | 1,722,062 | 1,637,890 | ⚪ -4.9% |
| response length per trajectory, from verl | 8,236 | 8,927 | ⚪ +8.4% |
| rollout wall clock, from verl | 535 s | 568 s | ⚪ +6.0% |
| requests parked / engines dropped from a ballot | 0 / 0 | 13,670 / 1,256 | ⚪ the intervention |

🟢 flow control did better · 🔴 it did worse · ⚪ neither, this one describes
the workload or the mechanism rather than scoring it. Direction is not always
"lower is better": throughput and hit rate are better higher, everything
else better lower.

Workloads matched: same 64 tasks, generations and seed; engine requests
within 2.1% (1,883 vs 1,923); decode tokens within 4.9%; response length
within 8.4% (the flow-control arm generated more).

> **NOTE — the headline metrics are engine counters, not verl-derived.**
> *Preemptions* is vLLM's `num_preemptions_total`, read two ways -- by the
> router's periodic fleet snapshot and by an independent scraper of each
> engine's `/metrics` -- and the two agree exactly (194 and 110). The counter
> is a pod-wide aggregate, so the two engines on a pod report the same number
> and each pod contributes one delta. *Mean engine queue wait* is
> `request_queue_time_seconds` sum over count, the time a request sits in
> the engine's own queue before its first schedule. Neither contains a second
> of sandbox time.
>
> Significance: a Poisson rate-ratio test on 304 events with generated tokens
> as exposure gives z = 4.38, p = 1.2e-5. The earlier pair (164 vs 122 on
> 2.08M vs 2.33M tokens) gives z = 3.46, p = 5e-4.

> **NOTE — why we do not lead with rollout wall clock or verl's throughput.**
> Rollout wall clock (`timing_s/gen`) equals the slowest trajectory's time,
> and on this workload that trajectory is mostly sandbox time: in the 7B
> audit the same clock swung by +/-20% on tool luck alone and produced a
> throughput claim we had to retract. Here it reads +6.0% for an arm that
> generated 8.4% more tokens, which is a per-token wash and says nothing.
> verl 0.9's legacy trainer (`main_ppo_v0`, needed because its V1 trainer
> requires a package the image lacks) does not emit the per-trajectory
> `generate_sequences` and `tool_calls` timings that let us separate
> generation from tool time at 7B. So throughput is measured at the engines:
> decode tokens per second of engine time with at least one request running.

> **NOTE — two throughput numbers disagree in sign, and both are right.**
> Decode throughput per busy engine-second is down 8.5%. All tokens
> processed per busy engine-second is up 10.5%. The gap is prefill: the
> flow-control arm prefilled 17.8% more prompt tokens, partly because its
> responses were 8.4% longer and partly because its prefix-cache hit rate
> fell. Whether that extra prefill is "work done" or "work wasted" is the
> whole question, and the section below prices it. We report decode
> throughput as the cost because it is the number that does not credit
> recomputation.

> **NOTE — occupancy above the threshold is higher under the gate, by
> design.** 32.3% of samples at KV >= 0.90 against 27.0%. The gate releases
> a waiter the moment an engine reads below 0.90, so it holds engines *at*
> the threshold. What it prevents is the overshoot: time at 0.99+ halved.

> **NOTE — preemptions fell but did not vanish.** 110 remain. The gate acts
> on a 100 ms-old snapshot and admits into engines whose resident requests
> are still growing; it cannot see decode growth already committed.

---

## Where the waiting went

Under the gate a request waits in one of two places: at the router before
admission, or in the engine's queue after it. The engine side is measured.

| | requests | mean engine queue wait | total engine queue time |
|---|---|---|---|
| unthrottled | 1,883 | 3.78 s | 7,124 s |
| flow control | 1,923 | 0.98 s | 1,888 s |

The gate removed about 5,200 request-seconds of engine queueing and parked
13,670 times to do it. Park durations are not recorded per request, so we
cannot say whether the total wait fell or merely moved; the rollout wall
clock (+6% for +8% more tokens) suggests moved. What did change is *where*
the request waits: in the engine's queue it holds a KV reservation and
contributes to the next eviction; at the router it holds nothing.

---

## Where the prefill went

Every generated token in this workload carries about 7 prompt tokens of
context. Two things decide how many of those the GPU actually computes: the
prefix cache, and preemption.

| | prompt tokens | prefix-cache hit rate | preemptions | prompt tokens per decode token |
|---|---|---|---|---|
| unthrottled | 11.49M | 30.7% | 194 | 6.67 |
| flow control | 13.54M | 24.2% | 110 | 8.27 |

The flow-control arm prefilled 2.04M more prompt tokens. Its responses were
8.4% longer, which accounts for roughly 0.97M of that at the unthrottled
arm's ratio. The remaining ~1.07M is the cache-miss penalty: a parked
trajectory returns later, and under saturation its prefix has been evicted
in the meantime, so the turn re-prefills from scratch.

Set against that, the 84 avoided preemptions each saved one re-prefill of a
resident context. At the arm's ~10k-token mean context that is roughly
0.84M prompt tokens not recomputed.

**The gate traded ~0.84M tokens of preemption recompute for ~1.07M tokens of
cache-miss recompute.** On GPU work it is a wash, slightly negative. This is
the mechanism behind the 8.5% decode-throughput cost: busy engine-seconds
went to prefill that the unthrottled arm's warmer cache did not need.

---

## Cost of an avoided preemption

This puts a price on one avoided preemption, from measured values only.

```
extra prompt tokens          13,537,253 - 11,493,364  =  2,043,889
  of which response growth   (8,927 / 8,236 - 1) x 11,493,364  =  ~967,000
  attributable to the gate                              =  ~1,077,000
avoided preemptions          194 - 110                =  84

1,077,000 / 84  =  ~12,800 extra prefill tokens per avoided preemption
```

A preemption re-prefills the evicted request's context, ~10,000 tokens here.
So the gate paid about 1.3 tokens of prefill for every token it saved.

**What the number is.** The exchange rate between the two kinds of recompute
under this gate at this load. It is roughly 1, which is why every throughput
measurement across the 7B and 32B campaigns came out near a wash.

**What it is for.** Deciding whether to enable the gate. If preemptions are
merely recompute, the gate is close to free and close to useless. If they
are something worse -- a latency-SLO breach, a cascade into the deadlock
attractor this stack has hit before, or an engine whose eviction path is
costlier than re-prefill -- the gate removes 40% of them for a ~1:1 token
trade and a 4x cut in engine queue wait.

**What to be careful about.** One step per arm. The response-length
adjustment assumes prefill scales linearly with response length, which
holds only on average. The 10k-token context is the arm's mean, and
evictions are biased toward the longest residents, so 0.84M is a floor and
the exchange rate may be closer to 1:1 than 1.3:1.

---

## What we cannot explain yet

**Both flow-control arms generated more.** +12% response tokens in the first
pair, +8.4% in the second. The tasks, generations and seed are identical,
so the trajectories should be statistically alike. Two candidates: fewer
preemptions mean fewer requests resumed from a truncated state, or lower
engine queueing lets more trajectories reach their natural end before the
turn cap. Either would be a real effect of the gate on the *content* of a
rollout, not only its cost. The cheap test is a two-step pair with the
per-trajectory turn counts recorded.

**Whether total request latency fell.** Engine queue wait fell 74% but park
time at the router is unrecorded. The hook should log park duration per
request; until it does, "moved" is the honest reading.

---

## Files

| file | contents |
|---|---|
| `metrics.csv` | every number in this document, both pairs, with source and direction |
| `logs/pair2_baseline-32b.driver.log.gz`, `logs/pair2_gate-kv90-32b.driver.log.gz` | raw verl driver logs for the instrumented pair. Source of every training-loop metric and the router's fleet snapshots |
| `logs/pair2_engine_scrapes_pod-*.tsv.gz` | raw vLLM `/metrics` counters and gauges per engine, sampled every 15 s, one file per worker pod. Source of every engine metric |
| `logs/pair1_*.driver.log.gz` | raw driver logs for the earlier pair (preemptions and verl metrics only; no engine scraper ran) |
| `perf_window.py` | regenerates every engine-side number from the scrapes for a time window |
| `engine_scraper.sh` | the scraper, restricted to ports owned by vLLM server processes |
| `sweep32b.sh` | the exact run script; arms are `integration/verl/examples/runtime-env-32b-{off,on}.yaml` |
