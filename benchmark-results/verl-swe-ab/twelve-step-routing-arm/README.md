# Our router vs verl's balancer, local-only KV — SWE-bench agent RL, Qwen2.5-32B, 12 steps

## TLDR

Routing alone, with no KV tier, did not beat verl's own balancer. Over 12
training steps our router's **sampling time per trajectory landed within 4%**
of verl's, in our favour in 3 of 12 steps, and **per assistant turn the two
are identical** (14.35 s against 14.40 s). Rollout time per step read 7%
lower, in our favour in 9 of 12 steps, but the 95% interval on that spans
-22% to +8%, and most of the gap is which arm drew the stuck sandboxes. **On
this workload, with local-only KV, our router has no measurable edge over
verl pinning each trajectory to one engine.**

The run also measured the premise the profile was designed on, and it was
wrong. With all four engines read at the same instant, an engine was idle
while another had a queue in 3 of 3,677 busy snapshots. The earlier
per-engine scrape that showed it in 14 of 24 read the engines minutes apart.

| | verl balancer | our router | ours vs verl |
|---|---|---|---|
| sampling time per trajectory | 169.3 s | 176.1 s | +4.0% |
| sampling time, slowest trajectory | 811.7 s | 801.1 s | -1.3% |
| rollout time/step (`timing_s/gen`) | 1,095.8 s | 1,019.5 s | -7.0% |
| prompt tokens recomputed on GPU | 279.3M | 319.6M | **+14.5%** |

---

## What we tested

verl routes each trajectory to the engine with the fewest requests in flight
at its first turn and pins it there for life. It never looks at engine load
again, and it never moves a trajectory off a queue. We theorized that a router
which sees every engine's queue and KV occupancy, keeps a trajectory with its
KV, and moves it only when its engine is saturated, would finish rollouts
sooner.

One question, one arm against the local-only arm of the twelve-step
three-arm run: **does load-aware routing beat pinning when there is no KV
tier?** The arms differ by routing alone. Same node, same engines, same
sandbox pool, no `kv_transfer_config` in either.

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
| KV tier | none in either arm |
| sandbox pool | 21 x e2-standard-16, gVisor |

| arm | routing | what it is |
|---|---|---|
| verl balancer | `GlobalRequestLoadBalancer` | least in-flight at the first turn, then pinned to that engine for the trajectory's life. This is the local-only arm of the three-arm run, unchanged |
| our router | `PyInferenceAgentLoopManager` + `configs/swe-backpressure.yaml` | the profile below, fed by every engine's `/metrics` and a fleet-wide in-flight ledger shared by the 8 agent-loop workers |

The verl arm ran on 2026-09-22 and ours on 2026-09-23, on the same node type,
the same pods recreated from the same image line, and the same 21 sandbox
nodes.

### Workload

Identical to the three-arm run: 128 SWE-bench tasks x 4 generations, so
**512 trajectories per rollout**, 12 steps, `gpu_memory_utilization` 0.38,
15 s shell command timeout, 32-turn cap.

| | verl balancer | our router |
|---|---|---|
| turns per trajectory | 24.5 | 25.6 |
| response length per trajectory | 5,138 | 5,457 |
| tokens per training batch | 2,896,195 | 3,059,492 |
| prompt length | 518.2 | 518.9 |

The workloads are not as tightly matched as the three arms were. Our arm
sampled 4% more turns and 6% more tokens per step. Sampling is stochastic and
the trajectories diverge, so the rows below that scale with work are also
shown per turn.

---

## How our router decides

Every turn of every trajectory is a routing decision. The profile
(`configs/swe-backpressure.yaml`) runs these in order.

| | plugin | what it does | why |
|---|---|---|---|
| 1 | `saturation` filter, queue >= 8 or kv >= 0.98 | removes a saturated engine from the ballot. Falls through when every engine is over | the only way a trajectory leaves the engine holding its KV |
| 2 | `request_affinity`, weight 8.0 | votes for the engine that ran this trajectory's previous turn. No vote on a first turn | the context's KV is there; anywhere else re-prefills it. Weight above the load scorers' 7.05 so the holder wins whenever it is a candidate |
| 3 | `waiting_queue` 4.0, `kv_cache` 2.0, `least_queue` 1.0 | place first turns and migrating turns by engine queue, KV occupancy and fleet-wide in-flight count | the load awareness verl does not have |
| 4 | `jitter` 0.05 | breaks exact ties | otherwise engine 0 wins every tie |

Not `sticky_session`: it hashes the id to a fixed home engine and after a move
would send the trajectory back to an engine that no longer holds the recent
context. Not `prefix_cache`: it hashes the prompt body, so all 512 first turns
match the shared system prompt on whichever engine saw it first.

Three things had to be true for the measurement to mean anything, and the
run's own instruments checked each. The 8 workers saw all 4 engines (a partial
view pins a worker to a subset). The in-flight ledger was fleet-wide: its sum
matched the engines' own running plus waiting to within 4% (a private ledger
sees one eighth). And the engine metrics were per engine: in 0 of 593 busy
snapshots at the step-1 gate did all four report the same numbers. The first
two attempts failed the third check and were stopped; see "Files".

---

## Results

12-step means.

| | verl balancer | our router | ours vs verl |
|---|---|---|---|
| **sampling time per trajectory** | **169.3 s** | **176.1 s** | ⚪ +4.0% |
| sampling time per assistant turn | 14.40 s | 14.35 s | ⚪ -0.3% |
| **sampling time, slowest trajectory** | **811.7 s** | **801.1 s** | ⚪ -1.3% |
| rollout time/step (`timing_s/gen`) | 1,095.8 s | 1,019.5 s | ⚪ -7.0% |
| prompt tokens recomputed | 279,253,366 | 319,626,641 | 🔴 +14.5% |
| local prefix-cache share of prompt tokens | 35.7% | 30.0% | 🔴 -5.7 pts |
| tool-call time per trajectory | 45.9 s | 52.0 s | 🔴 +13.2% |
| tool-call time per assistant turn | 3.91 s | 4.24 s | 🔴 +8.4% |
| `perf/throughput` from verl, for reference | 270.1 | 299.9 | 🟢 +11.0% |
| preemptions over the run | 408 | 431 | ⚪ +5.6% |
| policy entropy | 0.2037 | 0.2009 | ⚪ -1.4% |
| decode tokens produced | 7,400,972 | 7,701,754 | ⚪ +4.1% |

🟢 our router did better · 🔴 it did worse · ⚪ neither: inside noise, or
describing the workload rather than scoring it. Direction is not always "lower
is better": throughput and decode tokens are better higher, everything else
better lower.

> **NOTE — the two headline metrics.** Same definitions as the three-arm run.
> *Sampling time per trajectory* is
> `timing_s/agent_loop/generate_sequences/mean`, the time one trajectory spends
> inside generate calls summed over its turns and averaged over all 512.
> *Slowest trajectory* is that quantity's maximum. Paired over the 12 steps,
> the per-trajectory difference is +6.8 s with a 95% interval of -44 to +58 s,
> t = 0.29. That is a tie by any reading, and per assistant turn the two arms
> are the same to 0.3%.

> **NOTE — rollout time/step reads 7% lower, and most of it is the sandbox
> lottery.** Our router won 9 of 12 steps on `timing_s/gen`, mean difference
> -76 s, 95% interval -237 to +84 s, t = -1.05. Every step's wall is one
> trajectory's sampling plus tool time, and in five of the twelve steps one
> arm or the other had a wall-setter stuck in shell commands for 900 to
> 1,380 s: verl in steps 5, 7, 8 and 11, ours in steps 1 and 8. Over those
> five steps ours reads -10.2%. Over the other seven, where the wall was set by
> sampling, it reads -3.2%. The stuck trajectories are the same phenomenon in
> both arms (a command that leaves a process behind runs to the 45 s exec
> ceiling every turn until the 32-turn cap) and have nothing to do with
> routing.

> **NOTE — `perf/throughput` is the one row with a significant sign, and it
> is not a routing result.** It is `total_num_tokens / (timing_s/step x
> n_gpus)`. Our arm put 5.6% more tokens into each training batch and spent
> 4.3% less wall clock per step, and the ratio of the two is the +11%. The
> extra tokens are more turns per trajectory, which is sampling variance, not
> something the router did.

> **NOTE — tool-call time is drift, as before.** Our arm ran a day later on
> the same 21 sandbox nodes. Its very first attempt, before any of this
> routing was live, already showed tool time 15% above the verl arm's on the
> same cold step. The router does not touch the sandbox.

---

## What the instruments showed

The FLEET lines read all four engines within milliseconds of each other, 4,252
times over the run, 3,677 of them with the fleet busy. They are the
measurement the profile's premise should have had.

| | value |
|---|---|
| decisions per engine | 25.2% / 25.3% / 24.6% / 24.8% |
| turns kept on the engine holding their KV | 63,205 (88.3%) |
| turns moved by the saturation filter | 8,355 (11.7%) |
| first turns | 6,144 (= 512 x 12) |
| busy snapshots with an engine idle while another had a queue | 3 of 3,677 |
| running-count spread across engines, median / p90 | 0.36 / 0.67 of the mean |
| per-engine KV occupancy, median / p75 / share >= 0.95 | 0.93 / 0.97 / 42% |
| filter state: all four engines over threshold (falls through, behaves as a pin) | 1,846 snapshots (50%) |
| filter state: some over, some under (moves possible) | 1,155 (31%) |
| filter state: none over | 676 (18%) |
| fallbacks to verl's balancer | 0 |
| engine failures | 0 |

**The engines were never idle while others queued.** The 2.1x running-count
spread and the 14-of-24 idle-while-queued snapshots the profile was built on
came from a scrape that reads the four engines one after another, about seven
minutes per sweep, longer than a rollout phase. Read together, the spread is
0.36 of the mean and the idle case does not occur. Under this load the fleet
is uniformly saturated three quarters of the time.

**So the router had nothing to route to, and every move cost a full-context
prefill.** 11.7% of turns migrated. Each migration re-prefills a 5k to 29k
token context on an engine that is itself near full, and it evicts other
trajectories' cached blocks there. That is the -5.7 points of local
prefix-cache hits and the +14.5% recomputed prompt tokens. It did not show up
in sampling time per turn because prefill compute is small against queueing at
this load, the same 33x gap the three-arm run measured, so the extra work was
absorbed. It bought nothing, and it cost nothing visible, which is what a tie
made of two offsetting effects looks like.

**Half the time the profile was verl's pin.** In 50% of busy snapshots all
four engines were over the saturation threshold, the filter fell through, and
`request_affinity` at weight 8 did exactly what verl's balancer does. The
other half is where the two arms differed, and it made no measurable
difference.

---

## Where the design was wrong

Three things, in order of consequence.

**The premise was a measurement artifact.** Idle-while-queued engines were
inferred from a sequential scrape. The simultaneous instrument shows they do
not exist under this load. A profile built to fix them cannot win.

**Moves are not free without a tier.** With local-only KV, the only way to
relieve a queued engine is to move a trajectory and re-prefill its whole
context somewhere else. When every engine is near full that is pure added
work. This is the case for the third experiment on the list, our router with
the KV tier against verl with the tier, where a moved trajectory's context
comes from the tier in milliseconds instead of from the GPU.

**Two of three attempts routed blind.** The first smoke ran with one
`PROMETHEUS_MULTIPROC_DIR` shared by all four engines, so every engine's
`/metrics` served the fleet aggregate and the load scorers saw the same number
everywhere. The first 12-step attempt ran with the variable removed, and our
own engine patch then set it after `prometheus_client` had been imported, so
every `/metrics` came back empty. Both were caught by the gate on the run's
own instruments, not by looking at the results, and the patch now does
nothing but expose the stats RPC. The per-engine attribution check is in
`sched_gate.py`.

---

## Files

| file | contents |
|---|---|
| `sched_driver.log.gz` | raw verl log of the 12-step run, with the RLS / FLEET / AFFINITY instrument lines. Source of every training-loop and routing metric |
| `sched_scrape.log.gz` | raw vLLM `/metrics`, one sweep of all four engines every ~7.5 min. Source of every engine metric |
| `sched_attempt1_blind_metrics_driver.log.gz` | the first 12-step attempt, stopped by the gate after step 1 with every engine metric reading zero. Kept for the record. Not used in any table |
| `gate_step1.txt` | the ten instrument checks the run passed after step 1 |
| `compat_check.txt` | the GPU-free routing check run on the pods before launch: full endpoint view, load-aware routing away from a saturated fake engine, affinity with the filter as the only way off the holder, ledger back to zero |
| `sched.md` | per-step detail for the arm |
| `analyze_routing.py` | the FLEET / AFFINITY / decision analysis behind "What the instruments showed" |
| `sched_gate.py`, `sched13.sh`, `p40_arm.sh` | the gate, the run driver, and the exact run script |
| `../twelve-step-three-arm/recompute*.{md,log.gz}` | the verl-balancer arm, unchanged from the three-arm run |
| `../../../configs/swe-backpressure.yaml` | the profile |
