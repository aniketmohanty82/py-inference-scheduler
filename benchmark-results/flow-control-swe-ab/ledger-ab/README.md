# Ledger A/B: the preemption pathology was the integration, not the policy

Rerun of the threshold sweep's baseline and kv-0.90 arms after fixing two
verl-integration bugs found while investigating why gated arms preempted as
much as (or nominally more than) the ungated baseline.

## TLDR

Preemptions went from **713-746 per arm to 0 in both arms**. Neither of the
two suspects -- stale vLLM metrics, or a gate that concentrates load -- was
the cause. The cause was the integration fragmenting the scheduler across
verl's 8 AgentLoopWorker actors: each worker held a private inflight count
covering 1/8th of the fleet's dispatches, and each could bootstrap a PARTIAL
endpoint view (1 or 3 of 4 engines) that it kept for the whole run, pinning
every one of its trajectories to whatever engines it happened to see. With
both fixed, load spreads evenly (max engine share 52.6% -> 28.8%), the fleet
stays cool (kv p99 ~0.5), and the saturation gate has almost nothing left to
do: 0 parks and 8 drops across ~11,000 decisions, with a throughput delta
(-4.3%) inside this workload's demonstrated 33% run-to-run variance.

## The two bugs

| Bug | Mechanism | Consequence | Fix |
|---|---|---|---|
| Fragmented inflight counts | verl fans each batch over 8 AgentLoopWorker actors; the hook built an independent `_SchedulerCore` (and `InflightStore`) in each, so `queue_len` covered only that worker's dispatches | `least_queue` balanced each worker's 1/8th slice while fleet-wide stacking stayed invisible | shared inflight ledger: a named Ray actor all workers publish to and read from (`76070f5`) |
| Partial endpoint views | the one-shot bootstrap drain raced incremental engine registration at startup and was cached forever | a worker with a 1-engine view routes ALL of its trajectories to that engine unconditionally; measured in every earlier run (old baseline: 31 one-engine / 105 three-engine / 81 four-engine FLEET lines) and a direct cause of the 52.6% single-engine share | re-drain on every request until the view covers `get_all_servers` (`baed7e0`) |

Observability fixed alongside: Ray collapses identical stdout lines across
actors, so all prior park/drop/FLEET counts were undercounts. Runs now set
`RAY_DEDUP_LOGS=0` and every hook print carries the worker pid (old runs
logged ~670 FLEET lines per arm; these log ~3,400 for the same duration).

## Setup

Identical to `../threshold-sweep/` (run_swe.sh: 7B + LoRA r32, tp=2, 4
engines, gmu 0.22, 64 tasks, batch 64 x n4 = 256 trajectories, 25 turns, 2
steps; `least_queue` routing in both arms; sandbox pool reset before each
arm) except both arms carry the two code fixes above. Arm configs are
byte-identical to the sweep's `lq-off.yaml` and `lq-kv90.yaml` (gate: kv
0.90, waiting 4), so within this pair the gate is the only variable.

Both workers' views verified 4/4 at run time in both arms (per-pid "endpoint
view" lines), and all 8 ledger init markers present -- the shipped code is
what ran.

## Results

Tokens = 256 x sum of `response_length/mean` over steps; throughput = tokens
/ sum of `timing_s/gen`. Preemptions are the within-arm delta of the
pod-cumulative counter, cross-checked by a hook-free engine scraper.

| Arm | Parks | Drops | Gen wall | Tokens | tok/s | vs base | Preempt | Max share | FLEET kv p50/p99/max | kv>=0.9 samples |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-v2 | 0 | 0 | 2709.0 s | 4.453 M | 1,644 | — | **0** | 28.8% | 0.05 / 0.47 / 0.92 | 4 / 3344 |
| gate-kv90-v2 | 0 | 8 | 2826.0 s | 4.446 M | 1,573 | -4.3% | **0** | 29.1% | 0.04 / 0.49 / 0.68 | 0 / 3439 |

Old sweep, same configs on the broken integration, for reference: 713 / 733
preemptions, 52.6% max share (baseline), 2,213 / 1,985 tok/s.

Work is comparable across arms: 43.6 / 43.6 vs 43.6 / 44.1 turns per
trajectory, token totals within 0.2%.

## Analysis

**A. The preemptions belonged to the rig, not the policy.** Zero in both
arms, confirmed by two independent instruments: the within-arm
`num_preempted` delta (746 -> 746 against the pod-cumulative offset) and the
scraper's counter (flat at 746.0 across 656 + 660 samples). Every earlier
conclusion drawn from preemption counts -- including "the gate raises
preemptions" -- was noise on top of these bugs.

**B. Concentration is cured in the baseline itself.** 28.8 / 25.1 / 23.2 /
22.9% selection shares (was 52.6% max). The gate arm is the same (29.1%
max). This was hypothesis (2)'s observable, and it traced to partial views,
not to the gate.

**C. The gate under a healthy fleet is almost inert -- and correct when it
acts.** 8 drop events total: 7 on the waiting threshold (values 4-5), and
exactly one genuine saturation moment where 3 engines read kv=0.91, were
dropped, and the 4th took the request. That is the designed behavior,
occurring once in ~11,000 decisions because even spreading prevents the
pressure the gate exists to relieve.

**D. The dispatch-lag window is real but harmless at this scale.** The
ledger captured it live: one snapshot shows 77 requests committed to an
engine (q77) while every engine gauge still read r0 / kv~0.02 -- the burst
is admitted before any engine metric can move. With even spreading the fleet
absorbs it (engine-side running peaked at 32, kv max 0.92 for one 15s
sample) without a single preemption, so gating on the engine's real kv
metric suffices here; no router-side reconstruction of kv is warranted on
this evidence.

**E. Throughput deltas remain unresolvable at n=1.** -4.3% for the gate arm
is well inside the 33% spread two identical earlier baselines showed. Gen
walls are also longer than the old sweep's (2,709 vs 1,968 s baseline);
cross-sweep wall comparisons are confounded by that same variance and by the
fixes themselves, and are not evidence of anything.

## Scrutiny

- One run per arm. The categorical result (746-per-arm -> 0 preemptions) is
  far outside any noise band; the throughput deltas are not.
- The FLEET `p` field reads 0 briefly at engine boot before the shared
  multiproc registry loads, then the pod-cumulative 746; within-arm deltas
  use the steady segment, and the scraper agrees.
- Both arms carry both fixes, so old-vs-new differences include the fixes;
  gate effects must be read only within this pair.
- Raw artifacts: `/work/sweep2/*.FINAL.log`, `/work/sweep2/STATUS.txt` on
  the `swe-fc` head pod; scraper tsv archived off-pod.
