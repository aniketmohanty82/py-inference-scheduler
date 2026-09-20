# Threshold sweep at storm pressure: the gate is robust, the axes are not equal

Four same-pod arms at gmu 0.185 x 4 steps (batch 64 x n4), varying only the
gate config: no gate, kv 0.90/w4, kv 0.90/w2, kv 0.80/w4. Run 09-20 after
the rig's pods were replaced -- which turned out to matter enormously (see
the regime note).

## TLDR

At storm-level pressure the gate cuts preemptions at EVERY setting tried:
-44% (kv90/w4), -30% (kv90/w2), -46% (kv80/w4) versus the same-pod
ungated baseline, each >= 5 sigma on counter-verified counts (509 baseline
events). The KV axis saturates -- kv 0.80 and 0.90 land within 0.4% of
each other per Mtok -- while tightening the waiting axis (w4 -> w2) makes
things WORSE, not better. Recommended setting stands at **kv 0.90 / w 4**:
same protection as kv 0.80 with 28% fewer parks. Per-arm throughput
comparisons are unresolvable at one run per cell (see the coupling note);
no throughput claim is made in either direction.

## Results (all arms same pod, same day, sandbox pool reset per arm)

| Arm | Preempt | Preempt/Mtok | vs base | z | Parks | Drops | Tokens | Turns |
|---|---|---|---|---|---|---|---|---|
| baseline (no gate) | 509 | 56.9 | — | — | 0 | 0 | 8.947 M | 172.5 |
| gate kv 0.90 / w 4 | 294 | 31.8 | **-44%** | 8.0 | 26,711 | 7,453 | 9.253 M | 173.1 |
| gate kv 0.90 / w 2 | 352 | 39.7 | -30% | 5.2 | 44,934 | 10,034 | 8.869 M | 175.5 |
| gate kv 0.80 / w 4 | 274 | 31.7 | **-46%** | 7.9 | 37,105 | 7,698 | 8.636 M | 171.6 |

Preemption deltas are within-arm counter ranges, matched exactly by the
hook-free scraper. Every gated arm's work is within +-3.5% of baseline on
tokens and +-1.7% on turns. Drop-reason mixes confirm the axes: kv80/w4 is
kv-dominated (14,660 kv vs 1,642 waiting), kv90/w2 waiting-heavy (8,459
waiting-triggered).

## Analysis

**A. The gate's preemption effect is now beyond argument on this rig.**
Three settings, three reductions, all >= 5 sigma against a 509-event
baseline, replicating the direction of both earlier pairs (22->12, 53->38)
at 10x the statistical weight.

**B. The axes are not symmetric.** Lowering kv 0.90 -> 0.80 changes
nothing (31.8 -> 31.7/Mtok): by the time an engine reads kv 0.80 under
storm dynamics it reaches 0.90+ before the next decision anyway, so the
earlier trigger only buys more parks (37k vs 27k). Tightening waiting
4 -> 2 is counterproductive (39.7/Mtok): it parks on queue depths the
engines could absorb, delaying turns without protecting the pool --
engine-side waiting converts to preemption only through KV growth, which
the kv threshold already covers.

**C. Throughput is NOT resolvable per-arm, including one tempting number.**
kv80/w4 shows +45% LLM-side tok/s over baseline -- but the same arm shows
+75% tool time, and arrival density physically couples the two: slow tool
calls thin engine load, which empties queues and deflates
generate_sequences regardless of the gate. The instrument is
per-metric-independent but not system-independent. Symmetrically, kv90/w4's
apparent -21% carries the densest arrivals. One run per cell cannot
separate these; a replicate grid could.

## The regime shift (major reproducibility caveat)

Between 09-18 and 09-20 both rig pods were replaced (spot preemption +
head restart). The identical baseline config went from **53 preemptions
(6.4/Mtok, ~750 s/trajectory LLM time)** on the old pods to **509
(56.9/Mtok, ~175 s/trajectory)** on the new ones: ~4x faster per-turn
generation densifies turn overlap, which multiplies pool pressure ~9x --
the bistability sizing law in action. Suspected cause: the RayCluster pins
a MUTABLE image tag (`rllm-verl-mooncake:swe10`); new pods resolved digest
`sha256:0575431f...`, and the old pods' digest was never recorded, so a
silent repush cannot be distinguished from node variance. Consequences:
cross-pod-generation comparisons are void (this sweep is internally
same-pod for that reason), and future rigs should pin image DIGESTS.

## Instrument corrections logged this sweep

| Trap | Symptom | Fix |
|---|---|---|
| FLEET kv distributions under parking | p50 read 0.55-0.75; scraper's uniform sampling shows 0.14-0.15 | the flow-control watcher polls at 10 Hz while parked, oversampling saturated instants ~10x; use the scraper for distributions, FLEET only for counts/views |
| Poller match string | results sat 3 h | completion pollers must match the orchestrator's actual finish line |
| Timing metrics under tool-luck | +45%/+75% co-movement | arrival density couples sandbox luck into every timing metric; only counters are luck-proof at n=1 |

## Scrutiny

- One run per cell; preemption conclusions rest on counters and large
  counts, throughput conclusions are withheld.
- Hit rates: 3.4% / 2.6% measured on the two sweep arms -- the dead-cache
  regime of the 45k pool, unchanged.
- Raw logs pulled off-pod this time (`sweep6-logs/` in the session
  archive); pod storage is ephemeral and already ate sweeps 2-5's raws.
