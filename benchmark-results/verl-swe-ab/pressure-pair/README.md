# verl-native SWE A/B - high-KV-pressure pair (32B + LoRA)

Store-vs-recompute on verl 0.8.0's native agent loop, 2026-09-09.
Qwen2.5-32B-Instruct + LoRA r32/a32 (fsdp2, dynamic-bsz 16384), 12 GRPO steps
per arm, batch 16 x n 4 (64 trajectories/rollout), seed 42, single 8xH200
node (same worker pod both arms), tp=2 (4 engines), gmu 0.317,
`free_cache_engine=False` BOTH arms, gVisor sandboxes (pool pinned at 6
nodes, autoscaling off, whole pair), dataset baked in image `swe2` (256
rows, fresh tasks every step), 1TiB store, arms differ by exactly the
kv_transfer_config flags. Every flag verified on the live Hydra command
line before each arm ran.

**The pressure regime this pair exists for (the 7B pair recorded zero
preemptions): KV pool squeezed to 2,125 blocks = 34k tokens per engine
(recorded `cache_config_info`; vLLM's floor for the 32k max_model_len is
2,048), gated by a 2-step store smoke requiring recorded preemptions > 0.**

## Validity (all recorded, all pass)

| gate | recompute | store |
|---|---|---|
| steps completed / rc | 12/12, rc=0 | 12/12, rc=0 |
| wall-clock audit (sum step times vs wall) | 8,289s vs 8,548s (259s init) | 8,915s vs 9,219s (304s init) |
| engine-counter continuity | max stall 2.0 min over 142 min | no stall > watcher threshold |
| actor/entropy range | 0.183-0.245 | 0.179-0.331 |
| num_turns/mean range | 37.3-44.6 | 38.7-46.4 |
| rollout-error / OOM / ENGINE_DEAD | 0 / 0 / 0 | 0 / 0 / 0 |
| by_source identity | sums EXACTLY to prompt_tokens_total | same |
| PRESSURE: num_preemptions_total | **39** (11/11/14/3 per engine) | **47** (12/19/8/8) |
| PRESSURE: kv_cache_usage_perc max | **0.992** | **0.9995** |

The wall-clock audit is new to this pair: two store attempts before the
valid one silently froze mid-run while their driver logs kept growing
(connector bug, section below), so validity now requires recorded step time
to account for the wall clock and engine counters to advance continuously.

## Results (per-step means over 12 steps; deltas store vs recompute)

| verl metric | recompute | store | delta | store lower |
|---|---|---|---|---|
| timing_s/agent_loop/generate_sequences/mean | 46.31s | 40.10s | **-13.4%** | **9/12** |
| timing_s/agent_loop/generate_sequences/max | 195.0s | 179.3s | -8.0% | 8/12 |
| timing_s/gen | 636.0s | 687.9s | +8.2% | 5/12 |
| timing_s/step | 690.8s | 742.9s | +7.5% | 5/12 |
| timing_s/agent_loop/tool_calls/mean | 131.5s | 129.8s | -1.3% | 6/12 |
| response_length/mean | 7,521 | 7,582 | +0.8% | - |
| num_turns/mean | 41.5 | 41.7 | +0.6% | - |
| critic/score/mean | 0.0575 | 0.0563 | -2.1% | - |

tool_calls/mean lands at -1.3% with 6/12 - pairing symmetry passes (the
invalid plain-pod pair failed this at -35%, 12/12); it is store-independent
by construction, which is what makes it the symmetry check.

NOTE (rollout wall is not the instrument here): per-step
`timing_s/gen` equals the single slowest trajectory's time to within 0.2-4s
every step, in both arms (corr = 1.000), and that trajectory is 92-97% TOOL
time - an agent stuck burning the 60s no-egress command timeout for up to
~19 consecutive turns. timing_s/gen is a max over 64 heavy-tailed sandbox
draws (recompute's own steps range 386-1,206s, sd/mean ~50%); the store can
only touch the ~30-60s of LLM time inside it. The +8.2% at 5/12 is
straggler lottery, not a serving result; the store-sensitive instruments
are the generate_sequences rows and the compute split below.

## Serving split (final cumulative counters, 4 engines, identity EXACT)

| source | recompute | share | store | share |
|---|---|---|---|---|
| local_compute (true prefill) | 15,225,359 | 21.87% | **4,442,672** | **6.34%** |
| local_cache_hit | 54,391,312 | 78.13% | 54,921,184 | 78.39% |
| external_kv_transfer | 0 | 0% | **10,696,528** | **15.27%** |
| TOTAL (= prompt_tokens_total) | 69,616,671 | | 70,060,384 | |

Workloads matched to +0.6% total prompt tokens. Local cache-hit share is
IDENTICAL across arms (78.1% vs 78.4%): the external tier is not
cannibalizing the local cache - it serves almost exactly the tokens that
eviction pressure forces recompute to re-prefill. **True prefill compute:
15.2M vs 4.4M tokens = 3.4x reduction.**

Mooncake ops (store arm, all ok, zero failed keys): save_put 223,458
(3.9ms/op mean, 878s total across the run), load_get 6,438 at 42.3ms/op
mean, 1,661 tokens/get (DERIVED) ~= 25ms per 1k tokens pulled (DERIVED).
Master postrun counters: the 1TiB tier FILLED during the run - 9
successful eviction rounds, 224,631 keys / 471GB evicted (total written
~1.3TB, DERIVED as evicted + resident). The run stayed healthy through
all of it (zero failed keys, zero engine errors) - first recorded
evidence that master eviction is non-fatal on this stack (the 08-31
except-path connector fix is what made failed/evicted saves survivable).
Consequence: the 15.27% external share is partly churn-capped - keys
evicted before their reuse window are lost rescues.

## Interpretation (S5)

The low-pressure 7B pair measured the store harmless (timing_s/gen -3.8%,
generate_sequences/mean -5.9% at 10/12) with nothing to rescue (external
0.22%). This pair gives it something to rescue and the hypothesis -
pressure widens the generation-side delta - holds:
generate_sequences/mean improves -5.9% -> **-13.4%** (9/12), and the
external tier share goes 0.22% -> **15.27%** at unchanged local hit rate.
generate_sequences/max also improves (-8.0% at 8/12) but less than at low
pressure (-20.9% at 7/12): under preemption, resumed store-arm trajectories
pay pull latency on refill (one 477s generate_sequences/max landed in the
step-2 preemption burst), which eats into the tail win. timing_s/gen is a
wash (+8.2% at 5/12) because this workload's makespan is sandbox-timeout
bound, not generation bound - the store's measured value under pressure is
compute (3.4x less true prefill) and per-trajectory generation latency,
not step wall clock. Quality metrics are indistinguishable (turns +0.6%,
response length +0.8%, score -2.1%, entropy bands overlapping).

## Store-arm connector fix (two discarded wedged attempts)

Store attempts 1 and 2 froze silently mid-run (after steps 4 and 1
respectively): one engine pinned at ~98% KV usage with zero running
requests, all processes idle, counters bit-identical for hours, no error
logged anywhere. Root cause (read from the vendored vLLM mooncake store
connector, `store/worker.py::_get_and_clear_finished_sending`): preemption
deletes a request's save-accounting entry; if the request then finishes
before any post-resume save recreates it, the finish loop's `None` case
emits no done-signal - while the scheduler delay-frees the request's
blocks on pre-preemption `num_saved_tokens > 0` and waits forever (vLLM
has no timeout on connector-delayed frees). Leaked leases accumulate until
the engine starves. Sibling of the 08-31 pb2 fix (which repaired the
exception path; this hole is in the success path and raises nothing).
Short SWE turns (~175 tokens) make finish-before-new-save common; fat
pullbench requests never hit it.

Fix: 3-line patch (this dir, `connector_preemption_orphan_patch.py`) -
remember preemption-cleared request ids, emit the done-signal for them at
finish, log each rescue. Applied to the worker pod's vLLM install; the
valid store arm ran with it. **The rescue path never fired in the valid
run (0 warnings at 47 preemptions)** - the wedge did not recur but the
patched branch is validated by the static argument above, not by in-run
observation; a deterministic unit test + image-baked patch + upstream vLLM
report are follow-ups. The recompute arm never loads connector code, so
the one-flag-diff invariant is unaffected. Wedge evidence (py-spy stacks
incl. thread locals, per-minute pin timelines, driver logs) is retained in
operator scratch, not in this dir (invalid runs stay out).

## Measurement caveats

- Store arm ran ~11.5h after recompute (two wedged attempts between) on
  day-warmed gVisor nodes; recompute step 1 paid cold task-image pulls
  (1,206s gen vs its own 636s mean). Tool-time symmetry over 12 steps
  (-1.3%, 6/12) says the pairing survived; step-level gen deltas at steps
  1-4 should not be read individually.
- Store keys carry no weight version and LoRA weights change per step;
  fresh-tasks-per-step confines cross-step reuse to shared prefixes.
  Entropy (0.179-0.331 vs recompute 0.183-0.245) and score (-2.1%) show no
  drift signature; per-step external-hit split was not recorded
  (run-level finals only), so stale-reuse remains bounded-by-design, not
  measured-per-step.
- verl's `agent_loop/*/num_preempted` is -1 on this stack; preemption
  counts come from engine `/metrics` (recorded source).
- Sandboxes have no network egress; model `pip install` attempts burn the
  60s command timeout in both arms equally and drive the straggler
  variance described above.
- Engine scrape ports probe the mooncake RDMA handshake listener too,
  producing benign `SocketHandShakePlugin: malformed json` noise in engine
  logs throughout (both arms' scrapers identical).

## Files

- `recompute_driver.log.gz`, `store_driver.log.gz` - full driver logs
  (verl per-step metric lines; the A/B table's source).
- `engine_scrapes.log.gz` - per-minute in-worker engine /metrics samples
  (by_source, preemptions, KV usage; smoke + both arms, SNAP-timestamped).
- `sidecar_final.log.gz` - netns-fixed sidecar archive tail (final
  cumulative counters incl. mooncake op-time histograms).
- `master_postrun.txt.gz` - mooncake master counters after the store arm.
- `connector_preemption_orphan_patch.py` - the exact patch the store arm's
  engines ran (exact-anchor replacement, self-verifying).
