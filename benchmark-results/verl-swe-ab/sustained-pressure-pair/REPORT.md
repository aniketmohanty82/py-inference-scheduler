# Shared RDMA KV tier vs local-only recompute — verl SWE-bench agent RL, 32B + LoRA, 512 trajectories per rollout

## TLDR

Under KV pressure held across every step, serving evicted prefixes from a
shared RDMA KV tier beats recomputing them. **On-GPU prefill drops 2.27x.
Per-trajectory generation time drops 35.1%, in 4 of 4 steps.**

The mechanism is not saved arithmetic. The avoided prefill is worth about
0.11 s of FLOPs per turn. It is **queue relief**. Prefill and decode share the
same GPUs. At 86% recompute the engine runs far past its knee. Work you remove
from one request shortens the wait for every other request. The measured
amplification is **39x**.

| headline | local-only (baseline) | shared KV tier | delta |
|---|---|---|---|
| `local_compute` tokens (on-GPU prefill) | 148,596,234 | 67,037,475 | **-54.9%** (2.27x) |
| `timing_s/agent_loop/generate_sequences/mean` | 348.2 s | 226.2 s | **-35.1%**, 4/4 steps |
| prefill tokens per turn | 2,503 | 1,104 | **-55.9%** |

Why these two metrics. A rollout is not finished until its slowest trajectory
is, so per-trajectory generation time is what a practitioner feels.
`local_compute` is the only figure here that is a direct count rather than a
timing. **Both are population statistics over 512 trajectories.** That is what
makes them hold up. Analysis (c) covers the wall-clock metrics that do not.

---

## 1. Purpose

Production RL rollouts run **local-only** today. That means vLLM's paged prefix
cache, and a full recompute on eviction. It is the baseline because it is what
runs, not because it is easy to beat. It is a strong baseline whenever the
working set fits. Earlier runs in this program found regimes where the shared
tier buys nothing at all.

This report answers a narrower question. **When the local cache is genuinely
failing, is fetching KV over RDMA cheaper than recomputing it?**

Answering it needs a regime where the baseline has almost no local cache left.
That pressure has to last the whole run, not just the first two steps.
Section 3.3 covers how it was built and gated.

---

## 2. Baseline mechanics

Both arms run the same engine, the same scheduler, and the same local prefix
cache. They differ by exactly the `kv_transfer_config` flags, verified on the
live Hydra command line before either arm started.

### 2.1 Local-only (baseline)

vLLM hashes every KV block. The hash chains:
`H(block_hash[i-1], token_ids_i, extra_keys)`. It matches the longest cached
prefix and prefills the rest on-GPU.

Under memory pressure the scheduler preempts running requests and frees their
blocks. Those requests re-prefill their **entire** context on resume. There is
no second tier to catch them.

### 2.2 Shared RDMA KV tier (`DecodeKVSavingConnector` over Mooncake)

A vLLM v1 KV-connector subclass. It reuses the *same* block hashes, so the
store key is `PoolKey(model_name, tp_rank, pp_rank, group_id, block_hash)`.
Local and remote caches are keyed identically. That is why they compose instead
of competing.

Read from source at the pinned image: `swe12`, vLLM 0.22.1,
`…/kv_connector/v1/mooncake/store/`.

A load fires only if **all five** of these hold
(`get_num_new_matched_tokens`):

| # | condition | rationale |
|---|---|---|
| 1 | the local cache does not already cover the block-aligned prompt | the lookup is a blocking ZMQ hop plus a master RPC inside the scheduler loop; comparing locally first costs nothing |
| 2 | the store holds an unbroken block run from position 0 | `lookup()` counts the longest *contiguous* prefix; a gap truncates the match there |
| 3 | no store flush is armed but unexecuted | matching now returns keys about to be wiped, turning a hit into a failed load |
| 4 | the match exceeds `RLS_MIN_PULL_TOKENS` (1024) | a small pull costs a scheduler round-trip for KV that local prefill regenerates faster |
| 5 | in-flight loads < `RLS_MAX_INFLIGHT_LOADS` (1) | a waiter reserves its **full** context in HBM for the whole round trip |

On a pass, three things happen. The request's full block set is allocated. The
request parks in `WAITING_FOR_REMOTE_KVS`. The transfer is issued from
`get_finished()`, **after** the model forward launches, so it overlaps compute.

A background thread then RDMA-writes straight into the paged cache, at
`base_addr = cache_storage.data_ptr()`. There is no host bounce and no staging
copy. A later step sees `finished_recving` and promotes the request. It then
prefills only the tail the store did not cover.

**A failed pull degrades to recompute** (`get_block_ids_with_load_errors`). That
is why declining at any gate above is always safe.

### 2.3 Config profile, verbatim

Identical in both arms except the connector block, which the baseline omits:

```
data.train_batch_size=128  actor_rollout_ref.rollout.n=4
data.max_prompt_length=4096  data.max_response_length=28672  data.seed=42
actor_rollout_ref.model.path=Qwen/Qwen2.5-32B-Instruct
actor_rollout_ref.model.lora_rank=32  actor_rollout_ref.model.lora_alpha=32
actor_rollout_ref.actor.strategy=fsdp2  actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
actor_rollout_ref.actor.ppo_mini_batch_size=32  actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.rollout.mode=async  actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.38
actor_rollout_ref.rollout.free_cache_engine=False
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=32
actor_rollout_ref.rollout.load_format=safetensors
+actor_rollout_ref.rollout.engine_kwargs.vllm.prefix_caching_hash_algo=sha256_cbor

# store arm only:
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config={
   kv_connector: DecodeKVSavingConnector,
   kv_connector_module_path: py_inference_scheduler.datalayer.connectors.mooncake.decode_save,
   kv_role: kv_both, kv_connector_extra_config: {save_decode_kv: true}}
```

Pod environment (both arms): `SWE_CMD_TIMEOUT_S=15`, `SWE_OBS_MAX_CHARS=20000`,
`RLS_DECODE_SAVE_MIN_BLOCKS=8`, `RLS_MIN_PULL_TOKENS=1024`,
`RLS_MAX_INFLIGHT_LOADS=1`, `PYTHONHASHSEED=0`.

`PYTHONHASHSEED=0` and `sha256_cbor` are load-bearing, not hygiene. The block
hash must be byte-identical across processes. Otherwise a block written by
engine 0 is invisible to engine 1.

---

## 3. Methodology

### 3.1 Hardware and stack

| component | value |
|---|---|
| node | 1 x a3-ultragpu-8g, 8 x NVIDIA H200 (143 GB each) |
| engines | 4 vLLM engines, tp=2 |
| model | Qwen2.5-32B-Instruct + LoRA r32/a32 |
| vLLM / verl / torch | 0.22.1 / 0.8.0 / 2.11.0+cu130 |
| image | `rllm-verl-mooncake:swe12` |
| KV store | Mooncake, RDMA, 8 x 128 GB segments, master `--default_kv_lease_ttl 60000` |
| sandboxes | gVisor pool, 21 x e2-standard-16 |

### 3.2 Workload

Each GRPO step samples a **rollout**: 128 SWE-bench tasks x n=4 generations =
**512 trajectories**. A trajectory is a multi-turn agent episode. The model
emits a bash command. The harness runs it in an isolated gVisor sandbox. The
output comes back as an observation, and the loop repeats. It ends when the
model submits, hits 32 assistant turns, or exhausts its 28,672-token budget.

Each *turn* is a separate engine request carrying the whole accumulated
context. That is why prefix reuse dominates this workload.

| dimension | value |
|---|---|
| trajectories per rollout | 512 (128 tasks x n 4) |
| steps per arm | 4 = exactly 2 epochs of the 256-row dataset |
| turns per trajectory (mean) | 28.95 baseline / 29.65 store |
| tokens presented per turn | ~2,910 both arms |
| prompt length (mean) | 517.5 / 518.8 / 521.1 / 515.2 per step — **bit-identical across arms** |
| KV pool | ~106k tokens/engine, 424k total (gmu 0.38) |
| max single context | 29,200 tokens |

Tokens per turn is derived, so the arithmetic:
`173,441,194 prompt tokens / (512 traj x 115.8 turns) = 173,441,194 / 59,289 ≈ 2,925`.

**Batch must divide the dataset.** 4 steps x 128 = 512 draws, which is exactly
2 epochs of 256 rows. Every task appears twice, identically in both arms. Pick a
batch that does not divide the dataset and verl drops the remainder, then
reshuffles. The task set changes between steps and nothing warns you.

### 3.3 Regime construction and gate

KV pressure in a multi-turn agent loop is **bistable**. A request is one turn.
Between turns a trajectory holds zero allocated blocks, and its KV sits in the
evictable cached tier.

That gives two self-sustaining states. Above ~100% occupancy the cached tier is
squeezed to nothing, so every returning turn re-prefills its whole context,
which keeps occupancy high. Below ~60% the tier survives, turns prefill only
new tokens, and occupancy stays low. A regime that merely *reaches* pressure can
therefore fall out of it mid-run, and its means average two incompatible states.

So the knobs were chosen to leave no low state to fall into, and the result was
gated before either arm ran. `pressure_gate.py` requires **≥60% of the
engine-busy window above `kv_cache_usage_perc` 0.85**, plus preemptions > 0.

It has to be a fraction of the busy window, not a peak. Both arms touch 0.99 at
some point, so peaks do not discriminate. The 0.60 bar is calibrated against a
known bistable run. Its collapsed steps score 0.50 and its healthy steps score
0.00.

| | hot_frac | active snapshots | mean kv | preemptions |
|---|---|---|---|---|
| gating smoke | **0.68** (bar 0.60) | 28 | 0.783 | 179 |
| baseline arm | **0.72** | 69 | 0.802 | 317 |
| store arm | 0.56 | 50 | 0.736 | 325 |

The store arm scoring below the bar is the measurement, not a failure. The gate
exists to prove the *baseline* faced real pressure. An arm that relieves its own
pressure on identical work is the thing under test.

### 3.4 Metrics

| metric | definition | direction |
|---|---|---|
| `prompt_tokens_by_source{local_compute}` | prompt tokens actually prefilled on-GPU | lower better |
| `prompt_tokens_by_source{local_cache_hit}` | served from the engine's paged prefix cache | higher better |
| `prompt_tokens_by_source{external_kv_transfer}` | served from the shared tier. The three sum exactly to `prompt_tokens_total` | higher better |
| `timing_s/agent_loop/generate_sequences/mean` | per-trajectory sum of time inside generate calls, meaned over 512 trajectories. Includes queueing | lower better |
| `timing_s/gen`, `timing_s/step` | rollout / step wall clock. **See NOTE 1 — makespan** | lower better |
| `perf/throughput` | `total_num_tokens / (timing_raw["step"] * n_gpus)`. **See NOTE 2** | higher better |
| `kv_cache_usage_perc`, `hot_frac` | engine KV occupancy; fraction of busy window above 0.85 | context |
| `num_preemptions_total` | scheduler evictions of running requests | lower better |
| `actor/entropy` | policy entropy; validity gate is < 1.0 | context |
| `actor/perf/cpu_memory_used_gb` | host memory — the store's standing cost | lower better |

Engine-side values come from per-minute in-worker `/metrics` scrapes. verl
values come from the driver logs. Nothing here is inferred from a model of the
system.

---

## 4. Results

4-step means. Delta is store relative to the local-only baseline. `n/4` is the
number of steps in which the store's value is the smaller one.

| metric | local-only | shared KV tier | delta | store lower |
|---|---|---|---|---|
| **`local_compute` tokens** | **148,596,234** | **67,037,475** | **-54.9%** | 4/4 |
| `local_cache_hit` tokens | 24,844,960 | 32,874,000 | +32.3% | 0/4 |
| `external_kv_transfer` tokens | 0 | 77,760,400 | — | — |
| `prompt_tokens_total` | 173,441,194 | 177,671,875 | +2.4% | — |
| **`generate_sequences/mean`** | **348.2 s** | **226.2 s** | **-35.1%** | **4/4** |
| `generate_sequences/max` | 1,146.7 s | 883.8 s | -22.9% | 4/4 |
| `tool_calls/mean` | 57.1 s | 59.7 s | +4.5% | 1/4 |
| `tool_calls/max` | 985.6 s | 558.9 s | -43.3% | 3/4 |
| `timing_s/gen` (see NOTE 1) | 1,558.2 s | 970.7 s | -37.7% | 4/4 |
| `timing_s/step` (see NOTE 1) | 1,904.4 s | 1,316.8 s | -30.9% | 4/4 |
| `perf/throughput` (see NOTE 2) | 225.5 | 328.2 | +45.6% | 0/4 |
| `num_preemptions_total` (per step) | 79.3 | 81.3 | +2.5% | 1/4 |
| `kv_cache_usage_perc` (busy mean) | 0.751 | 0.659 | -12.3% | 4/4 |
| `hot_frac` | 0.673 | 0.460 | -31.6% | — |
| `num_requests_running` (peak) | 112.8 | 93.3 | -17.3% | — |
| `num_turns/mean` | 28.95 | 29.65 | +2.4% | 0/4 |
| `response_length/mean` | 6,251.7 | 6,249.6 | -0.03% | 2/4 |
| `perf/total_num_tokens` | 3,466,187 | 3,465,082 | -0.03% | 2/4 |
| `critic/score/mean` | 0.0288 | 0.0303 | +5.1% | 1/4 |
| `actor/entropy` | 0.1955 | 0.3647 | +86.6% | 0/4 |
| `actor/grad_norm` | 0.0023 | 0.0084 | +263% | 1/4 |
| **`cpu_memory_used_gb`** | **164.9** | **1,221.9** | **+641%** | 0/4 |

Workloads matched to 2.4% on total prompt tokens, 0.03% on batch tokens and
response length, and bit-identically on prompt length per step.

### Validity

| gate | baseline | store |
|---|---|---|
| steps completed / rc | 4/4, rc=0 | 4/4, rc=0 |
| wall audit (sum `timing_s/step` vs arm wall) | 7,618 s vs 7,920 s | 5,267 s vs 5,640 s |
| `response/aborted_ratio` | 0.000 | 0.000 |
| by_source identity vs `prompt_tokens_total` | EXACT | EXACT |
| connector errors / `EngineDeadError` / force-fails | n/a | 0 / 0 / 0 |
| Mooncake failed keys, all operations | n/a | 0 |
| metric coverage | 82/82 keys x 4 steps | 82/82 x 4 |

---

## 5. Run logs

| artifact | contents |
|---|---|
| `recompute_driver.log.gz`, `store_driver.log.gz` | full verl driver logs; source of every verl figure |
| `recompute_scrape.log.gz`, `store_scrape.log.gz` | per-minute in-worker engine `/metrics`, SNAP-timestamped; source of by_source, occupancy, preemptions |
| `smoke_driver.log.gz`, `smoke_scrape.log.gz`, `smoke_gate.txt` | the gating run and its verdict |
| `sandbox_fleet.log.gz` | 60 s samples of sandbox CRs and pending pods across both arms |
| `metrics.csv` | every recorded number: 93 metrics x 4 steps x 2 arms, verified bit-exact against the logs |
| `engine_timeseries.csv.gz` | the scrape in long form, one row per (snapshot, engine, metric), with `step` and `phase` stamped on. 17,141 rows |
| `harvest34.py`, `pressure_gate.py` | regenerate every table here; run the gate on any scrape |
| `p34_arm.sh`, `p34_smoke.sh` | the exact run scripts |

---

## 6. Analysis

**(a) The defensible result is prefill work, and it is a count, not a timing.**
`local_compute` falls from 148.6M to 67.0M tokens, a 2.27x cut, with total
prompt tokens matched to 2.4%. The baseline prefills 85.7% of everything it is
shown. The store arm prefills 37.7%.

Scheduling noise, stragglers and measurement choices cannot move this. It is the
engine's own counter, and the three by_source components sum exactly to
`prompt_tokens_total` in both arms.

**(b) The speedup is queue relief, not saved arithmetic. 39x amplification.**
Per turn the store avoids 1,399 tokens of prefill and saves 4.40 s of
`generate_sequences`. Those tokens are worth only **0.113 s** of raw compute.
(32B dense, 2 FLOP/param/token, tp=2 on H200 at ~40% MFU, so ~0.081 ms/token.)
The FLOP saving explains 2.6% of the gain.

The rest is contention. Prefill and decode share the GPUs, and
`num_requests_waiting` peaks at 299. At that load the engine is well past its
knee. A small cut in offered work buys a large cut in latency for *everyone*.

The cost model in (g) agrees independently: 3.07 ms of marginal system cost
against ~0.08 ms of hardware cost, the same ~38x.

**(c) HONEST NEGATIVE. The wall-clock numbers are makespan artifacts. Do not
quote them.** `timing_s/gen` equals the slowest single trajectory's
`generate_sequences + tool_calls` to within 0.1–3%, **in every step of both
arms**. It is a makespan over 1 trajectory out of 512.

Its 588 s mean gap breaks down as **+911 s** from that straggler's tool time and
**-325 s** from its generation time. The store's straggler actually generates
*longer*. Those net to 586 s against 588 s observed.

So 155% of the -37.7% is tool behaviour. `timing_s/step` and `perf/throughput`
have the same denominator and the same problem. Only (a) and
`generate_sequences/mean` survive it. That second one is a mean over 512
trajectories, where the straggler contributes ~1 s of 348 s.

**(d) HONEST NEGATIVE. The tool-time tail is not a treatment effect.** Mean tool
cost is identical: 1.97 s vs 2.01 s per call, with the store 4.5% *higher*. Only
the per-step maximum differs.

Normalise by `num_turns/max` across every arm-run on this harness and fifteen of
eighteen observations fall in 14–17.5 s/turn. The baseline sits in the middle of
them. The three outliers are the store arm's steps 2–4. That same arm's step 1
is 14.3, and its own smoke is 14.5 and 16.4, both on the norm.

`tool_calls/max` is 9–17x `tool_calls/mean`, so each step's value is one extreme
order statistic out of 512. Three low draws from a tail that heavy is
unremarkable.

Sandbox contention was tested directly and **refuted**. A 60 s sampler across
both arms shows indistinguishable fleet occupancy: mean 152.2 vs 154.4, p90 364
both, max 515 vs 539. There is no causal channel from a KV tier to how long a
shell command takes, and the data offers none.

**(e) HONEST NEGATIVE. The store did not reduce preemptions here.** 317 vs 325,
essentially identical. Earlier regimes showed the store halving preemptions.
This one does not.

The relief shows up as lower occupancy instead: `kv_cache_usage_perc` 0.751 to
0.659, peak running 112.8 to 93.3. Fewer evictions, no. Any claim that the tier
"prevents thrashing" is unsupported by this run.

**(f) The tiers compose rather than compete.** The store arm's *local* hit share
is the higher of the two, 18.5% against 14.3%, or +32.3% in absolute tokens.
That is despite it also pulling 43.8% externally.

The reason is simple. A pulled block is an ordinary cached block once it lands,
so it counts as a local hit on the next turn of the same trajectory. The tier is
strictly a fallback: gate 1 skips the store lookup entirely when the local cache
already covers the prompt.

**(g) Cost of an avoided prefill token, from recorded inputs only.** Divide the
recorded latency delta by the recorded token delta:
`122.1 s per trajectory / 39,824 avoided tokens = 3.07 ms`. See NOTE 3.

**(h) UNEXPLAINED. The store arm runs at ~2x the baseline's policy entropy.**
0.304–0.404 against 0.189–0.215, +86.6% on the mean. It is flat across all four
steps and well inside the < 1.0 validity gate. It is also present from step 1,
so it is a level offset, not a drift.

It appeared alongside a 43.8% external-transfer share. The leading hypothesis is
numerical. KV recomputed locally and KV pulled from the tier are probably not
bit-identical, because a prefix assembled from pulled blocks hits different
attention chunk boundaries. At a 43.8% share that may stop being negligible.
`actor/grad_norm` reaching 0.021 by step 4, against the baseline's 0.003, may be
the same effect.

The competing explanation is that the arms sampled divergent trajectories and
visited different states. `critic/score/mean` is nearly identical, 0.0288 vs
0.0303, which argues the outcomes are comparable. It does not settle it.

**Characterise this before using the tier for convergence work.** The cheap test
is a 1-step run comparing logprobs of pulled versus recomputed blocks on
identical prompts.

**(i) The standing cost is 1.2 TB of host memory.** `cpu_memory_used_gb` goes
from 164.9 to 1,221.9, which is the 8 x 128 GB Mooncake segments. On this node
that is affordable. On a smaller host it is the binding constraint. Either way
you pay it whether or not the tier is being hit.

**(j) Step 1 is not comparable to steps 2–4, symmetrically in both arms.**
`num_turns/mean` is 45.5 at step 1 and 23.0–24.9 afterwards, on identical tasks.
Steps 2–4 are stable to within 6%.

The cause is not determined. It is not the dataset, since prompt lengths are
bit-identical per step across arms. It is not the tier, since both arms show it.
Being symmetric, it changes no sign in the comparison. It does make cross-run
comparisons on turn-dependent quantities unsafe.

---

## NOTES — invited scrutiny

**NOTE 1. `timing_s/gen` and `timing_s/step` are single-trajectory makespans.**
Verified by identity, not assumed. Slowest `generate_sequences + tool_calls`
reproduces `timing_s/gen` to within 0.1–3% in all 8 arm-steps. Any rate divided
by these inherits a straggler. That is not a defect in the metric. It is what a
makespan is. It is a defect in using one as an A/B statistic.

**NOTE 2. `perf/throughput` is not a serving rate.** verl computes it as
`total_num_tokens / (timing_raw["step"] * n_gpus)`, in
`compute_throughout_metrics`, `verl/trainer/ppo/metric_utils.py`. The upstream
name has a typo.

The numerator is tokens *in the batch*, which is identical across arms by
construction at -0.03%. So the 2.27x cut in tokens actually **computed** is
invisible to it. The denominator is a step wall clock. Read it as "the step
finished sooner", never as "the engine served faster."

**NOTE 3. The 3.07 ms per-token figure is a capacity number, not a hardware
one.** `generate_sequences` is wall time and includes queueing. Avoided prefill
shortens the queue for everyone, so the figure carries the ~39x contention
amplification from (b) and rises with load. It also credits the whole latency
delta to avoided prefill, while the arms differ in scheduling too.

It is the right number for "what does a request cost in this system". It is the
wrong number for hardware sizing. Recompute it per regime. Do not carry it.

**NOTE 4. `hot_frac` is reported two ways.** Section 3.3 pools every snapshot
across an arm, giving 0.72 for the baseline. The results table means the
per-step values, giving 0.673. Both are correct. They answer slightly different
questions.

**NOTE 5. The store arm's generation tail is censored.** Its
`slowest/response_length` is pinned at 28,672 in all four steps, which is
exactly `max_response_length`. The baseline's is 5,299–16,478. So the store's
worst trajectory is one the config stopped, not one that was slow. A
longer-horizon run should raise the cap or report the clip rate.

---

## Appendix

### A1. Per-step detail

`recompute.md` and `store.md` carry all 22 verl step metrics per step plus the
engine-side decomposition per arm. `metrics.csv` carries all 93 metrics x 4
steps x 2 arms with per-arm means and deltas, verified bit-exact against the
source logs.

### A2. Serving split, per step (tokens)

| step | baseline `local_compute` | baseline `local_cache_hit` | store `local_compute` | store `local_cache_hit` | store `external` |
|---|---|---|---|---|---|
| 1 | 66,551,848 | 5,937,072 | 30,308,327 | 5,709,712 | 36,039,888 |
| 2 | 27,549,011 | 5,866,304 | 8,824,131 | 12,446,288 | 10,639,920 |
| 3 | 28,237,380 | 5,384,976 | 14,310,666 | 6,343,872 | 16,229,568 |
| 4 | 26,040,109 | 6,818,800 | 13,584,767 | 8,018,528 | 14,839,568 |

**These rows do not sum to the arm totals in §4, and should not.** §4 reports
the engines' final cumulative counters, which are the ground truth. The rows
above are *per-rollout-window deltas*. A window is the contiguous run of
snapshots with `num_requests_running` > 5. Tokens processed during ramp-up and
drain fall outside every window, so the decomposition is slightly lossy:

| | `local_compute` | `local_cache_hit` | `external_kv_transfer` |
|---|---|---|---|
| baseline, windows ÷ total | 99.85% | 96.63% | — |
| store, windows ÷ total | 99.99% | 98.92% | 99.99% |

Prefill concentrates inside the busy window, at 99.9%. Cache hits are
relatively more common during the quiet drain, which is why `local_cache_hit`
has the larger shortfall. Every `engine/*` row in `metrics.csv` is a window
delta on this basis. The §4 serving-split table is not.

### A3. Generation throughput, normalised

Decode tokens over `generate_sequences/mean x 512 trajectories` — independent of
how many tokens a step happened to present.

| step | baseline | store | ratio |
|---|---|---|---|
| 1 | 2.61 | 4.05 | 1.55x |
| 2 | 7.41 | 16.14 | 2.18x |
| 3 | 5.80 | 9.04 | 1.56x |
| 4 | 6.63 | 9.67 | 1.46x |
| mean | 5.61 | 9.72 | **1.73x** |

### A4. Mooncake operations, store arm (zero failed keys)

| operation | ops | keys | keys/op |
|---|---|---|---|
| `save_put` | 19,247 | 404,142 | 21.0 |
| `save_exists` | 21,710 | 9,451,476 | 435.4 |
| `load_get` | 4,182 | 2,547,176 | 609.1 |
| `lookup_exists` | 186,539 | 264,393,476 | 1,417.4 |

Operation *durations* were not scraped in this run, only counts — so no
per-pull latency is claimed anywhere in this report. That is the first
instrumentation gap to close.

### A5. Known instrumentation gaps

| gap | consequence |
|---|---|
| `simple_timer("tool_calls")` wraps `await sandbox_future` | a trajectory's first tool call is charged the whole sandbox boot (`wait_ready` 600 s default + `BASELINE_CMD` 120 s) plus executor queueing (512 trajectories, 8 workers x 32 threads = 256). `tool_calls` is not purely tool time |
| `SandboxClient.exec` swallows retry exceptions without logging | silent websocket retries are indistinguishable from none; 0 `SandboxError` is weak evidence |
| the agent loop never emits its `reason` | nothing records *why* a trajectory stopped — submitted, max-turns, token-budget, or error. This is what blocks explaining (j) |
| Mooncake op durations not scraped | no per-pull latency, so the cost of the store side of (g) is unquantified |
| no per-exec sandbox latency | the tail in (d) can be bounded statistically but not explained |
