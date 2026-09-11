# recompute arm - per-step detail

Control arm of `README.md`: no kv_transfer_config, so every prompt token is
either served by the local prefix cache or recomputed. This is the arm the
sustained-pressure gate is about - it passed at **hot_frac 0.72** with
`local_cache_hit` never above 20.8% in any step.

Means and the store comparison live in `README.md`; this file is the per-step
breakdown behind them.

## verl step metrics

| verl metric | step 1 | step 2 | step 3 | step 4 | mean |
|---|---|---|---|---|---|
| `timing_s/step` | 2,641 | 1,931 | 1,321 | 1,725 | 1,904 |
| `timing_s/gen` | 2,072 | 1,657 | 1,054 | 1,450 | 1,558 |
| `timing_s/update_actor` | 427 | 203 | 199 | 205 | 259 |
| `timing_s/agent_loop/generate_sequences/mean` | 888.7 | 175.5 | 174.7 | 154.1 | 348.2 |
| `timing_s/agent_loop/generate_sequences/max` | 1,847 | 982 | 923 | 835 | 1,147 |
| `timing_s/agent_loop/tool_calls/mean` | 86.9 | 48.4 | 43.0 | 50.2 | 57.1 |
| `timing_s/agent_loop/tool_calls/max` | 886 | 1,055 | 1,009 | 992 | 986 |
| `timing_s/agent_loop/slowest/tool_calls` | 886 | 1,055 | 1,009 | 992 | 986 |
| `timing_s/agent_loop/slowest/generate_sequences` | 1,137 | 599 | 44 | 455 | 559 |
| `timing_s/agent_loop/slowest/response_length` | 16,478 | 15,784 | 5,299 | 7,935 | 11,374 |
| `num_turns/mean` | 45.48 | 23.45 | 23.85 | 23.02 | 28.95 |
| `num_turns/max` | 65 | 65 | 65 | 65 | 65 |
| `response_length/mean` | 10,499 | 4,848 | 4,783 | 4,877 | 6,252 |
| `response_length/clip_ratio` | 0.0098 | 0.0098 | 0.0078 | 0.0078 | 0.0088 |
| `prompt_length/mean` | 517.5 | 518.8 | 521.1 | 515.2 | 518.2 |
| `perf/total_num_tokens` | 5,640,222 | 2,747,659 | 2,715,920 | 2,760,947 | 3,466,187 |
| `perf/throughput` | 267.0 | 177.9 | 256.9 | 200.1 | 225.5 |
| `actor/entropy` | 0.1887 | 0.1723 | 0.2151 | 0.2059 | 0.1955 |
| `actor/ppo_kl` | +1.163e-05 | +2.522e-04 | +2.090e-04 | +1.329e-04 | +1.514e-04 |
| `actor/grad_norm` | 0.0028 | 0.0011 | 0.0019 | 0.0034 | 0.0023 |
| `critic/score/mean` | 0.0508 | 0.0117 | 0.0156 | 0.0371 | 0.0288 |
| `actor/perf/cpu_memory_used_gb` | 154.0 | 167.1 | 169.1 | 169.5 | 164.9 |

## Engine-side decomposition (per-minute `/metrics` scrape)

`busy min` counts snapshots with `num_requests_running` > 5. `hot%` is the
fraction of those above `kv_cache_usage_perc` 0.85 - the sustained-pressure
measure. Token columns are deltas across the rollout window.

| step | busy min | kv_avg | hot% | `waiting` peak | preempt | `local_compute` | `local_cache_hit` | `external_kv_transfer` | hit% | decode tok/traj-s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 29 | 0.827 | 73% | 299 | 162 | 66,551,848 | 5,937,072 | 0 | 8.2% | 2.61 |
| 2 | 14 | 0.725 | 67% | 95 | 63 | 27,549,011 | 5,866,304 | 0 | 17.6% | 7.41 |
| 3 | 14 | 0.678 | 60% | 95 | 48 | 28,237,380 | 5,384,976 | 0 | 16.0% | 5.80 |
| 4 | 12 | 0.774 | 69% | 94 | 44 | 26,040,109 | 6,818,800 | 0 | 20.8% | 6.63 |

## Reading notes

- **Step 1 is the cold step.** `num_turns/mean` is 45.5 against 23.0-23.9 for
  steps 2-4, on the same dataset. The store arm shows the same shape, so it is
  symmetric and does not change any sign in the comparison.
- Steps 2-4 are stable to within 6% on turns and token volume.
- `hit%` never exceeds 20.8%: under this regime the local prefix cache is
  destroyed faster than it rebuilds, which is the condition the pair was built
  to create.
- `slowest/*` selects `argmax(generate_sequences + tool_calls + compute_score)`
  (`agent_loop.py:1146`), a different trajectory each step. Use the `/max`
  rows for anything comparative.
