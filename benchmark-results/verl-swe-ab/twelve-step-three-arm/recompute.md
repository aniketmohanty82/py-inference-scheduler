# local-only arm - per-step detail

Control arm of `README.md`: no kv_transfer_config, so every prompt token is
either served by the local prefix cache or recomputed. Over 12 steps it served
**35.7%** of prompt tokens from local cache and recomputed the rest.

Means and the three-arm comparison live in `README.md`; this file is the
per-step breakdown behind them.

## verl step metrics

| verl metric | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `timing_s/step` | 2,408 | 1,173 | 1,127 | 924 | 1,641 | 1,036 | 1,243 | 1,793 | 1,474 | 1,053 | 1,608 | 1,096 | 1,381 |
| `timing_s/gen` | 1,853 | 927 | 842 | 683 | 1,384 | 777 | 1,010 | 1,516 | 1,196 | 779 | 1,339 | 843 | 1,096 |
| `timing_s/update_actor` | 420 | 183 | 212 | 177 | 190 | 193 | 172 | 205 | 207 | 204 | 199 | 188 | 212 |
| `timing_s/agent_loop/generate_sequences/mean` | 808.5 | 113.5 | 149.9 | 93.2 | 113.0 | 100.2 | 75.9 | 132.3 | 128.5 | 127.2 | 104.0 | 85.4 | 169.3 |
| `timing_s/agent_loop/generate_sequences/max` | 1,751 | 888 | 740 | 450 | 791 | 685 | 476 | 777 | 1,052 | 636 | 684 | 810 | 812 |
| `timing_s/agent_loop/tool_calls/mean` | 71.8 | 41.4 | 39.3 | 41.9 | 44.5 | 42.8 | 46.1 | 46.4 | 40.5 | 46.7 | 40.9 | 49.0 | 45.9 |
| `timing_s/agent_loop/tool_calls/max` | 935 | 432 | 384 | 391 | 989 | 522 | 996 | 1,124 | 478 | 402 | 1,010 | 432 | 674 |
| `timing_s/agent_loop/slowest/generate_sequences` | 1,751 | 888 | 523 | 319 | 393 | 685 | 14 | 391 | 1,052 | 470 | 327 | 810 | 635 |
| `timing_s/agent_loop/slowest/tool_calls` | 23 | 38 | 318 | 364 | 989 | 90 | 996 | 1,124 | 143 | 307 | 1,010 | 32 | 453 |
| `num_turns/mean` | 45.64 | 23.19 | 23.38 | 23.02 | 22.69 | 21.56 | 21.30 | 22.97 | 21.96 | 23.36 | 22.21 | 22.79 | 24.51 |
| `response_length/mean` | 10,181 | 4,454 | 5,073 | 4,368 | 4,638 | 4,609 | 4,168 | 4,940 | 4,972 | 4,934 | 4,801 | 4,524 | 5,138 |
| `response_length/clip_ratio` | 0.0156 | 0.0059 | 0.0059 | 0.0000 | 0.0059 | 0.0117 | 0.0020 | 0.0039 | 0.0059 | 0.0000 | 0.0059 | 0.0078 | 0.0059 |
| `prompt_length/mean` | 517.5 | 518.8 | 521.1 | 515.2 | 520.1 | 516.2 | 515.7 | 520.6 | 514.2 | 522.1 | 523.3 | 513.0 | 518.2 |
| `perf/total_num_tokens` | 5,477,870 | 2,545,938 | 2,864,374 | 2,500,185 | 2,641,214 | 2,624,226 | 2,397,941 | 2,795,636 | 2,808,724 | 2,793,593 | 2,725,837 | 2,578,804 | 2,896,195 |
| `perf/throughput` | 284.3 | 271.2 | 317.8 | 338.3 | 201.2 | 316.5 | 241.1 | 194.9 | 238.2 | 331.5 | 211.9 | 294.0 | 270.1 |
| `actor/entropy` | 0.1880 | 0.2055 | 0.2136 | 0.2059 | 0.2037 | 0.2019 | 0.2106 | 0.1958 | 0.2026 | 0.2056 | 0.2032 | 0.2077 | 0.2037 |
| `actor/grad_norm` | 0.0023 | 0.0033 | 0.0031 | 0.0029 | 0.0008 | 0.0030 | 0.0022 | 0.0038 | 0.0025 | 0.0012 | 0.0018 | 0.0027 | 0.0025 |
| `critic/score/mean` | 0.0645 | 0.0156 | 0.0137 | 0.0215 | 0.0098 | 0.0273 | 0.0215 | 0.0234 | 0.0254 | 0.0156 | 0.0176 | 0.0273 | 0.0236 |
| `actor/perf/cpu_memory_used_gb` | 121.7 | 128.9 | 131.7 | 133.1 | 132.4 | 132.8 | 132.0 | 133.2 | 131.7 | 132.4 | 131.3 | 133.5 | 131.2 |

## Engine-side totals (full 12-step window)

| `vllm:prompt_tokens_by_source_total` | tokens | share |
|---|---|---|
| `local_compute` | 279,253,366 | 64.3% |
| `local_cache_hit` | 154,895,232 | 35.7% |
| `external_kv_transfer` | 0 | 0.0% |
| total | 434,148,598 | |

`vllm:num_preemptions_total` 408 over the run. `generation_tokens_total`
7,400,972. Peak `num_requests_running` 86, peak `num_requests_waiting` 264.

## Reading notes

- **Step 1 is the cold step**, in every arm. `num_turns/mean` is 45.6 against
  21-23 for steps 2-12, on the same dataset, and every timing is roughly
  double. All three arms show the same shape, so it does not change any sign in
  the comparison, but it is why 12-step means sit above the step 2-12 typical
  values.
- The `/metrics` sweep takes ~7.5 min per pass of all four engines, which is
  coarser than a rollout here (8-17 min), so the scrape does not resolve
  individual steps cleanly. Engine numbers are therefore reported as run
  totals, which are exact, rather than per step.
- `slowest/*` selects `argmax(generate_sequences + tool_calls + compute_score)`,
  a different trajectory each step. Use the `/max` rows for anything
  comparative.
