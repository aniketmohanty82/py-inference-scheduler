# our tier arm - per-step detail

`RLPullPolicyConnector`: the upstream connector plus one rule, decline any
tier fetch under 1,024 matched tokens and recompute locally. `save_decode_cache:
true`, same as native. Served **41.3%** of all prompt tokens from the external
tier.

This is the rerun. The first attempt crashed an engine at step 9 (orphaned
fetch record on a declined match, fixed with one line); its log is kept as
`store_attempt1_crashed_driver.log.gz` and used in no table. Means and the
three-arm comparison live in `README.md`; this file is the per-step breakdown
behind them.

## verl step metrics

| verl metric | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `timing_s/step` | 2,006 | 917 | 1,295 | 680 | 1,443 | 921 | 848 | 1,513 | 694 | 1,363 | 1,262 | 770 | 1,143 |
| `timing_s/gen` | 1,434 | 671 | 1,034 | 420 | 1,152 | 648 | 577 | 1,233 | 447 | 1,119 | 982 | 529 | 854 |
| `timing_s/update_actor` | 431 | 181 | 192 | 191 | 215 | 202 | 199 | 206 | 182 | 179 | 205 | 177 | 213 |
| `timing_s/agent_loop/generate_sequences/mean` | 278.3 | 28.7 | 39.8 | 29.1 | 51.9 | 39.7 | 35.0 | 37.0 | 30.0 | 29.4 | 49.3 | 27.5 | 56.3 |
| `timing_s/agent_loop/generate_sequences/max` | 848 | 502 | 220 | 251 | 305 | 512 | 543 | 495 | 158 | 380 | 582 | 492 | 441 |
| `timing_s/agent_loop/tool_calls/mean` | 89.1 | 57.4 | 49.4 | 45.8 | 42.2 | 45.6 | 46.0 | 47.5 | 44.2 | 51.7 | 40.9 | 48.8 | 50.7 |
| `timing_s/agent_loop/tool_calls/max` | 1,018 | 485 | 964 | 373 | 1,074 | 471 | 392 | 1,169 | 388 | 1,059 | 959 | 438 | 733 |
| `timing_s/agent_loop/slowest/generate_sequences` | 332 | 502 | 69 | 46 | 76 | 512 | 543 | 62 | 57 | 59 | 21 | 492 | 231 |
| `timing_s/agent_loop/slowest/tool_calls` | 1,018 | 167 | 964 | 373 | 1,074 | 134 | 33 | 1,169 | 388 | 1,059 | 959 | 36 | 615 |
| `num_turns/mean` | 44.68 | 22.31 | 23.16 | 22.34 | 23.46 | 23.11 | 21.80 | 21.66 | 22.06 | 22.24 | 22.57 | 22.39 | 24.32 |
| `response_length/mean` | 10,339 | 4,402 | 4,686 | 4,594 | 5,190 | 4,883 | 4,809 | 4,945 | 4,379 | 4,395 | 4,945 | 4,319 | 5,157 |
| `response_length/clip_ratio` | 0.0234 | 0.0059 | 0.0039 | 0.0039 | 0.0020 | 0.0137 | 0.0078 | 0.0059 | 0.0020 | 0.0039 | 0.0117 | 0.0039 | 0.0073 |
| `prompt_length/mean` | 517.5 | 518.8 | 521.1 | 515.2 | 520.1 | 516.2 | 515.7 | 520.6 | 514.2 | 522.1 | 523.3 | 513.0 | 518.2 |
| `perf/total_num_tokens` | 5,558,788 | 2,519,233 | 2,666,234 | 2,616,000 | 2,923,541 | 2,764,379 | 2,726,216 | 2,798,279 | 2,505,381 | 2,517,490 | 2,799,760 | 2,474,021 | 2,905,777 |
| `perf/throughput` | 346.4 | 343.3 | 257.4 | 480.9 | 253.2 | 375.3 | 401.7 | 231.2 | 451.1 | 230.9 | 277.4 | 401.6 | 337.5 |
| `actor/entropy` | 0.1850 | 0.1907 | 0.2243 | 0.2161 | 0.2038 | 0.1772 | 0.2068 | 0.2007 | 0.2259 | 0.2036 | 0.1709 | 0.2074 | 0.2010 |
| `actor/grad_norm` | 0.0025 | 0.0008 | 0.0020 | 0.0022 | 0.0016 | 0.0037 | 0.0032 | 0.0023 | 0.0026 | 0.0018 | 0.0021 | 0.0040 | 0.0024 |
| `critic/score/mean` | 0.0449 | 0.0137 | 0.0293 | 0.0156 | 0.0117 | 0.0215 | 0.0156 | 0.0156 | 0.0137 | 0.0156 | 0.0117 | 0.0234 | 0.0194 |
| `actor/perf/cpu_memory_used_gb` | 1,181.4 | 1,189.3 | 1,192.0 | 1,192.3 | 1,191.9 | 1,192.2 | 1,192.8 | 1,192.7 | 1,193.7 | 1,194.0 | 1,192.1 | 1,193.7 | 1,191.5 |

## Engine-side totals (full 12-step window)

| `vllm:prompt_tokens_by_source_total` | tokens | share |
|---|---|---|
| `local_compute` | 27,224,236 | 6.5% |
| `local_cache_hit` | 219,500,400 | 52.2% |
| `external_kv_transfer` | 173,636,848 | 41.3% |
| total | 420,361,484 | |

`vllm:num_preemptions_total` 1,756 over the run. `generation_tokens_total`
7,330,035. Peak `num_requests_running` 59, peak `num_requests_waiting` 123.
Zero connector errors and zero engine crashes across the 12 steps.

## Reading notes

- **Ran ~10 h after the other two arms**, on the same node and pods, against a
  fresh tier. Engine-side numbers are comparable; sandbox-side numbers are not
  quite: `tool_calls/mean` is 7.8% above native in 10 of 12 steps on the same
  21 sandbox nodes with no scheduling events, and that alone accounts for the
  arm's higher `timing_s/gen`. See the sandbox NOTE in `README.md`.
- `actor/entropy` is 0.17-0.23 across the 12 steps, matching both other arms.
  The 2x offset the 4-step pair's connector showed is absent.
- Preempted 18% less than native (1,756 vs 2,146). No timing benefit is
  visible from it.
- `slowest/*` selects `argmax(generate_sequences + tool_calls + compute_score)`,
  a different trajectory each step. Use the `/max` rows for anything
  comparative.
