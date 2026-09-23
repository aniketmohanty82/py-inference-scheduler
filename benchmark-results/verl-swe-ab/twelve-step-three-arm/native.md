# vLLM native tier arm - per-step detail

The upstream `MooncakeStoreConnector` in vLLM 0.29.0, unmodified, with
`save_decode_cache: true`. Served **44.9%** of all prompt tokens from the
external tier. This is the arm our connector is measured against.

Means and the three-arm comparison live in `README.md`; this file is the
per-step breakdown behind them.

## verl step metrics

| verl metric | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `timing_s/step` | 1,681 | 783 | 848 | 772 | 1,479 | 832 | 1,370 | 872 | 887 | 1,294 | 738 | 770 | 1,027 |
| `timing_s/gen` | 1,123 | 531 | 583 | 523 | 1,214 | 570 | 1,136 | 614 | 623 | 1,007 | 499 | 504 | 744 |
| `timing_s/update_actor` | 421 | 185 | 195 | 183 | 195 | 193 | 172 | 189 | 194 | 211 | 175 | 196 | 209 |
| `timing_s/agent_loop/generate_sequences/mean` | 302.2 | 27.2 | 42.8 | 27.3 | 49.2 | 39.6 | 26.7 | 42.5 | 43.4 | 49.6 | 34.5 | 28.9 | 59.5 |
| `timing_s/agent_loop/generate_sequences/max` | 982 | 225 | 538 | 453 | 306 | 539 | 279 | 568 | 541 | 608 | 414 | 273 | 477 |
| `timing_s/agent_loop/tool_calls/mean` | 85.5 | 53.9 | 41.8 | 43.3 | 43.6 | 40.4 | 47.0 | 41.3 | 35.8 | 45.8 | 39.8 | 46.5 | 47.0 |
| `timing_s/agent_loop/tool_calls/max` | 458 | 486 | 451 | 491 | 987 | 418 | 1,102 | 388 | 380 | 951 | 440 | 454 | 584 |
| `timing_s/agent_loop/slowest/generate_sequences` | 982 | 43 | 538 | 453 | 236 | 539 | 31 | 568 | 541 | 54 | 408 | 49 | 370 |
| `timing_s/agent_loop/slowest/tool_calls` | 63 | 486 | 43 | 69 | 977 | 28 | 1,102 | 45 | 81 | 951 | 89 | 454 | 366 |
| `num_turns/mean` | 45.96 | 22.76 | 23.33 | 22.09 | 23.13 | 21.74 | 21.46 | 22.29 | 21.93 | 22.49 | 21.93 | 21.49 | 24.22 |
| `response_length/mean` | 10,172 | 4,514 | 4,744 | 4,451 | 4,756 | 4,654 | 4,135 | 4,578 | 4,646 | 5,104 | 4,260 | 4,709 | 5,060 |
| `response_length/clip_ratio` | 0.0117 | 0.0020 | 0.0039 | 0.0020 | 0.0039 | 0.0098 | 0.0059 | 0.0098 | 0.0039 | 0.0117 | 0.0039 | 0.0020 | 0.0059 |
| `prompt_length/mean` | 517.5 | 518.8 | 521.1 | 515.2 | 520.1 | 516.2 | 515.7 | 520.6 | 514.2 | 522.1 | 523.3 | 513.0 | 518.2 |
| `perf/total_num_tokens` | 5,473,047 | 2,576,947 | 2,695,978 | 2,542,788 | 2,701,464 | 2,647,294 | 2,380,993 | 2,610,424 | 2,642,101 | 2,880,335 | 2,449,157 | 2,673,558 | 2,856,174 |
| `perf/throughput` | 406.9 | 411.3 | 397.6 | 411.7 | 228.3 | 397.7 | 217.3 | 374.4 | 372.4 | 278.3 | 415.1 | 433.9 | 362.1 |
| `actor/entropy` | 0.1910 | 0.2057 | 0.1949 | 0.2044 | 0.1989 | 0.1860 | 0.2128 | 0.1825 | 0.2080 | 0.1670 | 0.2003 | 0.2148 | 0.1972 |
| `actor/grad_norm` | 0.0028 | 0.0018 | 0.0022 | 0.0019 | 0.0031 | 0.0021 | 0.0049 | 0.0010 | 0.0023 | 0.0020 | 0.0028 | 0.0015 | 0.0024 |
| `critic/score/mean` | 0.0449 | 0.0117 | 0.0273 | 0.0195 | 0.0215 | 0.0195 | 0.0371 | 0.0137 | 0.0215 | 0.0176 | 0.0137 | 0.0156 | 0.0220 |
| `actor/perf/cpu_memory_used_gb` | 1,181.7 | 1,188.3 | 1,191.5 | 1,192.8 | 1,191.8 | 1,192.1 | 1,191.6 | 1,191.8 | 1,192.6 | 1,193.0 | 1,193.1 | 1,193.7 | 1,191.2 |

## Engine-side totals (full 12-step window)

| `vllm:prompt_tokens_by_source_total` | tokens | share |
|---|---|---|
| `local_compute` | 24,247,030 | 5.9% |
| `local_cache_hit` | 204,050,272 | 49.3% |
| `external_kv_transfer` | 185,883,248 | 44.9% |
| total | 414,180,550 | |

`vllm:num_preemptions_total` 2,146 over the run. `generation_tokens_total`
7,319,259. Peak `num_requests_running` 55, peak `num_requests_waiting` 150.

## Weight-update wipe

Verified at the Mooncake master rather than in the driver log (verl runs the
engines at WARN, so the connector's own success line never appears). At the
first step boundary the master's key count fell from 495,458 to 8,972 within
one 30 s sample, with a nonzero `DelAll` rate in the same sample, then rebuilt
from the next step's writes. Zero `Client not available`, `Reconnect failed`,
`RPC_FAIL` or `EngineDeadError` across the arm.

## Reading notes

- `actor/entropy` is flat at 0.17-0.21 across all 12 steps, matching
  local-only. There is no staleness drift, which is what the wipe is for.
- Ran back to back with local-only on the same pods. See `README.md` for why
  the store arm's sandbox-side numbers are not directly comparable to this
  one.
- `slowest/*` selects `argmax(generate_sequences + tool_calls + compute_score)`,
  a different trajectory each step. Use the `/max` rows for anything
  comparative.
