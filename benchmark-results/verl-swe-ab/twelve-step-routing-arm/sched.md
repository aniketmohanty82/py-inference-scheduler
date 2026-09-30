# our-router arm - per-step detail

Routing arm of `README.md`: no kv_transfer_config, routing by
`PyInferenceAgentLoopManager` under `configs/swe-backpressure.yaml`. Over 12
steps it served **30.0%** of prompt tokens from local cache and recomputed the
rest; 11.7% of turns were moved off the engine holding their KV by the
saturation filter.

Means and the comparison against verl's balancer live in `README.md`; this
file is the per-step breakdown behind them. The verl arm's per-step file is
`../twelve-step-three-arm/recompute.md`.

## verl step metrics

| verl metric | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `timing_s/step` | 2,787 | 1,458 | 1,316 | 933 | 1,304 | 1,009 | 1,183 | 1,582 | 1,068 | 1,020 | 1,188 | 1,014 | 1,322 |
| `timing_s/gen` | 2,248 | 1,109 | 1,026 | 672 | 991 | 747 | 919 | 1,306 | 810 | 751 | 915 | 740 | 1,020 |
| `timing_s/update_actor` | 408 | 261 | 215 | 193 | 233 | 194 | 195 | 205 | 190 | 200 | 203 | 204 | 225 |
| `timing_s/agent_loop/generate_sequences/mean` | 578.6 | 211.8 | 183.9 | 112.7 | 168.3 | 109.6 | 123.8 | 121.2 | 115.2 | 135.7 | 128.4 | 124.3 | 176.1 |
| `timing_s/agent_loop/generate_sequences/max` | 1,370 | 952 | 993 | 560 | 923 | 516 | 879 | 514 | 785 | 597 | 866 | 658 | 801 |
| `timing_s/agent_loop/tool_calls/mean` | 118.3 | 50.2 | 45.5 | 43.9 | 44.0 | 52.8 | 43.3 | 47.7 | 44.7 | 46.1 | 41.7 | 45.9 | 52.0 |
| `timing_s/agent_loop/tool_calls/max` | 1,380 | 481 | 431 | 372 | 368 | 456 | 410 | 1,045 | 420 | 408 | 402 | 456 | 552 |
| `timing_s/agent_loop/slowest/generate_sequences` | 723 | 952 | 993 | 423 | 882 | 289 | 879 | 258 | 785 | 342 | 782 | 283 | 633 |
| `timing_s/agent_loop/slowest/tool_calls` | 1,380 | 78 | 32 | 248 | 108 | 456 | 35 | 1,045 | 24 | 408 | 130 | 456 | 367 |
| `num_turns/mean` | 42.92 | 29.00 | 24.70 | 22.84 | 24.57 | 23.49 | 23.20 | 23.61 | 23.20 | 23.35 | 22.51 | 23.29 | 25.56 |
| `response_length/mean` | 9,771 | 6,416 | 5,220 | 4,670 | 5,639 | 4,712 | 4,752 | 4,993 | 4,646 | 4,821 | 4,930 | 4,919 | 5,457 |
| `response_length/clip_ratio` | 0.0137 | 0.0098 | 0.0117 | 0.0039 | 0.0117 | 0.0020 | 0.0059 | 0.0059 | 0.0020 | 0.0000 | 0.0078 | 0.0039 | 0.0065 |
| `prompt_length/mean` | 517.5 | 518.8 | 521.1 | 515.2 | 520.1 | 516.2 | 515.7 | 520.6 | 514.2 | 522.1 | 523.3 | 513.0 | 518.2 |
| `perf/total_num_tokens` | 5,267,898 | 3,550,506 | 2,939,194 | 2,654,669 | 3,153,561 | 2,677,051 | 2,697,169 | 2,822,792 | 2,641,760 | 2,735,635 | 2,792,252 | 2,781,413 | 3,059,492 |
| `perf/throughput` | 236.3 | 304.4 | 279.2 | 355.8 | 302.3 | 331.6 | 285.0 | 223.1 | 309.1 | 335.2 | 293.9 | 342.8 | 299.9 |
| `actor/entropy` | 0.1932 | 0.2026 | 0.1979 | 0.2145 | 0.1917 | 0.2099 | 0.1822 | 0.2181 | 0.1923 | 0.2094 | 0.1888 | 0.2100 | 0.2009 |
| `actor/grad_norm` | 0.0029 | 0.0051 | 0.0016 | 0.0042 | 0.0009 | 0.0017 | 0.0021 | 0.0019 | 0.0024 | 0.0029 | 0.0006 | 0.0028 | 0.0024 |
| `critic/score/mean` | 0.0430 | 0.0137 | 0.0176 | 0.0293 | 0.0156 | 0.0156 | 0.0273 | 0.0215 | 0.0195 | 0.0137 | 0.0059 | 0.0176 | 0.0200 |
| `actor/perf/cpu_memory_used_gb` | 124.50 | 131.48 | 134.86 | 135.91 | 135.42 | 135.99 | 134.76 | 135.47 | 134.43 | 135.35 | 134.56 | 135.49 | 134.02 |

## Engine-side totals (full 12-step window)

| `vllm:prompt_tokens_by_source_total` | tokens | share |
|---|---|---|
| `local_compute` | 319,626,641 | 70.0% |
| `local_cache_hit` | 137,034,048 | 30.0% |
| `external_kv_transfer` | 0 | 0.0% |
| total | 456,660,689 | |

`vllm:num_preemptions_total` 431 over the run. `generation_tokens_total`
7,701,754. Peak `num_requests_running` 38, peak `num_requests_waiting` 66 in
the scrape (per-engine, sequential); the simultaneous FLEET lines saw up to
65 running and 81 waiting on one engine.

## Routing totals (from the driver log's AFFINITY and FLEET lines)

| | |
|---|---|
| decisions | 77,712 (25.2% / 25.3% / 24.6% / 24.8% per engine) |
| kept on the holder | 63,205 |
| moved by the saturation filter | 8,355 |
| first turns | 6,144 |
| fallbacks to verl's balancer | 0 |
| idle-while-queued busy snapshots | 3 of 3,677 |

## Reading notes

- **Step 1 is the cold step**, as in every arm: `num_turns/mean` 42.9 against
  22-26 for steps 2-12, and every timing roughly double. Its wall-setter spent
  1,380 s in shell commands on a 6,149-token trajectory.
- **Wall-setters by cause**: steps 2, 3, 5, 7, 9 and 11 were set by a
  max-length trajectory (28,672 tokens, sampling-bound); steps 1, 6, 8, 10 and
  12 by a short trajectory stuck in tool calls (248 to 1,380 s); step 4 by a
  7,319-token trajectory with 423 s of sampling and 248 s of tools.
- The `/metrics` sweep takes ~7.5 min per pass of all four engines, so engine
  numbers are reported as run totals. Per-engine balance comes from the FLEET
  lines instead, which read all four engines together every 15 s per worker.
- `slowest/*` selects `argmax(generate_sequences + tool_calls + compute_score)`,
  a different trajectory each step. Use the `/max` rows for anything
  comparative.
