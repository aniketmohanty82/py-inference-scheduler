# store arm - per-step detail

Treatment arm of `README.md`: `DecodeKVSavingConnector` over mooncake, with
the per-step flush enabled (`RLS_FLUSH_STORE_ON_RESET`, default on). Served
**43.8%** of all prompt tokens from the external tier.

Means and the recompute comparison live in `README.md`; this file is the
per-step breakdown behind them.

## verl step metrics

| verl metric | step 1 | step 2 | step 3 | step 4 | mean |
|---|---|---|---|---|---|
| `timing_s/step` | 2,020 | 1,013 | 1,172 | 1,063 | 1,317 |
| `timing_s/gen` | 1,479 | 763 | 879 | 762 | 971 |
| `timing_s/update_actor` | 404 | 185 | 218 | 225 | 258 |
| `timing_s/agent_loop/generate_sequences/mean` | 577.7 | 77.7 | 130.8 | 118.4 | 226.2 |
| `timing_s/agent_loop/generate_sequences/max` | 1,388 | 730 | 804 | 613 | 884 |
| `timing_s/agent_loop/tool_calls/mean` | 89.5 | 59.7 | 44.7 | 44.9 | 59.7 |
| `timing_s/agent_loop/tool_calls/max` | 933 | 421 | 399 | 483 | 559 |
| `timing_s/agent_loop/slowest/tool_calls` | 44 | 32 | 74 | 148 | 74 |
| `timing_s/agent_loop/slowest/generate_sequences` | 1,388 | 730 | 804 | 613 | 884 |
| `timing_s/agent_loop/slowest/response_length` | 28,672 | 28,672 | 28,672 | 28,672 | 28,672 |
| `num_turns/mean` | 46.14 | 23.53 | 24.93 | 24.00 | 29.65 |
| `num_turns/max` | 65 | 65 | 65 | 65 | 65 |
| `response_length/mean` | 10,007 | 4,485 | 5,173 | 5,333 | 6,250 |
| `response_length/clip_ratio` | 0.0137 | 0.0098 | 0.0078 | 0.0059 | 0.0093 |
| `prompt_length/mean` | 517.5 | 518.8 | 521.1 | 515.2 | 518.2 |
| `perf/total_num_tokens` | 5,388,596 | 2,561,784 | 2,915,469 | 2,994,478 | 3,465,082 |
| `perf/throughput` | 333.5 | 316.1 | 311.1 | 352.1 | 328.2 |
| `actor/entropy` | 0.4043 | 0.3553 | 0.3043 | 0.3950 | 0.3647 |
| `actor/ppo_kl` | +6.368e-05 | +3.654e-05 | -3.240e-04 | +1.039e-04 | -2.996e-05 |
| `actor/grad_norm` | 0.0028 | 0.0023 | 0.0079 | 0.0206 | 0.0084 |
| `critic/score/mean` | 0.0469 | 0.0156 | 0.0195 | 0.0391 | 0.0303 |
| `actor/perf/cpu_memory_used_gb` | 1,213.3 | 1,222.4 | 1,225.6 | 1,226.4 | 1,221.9 |

## Engine-side decomposition (per-minute `/metrics` scrape)

Same columns as `recompute.md`, plus the external tier. `hit%` here is
`local_cache_hit` only - external transfers are counted separately.

| step | busy min | kv_avg | hot% | `waiting` peak | preempt | `local_compute` | `local_cache_hit` | `external_kv_transfer` | hit% | decode tok/traj-s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 0.817 | 76% | 303 | 170 | 30,308,327 | 5,709,712 | 36,039,888 | 7.9% | 4.05 |
| 2 | 9 | 0.564 | 20% | 50 | 49 | 8,824,131 | 12,446,288 | 10,639,920 | 39.0% | 16.14 |
| 3 | 10 | 0.670 | 55% | 108 | 58 | 14,310,666 | 6,343,872 | 16,229,568 | 17.2% | 9.04 |
| 4 | 11 | 0.584 | 33% | 97 | 48 | 13,584,767 | 8,018,528 | 14,839,568 | 22.0% | 9.67 |

## Flush events

The wipe is armed scheduler-side by `reset_cache()` and executed worker-side
when the generation reaches `bind_connector_metadata`, on the connector's
existing store client (building a second client in-engine clobbers segment
registration). Logged at every weight-update boundary:

| generation | keys removed |
|---|---|
| 1 | 0 (tier empty at first wake_up - expected) |
| 2 | 57,596 |
| 3 | 81,450 |
| 4 | 140,745 and 188,303 |

Generation 4 appears twice because the wipe is issued by **each engine's**
tp_rank 0, so `remove_all(force=True)` runs up to 4x per boundary. It is
global and idempotent, so this is wasteful rather than wrong.

`force=True` is required: mooncake grants leases on ACCESS, not on write, so a
wipe following recent reads removes nothing without it.

Zero `Client not available`, `Reconnect failed`, `RPC_FAIL` or
`EngineDeadError` across the arm; zero failed keys on every mooncake op.

## Mooncake operation totals (zero failed keys)

| operation | ops | keys | keys/op |
|---|---|---|---|
| `save_put` | 19,247 | 404,142 | 21.0 |
| `save_exists` | 21,710 | 9,451,476 | 435.4 |
| `load_get` | 4,182 | 2,547,176 | 609.1 |
| `lookup_exists` | 186,539 | 264,393,476 | 1,417.4 |

## Reading notes

- **`slowest/response_length` is pinned at 28,672 in every step** - exactly
  `data.max_response_length`. The gating trajectory is one the config stopped,
  not one that was slow, so this arm's generation tail is censored.
- `actor/entropy` is flat (0.304-0.404) where the un-flushed pair-v2 store arm
  climbed 0.198 -> 0.775. The drift is fixed; the ~2x level offset against
  recompute is a separate, unexplained effect - see README.
- This arm scores 0.56 on the sustained-pressure gate, below the 0.60 bar.
  That is the measurement, not a failure: it relieves its own pressure on
  identical work.
