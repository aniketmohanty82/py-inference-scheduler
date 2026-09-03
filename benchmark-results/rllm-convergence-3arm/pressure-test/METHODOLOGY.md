# Methodology: 128-task capped pair (store vs recompute)

The only benchmark in this folder. Every earlier run was invalidated by a
later regime change (slim-env sandboxes, the 1800s harness guillotine
censoring trajectories, uncapped turns) and has been deleted per the
valid-runs-only standard in ../RUN-STANDARDS.md.

## 1. Terminology

| term | meaning |
|---|---|
| step | one training step: sampling + training |
| rollout | the WHOLE sampling phase of a step (never one trajectory) |
| trajectory | one task attempt: turns + tool calls |
| turn | one LLM call + its tool execution |
| batch / generations | unique tasks / repeats per task for GRPO |

## 2. What differs between the arms

Exactly one line, verified by mechanical diff of both manifests: the store
arm passes `kv_transfer_config` (DecodeKVSavingConnector, save_decode_kv,
kv_role=kv_both); the recompute arm passes nothing. Same image **digest**,
same env, same task set, same bounds, same node, fresh mooncake master
between arms.

## 3. Frozen configuration

| knob | value |
|---|---|
| model / training | Qwen3-32B, LoRA r32 (`lora_rank`, merge=true), FSDP2 |
| topology | separated: 2 GPUs rollout (TP=2), 4 GPUs trainer, 1 node |
| sandboxes | per-task R2E images from the in-region AR mirror, 128-wide |
| turn cap | `RLLM_MSWEA_STEP_LIMIT=25` (binds) |
| agent timeout | `RLLM_HARNESS_RUN_TIMEOUT_S=10800` (must NEVER bind) |
| output length | `max_tokens=null` - vLLM fills the remaining 32k window |
| KV pool | `gpu_memory_utilization=0.30` (small pool = eviction pressure) |
| store tier | 2 x 512 GiB host-DRAM segments = 1 TiB |
| trainer memory | `max_split_size_mb:512`, entropy checkpointing, ulysses SP=4, `ppo_max_token_len_per_gpu=16384` |

## 4. Validity gates (a run counts only if all pass)

1. Zero trajectories terminated by the clock - every one ends by finishing
   or by hitting 25 turns. Otherwise faster serving leaks into
   "more turns before the deadline" instead of shorter trajectories, and
   the comparison is censored.
2. Zero filtered groups (a filtered group silently shrinks the batch and
   breaks arm symmetry).
3. Turn cap observed to bind (max turns == 25).
4. Store arm: 1 TiB mounted before sampling; counters drain at the end.
5. Both arms complete 128/128 trajectories.

## 5. Provenance: what is recorded vs derived

Nothing here is estimated. Every number is either read directly from a log
line/counter (RECORDED) or is arithmetic over recorded values (DERIVED,
with the formula given).

| quantity | provenance | source |
|---|---|---|
| trajectory wall, setup, agentflow, llm time, turn count, evaluator time | **RECORDED** - one `Rollout completed` line per trajectory | driver log |
| solved count / reward | **RECORDED** - `Rewards: [mini-swe-agent: N]` | driver log |
| rollout (sampling) wall time | **RECORDED** - tqdm elapsed at the 64-group threshold | driver log |
| groups consumed / filtered | **RECORDED** - rllm buffer counters | driver log |
| sandbox retries, enrich mismatches, store save/load errors | **RECORDED** - error lines | driver log |
| prompt tokens by source (local_compute / cache_hit / external) | **RECORDED** - vLLM counters, labels sum to the total exactly | metrics scrape |
| store ops, bytes, evictions, segment capacity | **RECORDED** - mooncake master :9003 | master scrape |
| verl step metrics (timing_s/*, loss, grad_norm, offpolicy/*, perf/*) | **RECORDED** - end-of-step table | driver log, only if the step completes |
| per-request queue/prefill/decode/TTFT/e2e | **RECORDED** - vLLM histograms | full `/metrics` poll |
| **llm seconds per turn** | **DERIVED** = llm / turns, per trajectory | from recorded |
| **tool seconds per turn** | **DERIVED** = (agentflow - llm) / turns | from recorded |
| **compute reduction factor** | **DERIVED** = recompute local_compute / store local_compute | from recorded |
| **projected 25-turn duration** | **DERIVED** = (agentflow/turns) x 25; linear, ignores context growth - used only for timeout headroom, never reported as a result | from recorded |

## 6. Known measurement gaps

- Episode files are empty despite `log_episodes=true`, so group **filter
  reasons** and per-task attribution are unavailable; filtering can only be
  counted, not explained.
- Rollout lines carry a per-run group uuid, not the task id, so trajectories
  cannot be matched across runs. Aggregate distributions only - which is
  sufficient because both arms are bounded identically.
