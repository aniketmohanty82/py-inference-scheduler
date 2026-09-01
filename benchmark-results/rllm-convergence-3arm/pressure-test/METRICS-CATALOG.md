# Metrics catalog: every measurable signal in this benchmark stack

Per source: what exists, what we record today, and known gaps.
(Terminology: METHODOLOGY.md sec 3.0.)

## 1. vLLM engine `/metrics` (Prometheus, per replica)

| family | signals | recorded today? |
|---|---|---|
| token counters | prompt_tokens_total (PROCESSED), generation_tokens_total, prompt_tokens_cached, **prompt_tokens_by_source{local_compute, local_cache_hit, external_kv_transfer}** (true-compute split) | YES (sidecar) |
| request histograms | request_prompt_tokens, request_generation_tokens, request_max_num_generation_tokens (sum/count/buckets) | YES |
| cache | prefix_cache_queries/hits, external_prefix_cache_queries/hits | YES |
| pressure | num_preemptions_total, kv_cache_usage_perc | YES |
| **latency histograms** | **time_to_first_token_seconds, time_per_output_token_seconds, e2e_request_latency_seconds, request_queue_time_seconds, request_inference_time_seconds, request_prefill_time_seconds, request_decode_time_seconds** | **GAP - sidecar filter drops them.** These decompose the 16-49s per-turn fixed cost into queue vs prefill vs decode DIRECTLY, engine-side. Now captured by the unfiltered external poller (cell 1) and the widened sidecar (cells 2-4 on). |
| scheduler state | num_requests_running, num_requests_waiting (gauges) | GAP in sidecar (gateway routing_stats logs them incidentally); in full dumps now |
| config/info | cache_config_info (num_gpu_blocks -> exact KV pool size) | GAP - would replace our derived pool-size arithmetic |

## 2. Gateway sqlite trace DB (per turn)

latency_ms, timestamp, prompt/completion token counts, full token IDs,
logprobs, finish_reason, weight_version, raw request/response (incl.
max_tokens on the wire), session id. Recorded: YES (harvested pre-teardown).
Everything per-trajectory (spans, tails, turn counts) derives from here.

## 3. rllm driver stream

Task progress bar (rollout wall time, consumed/filtered), per-trajectory
"Rollout completed ... in Ns" (duration + reward), sandbox retry attempts
("Attempt k/3 failed" + reason), full config dump, engine startup/error
lines (traceback-patched), routing decision logs (RLS_DECISION_LOG).
Recorded: YES (streamed from launch).

## 4. Mooncake master :9003

segment_allocated_bytes per segment (locality evidence), batch RPC
counters (exist/put/get + failures), eviction counters (sweeps, keys,
bytes). Recorded: YES (timestamped snapshots). GAP: no per-operation
latency from the master side.

## 5. Store connector worker ops

The transfer threads emit per-op records (save_exists/save_put/get with
duration, bytes, status) via a record_operation callback -> engine
Prometheus under kv-transfer names. GAP: never captured (filtered out);
in full dumps now - gives store-op latency distributions (the load-path
cost, currently only inferable).

## 6. Kubernetes / infra

Pod phases + events (sandbox failures - pod_watcher), node conditions,
kubelet journals (pull durations, per-pod scheduling evidence), GPU
telemetry via nvidia-smi exec sampling (GAP - no continuous GPU util/mem
series; would separate compute-bound vs idle phases directly).

## 7. Not measurable without new instrumentation

Per-request RDMA wire bytes (DRANET exposes no IB port counters in-pod),
scheduler-side per-decision latency, sandbox tool-execution timing
(approximated as inter-turn gaps from the trace DB).

## Standing gaps worth closing next

1. vLLM latency histograms (queue/prefill/decode/TTFT) - CLOSED as of
   cell 1 (external poller) / cells 2-4 (widened sidecar).
2. cache_config_info for exact KV pool sizes - comes free with (1).
3. nvidia-smi utilization sampling loop per run.
4. Store-op latency histograms - comes free with (1) if the connector
   registers them; verify in the first full dump.
