#!/bin/bash
# Arm-independent engine metrics scraper.
#
# FLEET lines come from our hook, which the BASELINE deliberately does not
# install, so hook-side instrumentation can never measure the baseline. This
# runs in the worker pod and polls the vLLM engines directly, so both arms are
# measured identically by something neither arm contains.
#
# Port discovery is restricted to sockets OWNED BY vLLMHttpServer processes.
# NEVER sweep every listening port: the FSDP actor and vLLM TP workers hold
# NCCL bootstrap listeners on ephemeral ports, and an HTTP GET landing on one
# during a cross-node communicator handshake wedges the rendezvous silently
# (all ranks parked in epoll_wait, no watchdog, forever). Two consecutive
# two-node 32B runs died exactly this way while a blind sweep was running;
# single-node runs never noticed because they have no cross-node bootstrap.
# Engines bind the POD IP on ephemeral ports, not loopback, so probing
# 127.0.0.1 finds nothing.
PODIP=$(hostname -i | awk '{print $1}')
OUT=/tmp/engine_metrics.tsv
INTERVAL=${INTERVAL:-15}
# Self-detach: every output below is an explicit file append, and a launching
# `kubectl exec` otherwise blocks on the inherited stdout for the process's
# whole lifetime.
exec </dev/null >/dev/null 2>&1
[ -s "$OUT" ] || echo -e "ts\tport\tkv\trunning\twaiting\tpreempted\tpc_queries\tpc_hits\tgen_tokens\tprompt_tokens\tq_sum\tq_count\ttpot_sum\ttpot_count" > "$OUT"

# Candidate ports = LISTEN sockets whose owning process is a vLLMHttpServer
# actor. Each such process holds its Ray worker port plus the uvicorn port;
# is_vllm() then keeps only the one that serves /metrics.
listening() {
  python3 - <<'PY' 2>/dev/null
import psutil
ports = set()
for c in psutil.net_connections("tcp"):
    if c.status != "LISTEN" or not c.pid:
        continue
    try:
        cmd = " ".join(psutil.Process(c.pid).cmdline())
    except Exception:
        continue
    if "vLLMHttpServer" in cmd:
        ports.add(c.laddr.port)
print("\n".join(str(p) for p in sorted(ports)))
PY
}

is_vllm() {
  curl -s --max-time 1 "http://$PODIP:$1/metrics" 2>/dev/null | grep -q '^vllm:' && echo "$1"
}
export -f is_vllm
export PODIP

# Engines from earlier jobs stay alive and idle, so a cached port list keeps
# answering with a frozen counter while the live job's engines sit on new
# ports. Rediscover on a timer, not only when the cache stops responding.
PORTS=""
CYCLES=0
REDISCOVER_EVERY=${REDISCOVER_EVERY:-12}
while true; do
  if [ -z "$PORTS" ] || [ $((CYCLES % REDISCOVER_EVERY)) -eq 0 ]; then
    PORTS=$(listening | xargs -P 32 -I{} bash -c 'is_vllm {}' 2>/dev/null | sort -un | tr '\n' ' ')
    [ -n "$PORTS" ] && echo "discovered vllm ports: $PORTS" >> /tmp/scraper.out
  fi
  ts=$(date -u +%H:%M:%S)
  for port in $PORTS; do
    body=$(curl -s --max-time 2 "http://$PODIP:${port}/metrics" 2>/dev/null) || continue
    kv=$(printf '%s' "$body" | grep -oE '^vllm:kv_cache_usage_perc[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    run=$(printf '%s' "$body" | grep -oE '^vllm:num_requests_running[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    wt=$(printf '%s' "$body" | grep -oE '^vllm:num_requests_waiting[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    pre=$(printf '%s' "$body" | grep -oE '^vllm:num_preemptions[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    # Prefix-cache counters are pod-wide multiproc aggregates (like the
    # preemption counter): per-arm hit rate = delta(hits)/delta(queries)
    # within the arm's window, not a per-engine figure.
    pcq=$(printf '%s' "$body" | grep -oE '^vllm:(gpu_)?prefix_cache_queries[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    pch=$(printf '%s' "$body" | grep -oE '^vllm:(gpu_)?prefix_cache_hits[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    # PERFORMANCE at the engine, immune to sandbox tool time (which owns the
    # rollout wall and makes it useless as a rate denominator). Counters, so
    # per-arm figures are end-minus-start deltas:
    #   gen tokens / busy engine-seconds  = real decode throughput
    #   d(queue_time_sum) / d(queue_count) = mean engine queue wait, the
    #     quantity the gate should move by holding requests at the router
    #   d(tpot_sum) / d(tpot_count)        = per-output-token latency
    gtok=$(printf '%s' "$body" | grep -oE '^vllm:generation_tokens_total[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    ptok=$(printf '%s' "$body" | grep -oE '^vllm:prompt_tokens_total[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    qsum=$(printf '%s' "$body" | grep -oE '^vllm:request_queue_time_seconds_sum[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    qcnt=$(printf '%s' "$body" | grep -oE '^vllm:request_queue_time_seconds_count[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    tsum=$(printf '%s' "$body" | grep -oE '^vllm:time_per_output_token_seconds_sum[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    tcnt=$(printf '%s' "$body" | grep -oE '^vllm:time_per_output_token_seconds_count[^ ]* [0-9.e+-]+' | awk '{print $2}' | sort -g | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$ts" "$port" "${kv:-na}" "${run:-na}" "${wt:-na}" "${pre:-0}" "${pcq:-na}" "${pch:-na}" \
      "${gtok:-na}" "${ptok:-na}" "${qsum:-na}" "${qcnt:-na}" "${tsum:-na}" "${tcnt:-na}" >> "$OUT"
  done
  CYCLES=$((CYCLES + 1))
  sleep "$INTERVAL"
done
