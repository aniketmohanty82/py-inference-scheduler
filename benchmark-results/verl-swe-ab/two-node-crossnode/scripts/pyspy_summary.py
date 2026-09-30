"""Summarise py-spy collapsed-stack output by where an engine process spends its samples.

    python3 pyspy_summary.py <raw file> [<raw file> ...]

Each input is `py-spy record --format raw` output: one line per distinct stack,
frames separated by ';', trailing sample count. A sample is attributed to the
first matching bucket scanning the stack from the leaf inward, so connector
work called from vLLM's scheduler counts as connector, and idle waits are
recognised by their leaf frames.
"""

import re
import sys

BUCKETS = [
    ("idle: shm queue / sched_yield", r"sched_yield|shm_broadcast\.py|acquire_read|dequeue \("),
    ("idle: socket / zmq / queue wait", r"zmq/|selectors\.py|queue\.py:\d+ \(get|threading\.py:\d+ \(wait|poll \(|recv_multipart|connection\.py"),
    ("our connector (rl_pull_policy)", r"rl_pull_policy\.py"),
    ("mooncake connector: scheduler side", r"mooncake/store/scheduler\.py|mooncake/store/connector\.py"),
    ("mooncake connector: worker side", r"mooncake/store/worker\.py"),
    ("cpu offload connector", r"offloading_connector\.py|kv_offload/|offloading/"),
    ("kv connector glue (base / mixin / factory)", r"kv_connector_model_runner_mixin|kv_connector/(utils|v1/base)\.py"),
    ("vLLM scheduler + kv cache manager", r"v1/core/sched/|v1/core/kv_cache|v1/core/block_pool|v1/core/single_type"),
    ("vLLM engine core loop", r"v1/engine/core\.py|v1/executor/"),
    ("vLLM model runner / forward / sampler", r"v1/worker/|model_executor/|attention/|v1/sample/|compilation/"),
    ("torch / cuda", r"torch/|cuda"),
    ("nccl / distributed", r"distributed/(device_communicators|parallel_state|utils)"),
    ("our hook (verl_hook / scheduler core)", r"verl_hook\.py|py_inference_scheduler/core/|py_inference_scheduler/plugins/|shared_inflight\.py"),
    ("ray rpc / actor calls", r"ray/_private|ray/actor|ray/_raylet|core_worker"),
    ("asyncio", r"asyncio/"),
]

for path in sys.argv[1:]:
    total = 0
    counts = {name: 0 for name, _ in BUCKETS}
    counts["other"] = 0
    leaf_top: dict[str, int] = {}
    for ln in open(path, errors="replace"):
        ln = ln.rstrip("\n")
        m = re.match(r"^(.*) (\d+)$", ln)
        if not m:
            continue
        stack, n = m.group(1), int(m.group(2))
        total += n
        frames = stack.split(";")
        leaf = frames[-1] if frames else "?"
        leaf_top[leaf] = leaf_top.get(leaf, 0) + n
        hit = "other"
        for frame in reversed(frames):
            done = False
            for name, pat in BUCKETS:
                if re.search(pat, frame):
                    hit = name
                    done = True
                    break
            if done:
                break
        counts[hit] += n
    print(f"== {path}: {total} samples")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {100 * n / total:5.1f}%  {name}")
    print("  top leaf frames:")
    for leaf, n in sorted(leaf_top.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {100 * n / total:5.1f}%  {leaf[:110]}")
