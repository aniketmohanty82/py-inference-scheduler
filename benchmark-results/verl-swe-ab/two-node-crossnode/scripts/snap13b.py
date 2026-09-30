"""Per-minute engine scrape, run inside the ray-worker container (v2).

Same output format as snap13.py (SNAP <unix_ts> then vllm:* lines per
=== PORT ===), same file, so the drivers' tick and harvest readers are
unchanged.

Why v2: v1 probed EVERY listening port on the pod each minute. On a two-node
pod that is dozens of NCCL, Ray and Mooncake listeners; each ate the 4 s
timeout, a sweep took ~10 min, and the Mooncake handshake port logged a
malformed-JSON error per probe. Only 2 of 24 snapshots in the smoke carried
engine data. v2 resolves the ports owned by the vLLMHttpServer actors through
/proc/<pid>/fd socket inodes, so the sweep touches engines only, and gives a
busy API server 20 s to answer.
"""

import os
import time
import urllib.request

OUT = "/tmp/snap13.log"
HOST = os.environ.get("POD_IP") or os.popen("hostname -i").read().split()[0]
KEEP = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_by_source_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:mooncake_store_operation",
)
REDISCOVER_S = 300


def engine_ports() -> list[int]:
    """Listening TCP ports whose socket belongs to a vLLMHttpServer process."""
    inodes = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if b"vLLMHttpServer" not in open(f"/proc/{pid}/cmdline", "rb").read():
                continue
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    link = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if link.startswith("socket:["):
                    inodes.add(link[8:-1])
        except OSError:
            continue
    ports = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(path).read().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            f = ln.split()
            if len(f) > 9 and f[3] == "0A" and f[9] in inodes:  # LISTEN, ours
                ports.add(int(f[1].rsplit(":", 1)[1], 16))
    return sorted(p for p in ports if 1024 < p < 65535)


def scrape(port: int, timeout: float) -> str:
    with urllib.request.urlopen(f"http://{HOST}:{port}/metrics", timeout=timeout) as r:
        return r.read().decode()


known: set[int] = set()
discovered_at = 0.0
with open(OUT, "a", buffering=1) as out:
    while True:
        if time.time() - discovered_at > REDISCOVER_S:
            for port in engine_ports():
                if port in known:
                    continue
                try:
                    if "vllm:" in scrape(port, 4):
                        known.add(port)
                except Exception:  # noqa: BLE001 - not an engine port, or not up yet
                    continue
            discovered_at = time.time()
        out.write(f"SNAP {int(time.time())}\n")
        for port in sorted(known):
            try:
                text = scrape(port, 20)
            except Exception:  # noqa: BLE001 - engine busy or gone; next minute
                continue
            out.write(f"=== PORT {port} ===\n")
            for line in text.splitlines():
                if line.startswith(KEEP):
                    out.write(line + "\n")
        time.sleep(60)
