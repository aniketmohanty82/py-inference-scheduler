# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


async def get_vllm_routing_stats(server) -> dict:
    """Fetches routing stats from vLLM server via its Prometheus metrics endpoint."""
    stats: dict[str, Any] = {
        "num_waiting_reqs": 0,
        "num_running_reqs": 0,
        "kv": 0.0,
        "kv_cache_size": 0,
        "error": None,
    }
    try:
        stats["kv_cache_size"] = kv_capacity_tokens(server)
        host, port = server.get_server_address()
        url = f"http://{host}:{port}/metrics"

        async with aiohttp.ClientSession() as session:  # noqa: SIM117
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:  # noqa: PLR2004
                    text = await response.text()

                    waiting = re.search(r'^(?:vllm:|vllm_)num_requests_waiting(?:\{.*?\})?\s+([\d.]+)', text, re.MULTILINE)  # noqa: E501
                    running = re.search(r'^(?:vllm:|vllm_)num_requests_running(?:\{.*?\})?\s+([\d.]+)', text, re.MULTILINE)  # noqa: E501
                    kv_matches = re.findall(r'^(?:vllm:|vllm_)kv_cache_usage_perc(?:\{.*?\})?\s+([\d.]+)', text, re.MULTILINE)  # noqa: E501

                    if waiting:
                        stats["num_waiting_reqs"] = int(float(waiting.group(1)))
                    if running:
                        stats["num_running_reqs"] = int(float(running.group(1)))
                    if kv_matches:
                        stats["kv"] = max(float(m) for m in kv_matches)
                    # VERIFY-ONLY: engine preemptions for the GPU runs.
                    preempted = re.search(r'^(?:vllm:|vllm_)num_preemptions_total(?:\{.*?\})?\s+([\d.e+]+)', text, re.MULTILINE)  # noqa: E501
                    if preempted:
                        stats["preempt"] = int(float(preempted.group(1)))
                    for key, name in (("pc_queries", "prefix_cache_queries"), ("pc_hits", "prefix_cache_hits")):  # VERIFY-ONLY  # noqa: E501
                        found = re.findall(rf'^vllm:{name}(?:_total)?(?:\{{.*?\}})?\s+([0-9.e+-]+)', text, re.MULTILINE)  # noqa: E501
                        stats[key] = int(sum(float(v) for v in found))
                else:
                    stats["error"] = f"HTTP error {response.status}"
    except Exception as e:  # noqa: BLE001
        stats["error"] = f"Exception in get_routing_stats: {e}"

    return stats


def kv_capacity_tokens(server) -> int:
    """KV capacity in tokens of the server's vLLM engine; 0 until the engine has sized its cache."""
    cache_config = server.engine.vllm_config.cache_config
    return int((cache_config.num_gpu_blocks or 0) * cache_config.block_size)
