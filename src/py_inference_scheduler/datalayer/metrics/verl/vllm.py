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
                    cache_config = re.search(r'^(?:vllm:|vllm_)cache_config_info\{([^}]*)\}', text, re.MULTILINE)  # noqa: E501
                    if cache_config:
                        stats["kv_cache_size"] = kv_capacity_tokens(cache_config.group(1))
                        stats["cache_labels"] = cache_config.group(1)[:600]  # VERIFY-ONLY
                    else:
                        stats["cache_labels"] = "MISSING"  # VERIFY-ONLY
                else:
                    stats["error"] = f"HTTP error {response.status}"
    except Exception as e:  # noqa: BLE001
        stats["error"] = f"Exception in get_routing_stats: {e}"

    return stats


def kv_capacity_tokens(cache_config_labels: str) -> int:
    """KV capacity in tokens from vLLM's cache_config_info labels; 0 when not reported."""
    labels = dict(re.findall(r'(\w+)="([^"]*)"', cache_config_labels))
    # Newer vLLM reports the size directly; older releases only give blocks and block size.
    if labels.get("kv_cache_size_tokens", "").isdigit():
        return int(labels["kv_cache_size_tokens"])
    blocks, block_size = labels.get("num_gpu_blocks", ""), labels.get("block_size", "")
    return int(blocks) * int(block_size) if blocks.isdigit() and block_size.isdigit() else 0
