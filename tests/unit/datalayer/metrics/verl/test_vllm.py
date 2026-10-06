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

from __future__ import annotations

from types import SimpleNamespace

from py_inference_scheduler.datalayer.metrics.verl.vllm import kv_capacity_tokens


def _server(num_gpu_blocks: int | None) -> SimpleNamespace:
    cache_config = SimpleNamespace(num_gpu_blocks=num_gpu_blocks, block_size=16)
    vllm_config = SimpleNamespace(cache_config=cache_config)
    return SimpleNamespace(engine=SimpleNamespace(vllm_config=vllm_config))


def test_capacity_is_gpu_blocks_times_block_size():
    assert kv_capacity_tokens(_server(4096)) == 65536


def test_capacity_is_zero_until_vllm_sizes_the_cache():
    assert kv_capacity_tokens(_server(None)) == 0
