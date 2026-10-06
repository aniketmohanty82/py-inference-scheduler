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

from py_inference_scheduler.datalayer.metrics.verl.vllm import kv_capacity_tokens


def test_reads_the_reported_kv_size():
    labels = 'block_size="16",kv_cache_size_tokens="157088",num_gpu_blocks="9818"'
    assert kv_capacity_tokens(labels) == 157088


def test_falls_back_to_blocks_times_block_size():
    assert kv_capacity_tokens('block_size="16",num_gpu_blocks="9818"') == 157088


def test_capacity_is_zero_until_vllm_sizes_the_cache():
    assert kv_capacity_tokens('block_size="16",num_gpu_blocks="None"') == 0
