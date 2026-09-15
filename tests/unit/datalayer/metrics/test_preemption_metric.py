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

import re

from py_inference_scheduler.datalayer.metrics.prometheus import empty_vllm_stats, parse_vllm

# Shape vLLM actually emits: HELP/TYPE lines, model_name label, float counter.
_VLLM_TEXT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-7B-Instruct"} 37.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-7B-Instruct"} 12.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="Qwen/Qwen2.5-7B-Instruct"} 0.97
# HELP vllm:num_preemptions_total Cumulative number of preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="Qwen/Qwen2.5-7B-Instruct"} 143.0
"""


def test_parse_vllm_reads_preemption_counter():
    stats = parse_vllm(_VLLM_TEXT)
    assert stats["num_preempted"] == 143
    assert stats["num_running_reqs"] == 37
    assert stats["num_waiting_reqs"] == 12
    assert stats["kv"] == 0.97


def test_preemption_defaults_to_zero_when_absent():
    """An engine that never preempted omits the counter entirely."""
    without = "\n".join(
        line for line in _VLLM_TEXT.splitlines() if "num_preemptions" not in line
    )
    assert parse_vllm(without)["num_preempted"] == 0
    assert empty_vllm_stats()["num_preempted"] == 0


def test_verl_regex_matches_the_same_counter():
    """The verl path scrapes by regex, not the prometheus client; keep them agreed."""
    pattern = r"^(?:vllm:|vllm_)num_preemptions(?:_total)?(?:\{.*?\})?\s+([\d.]+)"
    found = re.findall(pattern, _VLLM_TEXT, re.MULTILINE)
    assert [int(float(v)) for v in found] == [143]
    # Older builds drop the _total suffix; the same pattern must still match.
    legacy = _VLLM_TEXT.replace("num_preemptions_total", "num_preemptions")
    assert [int(float(v)) for v in re.findall(pattern, legacy, re.MULTILINE)] == [143]
