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

import pytest

from integration.verl.overscheduling.engine_metrics import (
    KV_USAGE,
    RUNNING,
    EngineMetricsPoller,
    parse_metrics,
)

SCRAPE = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage.
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.5
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_preemptions_total{engine="0",model_name="m"} 2.0
vllm:num_preemptions_total{engine="1",model_name="m"} 1.0
vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 100.0
vllm:prompt_tokens_by_source_total{engine="0",source="external_kv_transfer"} 40.0
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 4.0
vllm:num_requests_waiting_by_reason{engine="0",reason="deferred"} 2.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0"} 9.0
vllm:num_preemptions_created{engine="0"} 1.7e9
process_cpu_seconds_total 12.0
"""


def test_parse_sums_labels_keeps_sources_apart_and_skips_buckets() -> None:
    values = parse_metrics(SCRAPE)
    assert values[KV_USAGE] == 0.5
    assert values["vllm:num_preemptions_total"] == 3.0
    assert values["vllm:prompt_tokens_by_source_total[local_compute]"] == 100.0
    assert values["vllm:prompt_tokens_by_source_total[external_kv_transfer]"] == 40.0
    assert values["vllm:num_requests_waiting_by_reason[capacity]"] == 4.0
    assert values["vllm:num_requests_waiting_by_reason[deferred]"] == 2.0
    assert not any("bucket" in key or "created" in key for key in values)
    assert "process_cpu_seconds_total" not in values


def test_window_time_averages_gauges_and_takes_counter_deltas_over_the_pad() -> None:
    poller = EngineMetricsPoller(["a"], interval_s=1.0)
    poller.samples = [
        (9.0, "a", {KV_USAGE: 0.0, RUNNING: 0.0, "vllm:num_preemptions_total": 5.0}),
        (10.0, "a", {KV_USAGE: 0.0, RUNNING: 2.0, "vllm:num_preemptions_total": 5.0}),
        (12.0, "a", {KV_USAGE: 1.0, RUNNING: 4.0, "vllm:num_preemptions_total": 6.0}),
        (14.0, "a", {KV_USAGE: 1.0, RUNNING: 4.0, "vllm:num_preemptions_total": 8.0}),
        (15.0, "a", {KV_USAGE: 0.0, RUNNING: 0.0, "vllm:num_preemptions_total": 9.0}),
    ]
    stats = poller.window(10.0, 14.0, pad=1.0)["a"]
    # Ramp 0 -> 1 over 2 s, then flat at 1 for 2 s: 3 KV-seconds over 4 s.
    assert stats["kv_mean"] == pytest.approx(0.75)
    assert stats["kv_p50"] == 1.0
    assert stats["kv_max"] == 1.0
    assert stats["running_mean"] == pytest.approx(3.5)
    assert stats["delta:vllm:num_preemptions_total"] == 4.0


def test_window_skips_a_sampler_with_fewer_than_two_points() -> None:
    poller = EngineMetricsPoller(["a"], interval_s=1.0)
    poller.samples = [(10.0, "a", {KV_USAGE: 0.4})]
    assert poller.window(9.0, 11.0, pad=0.0) == {}
