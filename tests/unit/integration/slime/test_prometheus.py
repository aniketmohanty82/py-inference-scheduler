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

from py_inference_scheduler.datalayer.metrics.prometheus import parse_sglang

_SGLANG_METRICS = (
    "# HELP sglang:num_running_reqs running\n"
    "# TYPE sglang:num_running_reqs gauge\n"
    'sglang:num_running_reqs{model_name="qwen",engine="0"} 12.0\n'
    "# TYPE sglang:num_queue_reqs gauge\n"
    'sglang:num_queue_reqs{model_name="qwen",engine="0"} 3.0\n'
    "# TYPE sglang:token_usage gauge\n"
    'sglang:token_usage{model_name="qwen",engine="0"} 0.74\n'
    "# TYPE sglang:other_metric gauge\n"
    "sglang:other_metric 99.0\n"
)


def test_parse_sglang_extracts_labeled_gauges():
    stats = parse_sglang(_SGLANG_METRICS)
    assert stats["num_running_reqs"] == 12
    assert stats["num_waiting_reqs"] == 3
    assert stats["kv"] == 0.74
    assert stats["error"] is None
    # counts are ints, kv is a float
    assert isinstance(stats["num_running_reqs"], int)
    assert isinstance(stats["kv"], float)


def test_parse_sglang_missing_metrics_defaults_to_zero():
    stats = parse_sglang("# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs 5.0\n")
    assert stats["num_running_reqs"] == 5
    assert stats["num_waiting_reqs"] == 0
    assert stats["kv"] == 0.0


def test_parse_sglang_empty_payload():
    stats = parse_sglang("")
    assert stats == {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0, "error": None}


# Shape of a SGLang v0.5.20 scheduler scrape (observability/metrics_collector.py at
# the v0.5.20 tag): the full scheduler label set, the sibling utilisation gauges
# added next to token_usage, and a tokenizer histogram carrying is_streaming.
_SGLANG_V0_5_20_LABELS = (
    'model_name="Qwen/Qwen3-8B",engine_type="unified",tp_rank="0",pp_rank="0",'
    'moe_ep_rank="0",dp_rank="0"'
)
_SGLANG_V0_5_20_METRICS = (
    "# TYPE sglang:num_running_reqs gauge\n"
    f"sglang:num_running_reqs{{{_SGLANG_V0_5_20_LABELS}}} 12.0\n"
    "# TYPE sglang:num_queue_reqs gauge\n"
    f"sglang:num_queue_reqs{{{_SGLANG_V0_5_20_LABELS}}} 3.0\n"
    "# TYPE sglang:num_grammar_queue_reqs gauge\n"
    f"sglang:num_grammar_queue_reqs{{{_SGLANG_V0_5_20_LABELS}}} 9.0\n"
    "# TYPE sglang:token_usage gauge\n"
    f"sglang:token_usage{{{_SGLANG_V0_5_20_LABELS}}} 0.74\n"
    "# TYPE sglang:full_token_usage gauge\n"
    f"sglang:full_token_usage{{{_SGLANG_V0_5_20_LABELS}}} 0.91\n"
    "# TYPE sglang:swa_token_usage gauge\n"
    f"sglang:swa_token_usage{{{_SGLANG_V0_5_20_LABELS}}} 0.33\n"
    "# TYPE sglang:mamba_usage gauge\n"
    f"sglang:mamba_usage{{{_SGLANG_V0_5_20_LABELS}}} 0.0\n"
    "# TYPE sglang:kv_used_tokens gauge\n"
    f"sglang:kv_used_tokens{{{_SGLANG_V0_5_20_LABELS}}} 48000.0\n"
    "# TYPE sglang:cache_hit_rate gauge\n"
    f"sglang:cache_hit_rate{{{_SGLANG_V0_5_20_LABELS}}} 0.61\n"
    "# TYPE sglang:time_to_first_token_seconds histogram\n"
    'sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",'
    'engine_type="unified",is_streaming="false",le="0.1"} 5.0\n'
    'sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",'
    'engine_type="unified",is_streaming="false",le="+Inf"} 7.0\n'
    'sglang:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-8B",'
    'engine_type="unified",is_streaming="false"} 1.2\n'
    'sglang:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-8B",'
    'engine_type="unified",is_streaming="false"} 7.0\n'
)


def test_parse_sglang_v0_5_20_scrape_picks_exact_families():
    stats = parse_sglang(_SGLANG_V0_5_20_METRICS)
    assert stats["num_running_reqs"] == 12
    assert stats["num_waiting_reqs"] == 3
    # kv must come from token_usage, not the full_/swa_ siblings that share its prefix.
    assert stats["kv"] == 0.74
    assert stats["error"] is None


def test_parse_sglang_multiproc_takes_max_across_samples():
    # Prometheus multiprocess mode (e.g. TP>1) exposes one sample per PID; the
    # scheduler process reports the real value while others report 0.
    text = (
        "# TYPE sglang:num_running_reqs gauge\n"
        'sglang:num_running_reqs{pid="1"} 0.0\n'
        'sglang:num_running_reqs{pid="2"} 7.0\n'
        "# TYPE sglang:token_usage gauge\n"
        'sglang:token_usage{pid="1"} 0.2\n'
        'sglang:token_usage{pid="2"} 0.8\n'
    )
    stats = parse_sglang(text)
    assert stats["num_running_reqs"] == 7
    assert stats["kv"] == 0.8
