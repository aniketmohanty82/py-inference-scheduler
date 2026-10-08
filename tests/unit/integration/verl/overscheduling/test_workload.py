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

from integration.verl.overscheduling.workload import Workload


def test_same_seed_gives_identical_rows_so_arms_run_the_same_work() -> None:
    assert Workload(4, 50, 3, 64, 16, seed=7).rows() == Workload(4, 50, 3, 64, 16, seed=7).rows()
    other = Workload(4, 50, 3, 64, 16, seed=8).rows()
    assert other[0]["raw_prompt"] != Workload(4, 50, 3, 64, 16, seed=7).rows()[0]["raw_prompt"]


def test_rows_carry_unique_prompts_and_the_fixed_schedule() -> None:
    rows = Workload(3, 20, 4, 128, 32).rows()
    leads = [row["raw_prompt"][0]["content"].split(":")[0] for row in rows]
    assert leads == ["Request 0", "Request 1", "Request 2"]
    for i, row in enumerate(rows):
        assert row["index"] == i
        assert row["agent_name"] == "multiturn_load"
        assert row["extra_info"] == {
            "output_tokens": [128] * 4,
            "reply_tokens": [32] * 3,
            "tool_latency_s": [0.0] * 3,
        }
        assert len(row["raw_prompt"][0]["content"].split()) == 20 + 2


def test_tool_latency_is_seeded_and_leaves_prompts_unchanged() -> None:
    instant = Workload(4, 50, 3, 64, 16, seed=7).rows()
    timed = Workload(4, 50, 3, 64, 16, seed=7, tool_latency_s=3.0).rows()
    assert [r["raw_prompt"] for r in timed] == [r["raw_prompt"] for r in instant]
    assert timed == Workload(4, 50, 3, 64, 16, seed=7, tool_latency_s=3.0).rows()
    latencies = [r["extra_info"]["tool_latency_s"] for r in timed]
    assert all(len(row) == 2 and all(s > 0 for s in row) for row in latencies)
    other = Workload(4, 50, 3, 64, 16, seed=8, tool_latency_s=3.0).rows()
    assert other[0]["extra_info"]["tool_latency_s"] != latencies[0]


def test_tool_latency_averages_its_mean() -> None:
    rows = Workload(2000, 1, 6, 1, 1, seed=3, tool_latency_s=3.0).rows()
    draws = [s for row in rows for s in row["extra_info"]["tool_latency_s"]]
    assert len(draws) == 2000 * 5
    assert abs(sum(draws) / len(draws) - 3.0) < 0.05


def test_negative_tool_latency_is_refused() -> None:
    with pytest.raises(ValueError, match="tool_latency_s"):
        Workload(1, 1, 2, 1, 1, tool_latency_s=-1.0)


def test_max_context_adds_template_slack_to_the_prompt_only() -> None:
    workload = Workload(1, 1000, 3, 500, 100)
    assert workload.max_context_tokens(template_tokens=10) == 1000 + 10 + 3 * 500 + 2 * 100
