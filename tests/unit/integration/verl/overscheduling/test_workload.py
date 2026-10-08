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
        assert row["extra_info"] == {"output_tokens": [128] * 4, "reply_tokens": [32] * 3}
        assert len(row["raw_prompt"][0]["content"].split()) == 20 + 2


def test_max_context_counts_prompt_turns_and_replies_with_template_slack() -> None:
    workload = Workload(1, 1000, 3, 500, 100)
    assert workload.max_context_tokens(template_tokens=10) == 1000 + 10 + 3 * 500 + 2 * 110
