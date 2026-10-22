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

import os
import textwrap

from py_inference_scheduler import Scheduler

WITH_FLOW_CONTROL = textwrap.dedent(
    """
    profile_handler:
      type: single_profile
    profiles:
      default:
        flow_control:
          type: kv_saturation
        scorers:
          - type: least_queue
            weight: 1.0
        picker:
          type: max_score
    """
)


def test_sees_flow_control_before_the_first_decision(tmp_path):
    config_file = tmp_path / "scheduler.yaml"
    config_file.write_text(WITH_FLOW_CONTROL, encoding="utf-8")

    assert Scheduler(config_path=str(config_file)).has_flow_control()


def test_listing_plugins_leaves_reloads_to_schedule(tmp_path):
    config_file = tmp_path / "scheduler.yaml"
    config_file.write_text(WITH_FLOW_CONTROL, encoding="utf-8")
    scheduler = Scheduler(config_path=str(config_file))
    assert scheduler.has_flow_control()

    without = WITH_FLOW_CONTROL.replace("    flow_control:\n      type: kv_saturation\n", "")
    config_file.write_text(without, encoding="utf-8")
    os.utime(config_file, (scheduler.last_mtime + 2, scheduler.last_mtime + 2))

    assert scheduler.has_flow_control()
