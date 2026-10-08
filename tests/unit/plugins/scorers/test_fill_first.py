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

import pytest

from py_inference_scheduler.framework import CycleState, Endpoint, LLMRequest
from py_inference_scheduler.plugins import FillFirstScorer


class TestFillFirstScorer:
    def setup_method(self):
        self.scorer = FillFirstScorer()
        self.cycle_state = CycleState()
        self.request = LLMRequest(request_id="test_req")

    def test_the_busiest_endpoint_scores_highest(self):
        endpoints = {
            "ep_idle": Endpoint(name="ep_idle", attributes={"queue_len": 5}),
            "ep_mid": Endpoint(name="ep_mid", attributes={"queue_len": 15}),
            "ep_busy": Endpoint(name="ep_busy", attributes={"queue_len": 25}),
        }

        scores = self.scorer.score(self.cycle_state, self.request, endpoints)

        assert scores == {"ep_idle": 0.0, "ep_mid": 0.5, "ep_busy": 1.0}

    def test_score_all_equal(self):
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={"queue_len": 12}),
            "ep2": Endpoint(name="ep2", attributes={"queue_len": 12}),
        }

        scores = self.scorer.score(self.cycle_state, self.request, endpoints)

        assert scores == {"ep1": 1.0, "ep2": 1.0}


if __name__ == "__main__":
    pytest.main([__file__])
