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

from scheduling.framework import CycleState, Endpoint, LLMRequest, ScoredEndpoint
from scheduling.plugins.pickers.generic import MaxScorePicker


def _scored(name: str, score: float) -> ScoredEndpoint:
    return ScoredEndpoint(endpoint=Endpoint(name=name), score=score)


def test_unique_max_always_wins():
    picker = MaxScorePicker()
    eps = [_scored("a", 3.0), _scored("b", 2.0), _scored("c", 1.0)]
    req = LLMRequest(request_id="r")
    for _ in range(50):
        assert picker.pick(CycleState(), req, eps).endpoint.name == "a"


def test_exact_ties_break_randomly():
    picker = MaxScorePicker()
    eps = [_scored("a", 1.0), _scored("b", 1.0), _scored("c", 1.0), _scored("d", 0.5)]
    req = LLMRequest(request_id="r")
    winners = {picker.pick(CycleState(), req, eps).endpoint.name for _ in range(200)}
    assert winners == {"a", "b", "c"}


def test_empty_returns_none():
    assert MaxScorePicker().pick(CycleState(), LLMRequest(request_id="r"), []) is None
