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

from py_inference_scheduler.framework import CycleState, Endpoint, LLMRequest, build_scorer
from py_inference_scheduler.plugins import RequestAffinityScorer


def _pods(*names: str) -> dict[str, Endpoint]:
    return {name: Endpoint(name=name) for name in names}


def _request(request_id: str) -> LLMRequest:
    return LLMRequest(request_id=request_id)


def test_request_affinity_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        RequestAffinityScorer(capacity=0)


def test_request_affinity_is_registered():
    assert isinstance(build_scorer("request_affinity"), RequestAffinityScorer)


def test_request_affinity_first_turn_casts_no_vote():
    scorer = RequestAffinityScorer()
    pods = _pods("ep1", "ep2")

    assert scorer.score(CycleState(), _request("traj-1"), pods) == {"ep1": 0.0, "ep2": 0.0}
    assert scorer.holder("traj-1") is None


def test_request_affinity_follows_the_last_selected_endpoint():
    scorer = RequestAffinityScorer()
    pods = _pods("ep1", "ep2", "ep3")

    scorer.pre_request(CycleState(), _request("traj-1"), pods["ep2"])
    assert scorer.score(CycleState(), _request("traj-1"), pods) == {
        "ep1": 0.0,
        "ep2": 1.0,
        "ep3": 0.0,
    }

    # a migration (whatever caused it) moves the vote, it does not bounce back
    scorer.pre_request(CycleState(), _request("traj-1"), pods["ep3"])
    assert scorer.score(CycleState(), _request("traj-1"), pods) == {
        "ep1": 0.0,
        "ep2": 0.0,
        "ep3": 1.0,
    }
    assert scorer.holder("traj-1") == "ep3"


def test_request_affinity_ignores_a_holder_that_is_not_a_candidate():
    scorer = RequestAffinityScorer()
    pods = _pods("ep1", "ep2")
    scorer.pre_request(CycleState(), _request("traj-1"), Endpoint(name="ep9"))

    assert scorer.score(CycleState(), _request("traj-1"), pods) == {"ep1": 0.0, "ep2": 0.0}


def test_request_affinity_is_per_request_id():
    scorer = RequestAffinityScorer()
    pods = _pods("ep1", "ep2")
    scorer.pre_request(CycleState(), _request("traj-1"), pods["ep1"])

    assert scorer.score(CycleState(), _request("traj-2"), pods) == {"ep1": 0.0, "ep2": 0.0}


def test_request_affinity_evicts_least_recently_routed():
    scorer = RequestAffinityScorer(capacity=2)
    pods = _pods("ep1", "ep2")
    scorer.pre_request(CycleState(), _request("traj-1"), pods["ep1"])
    scorer.pre_request(CycleState(), _request("traj-2"), pods["ep2"])
    # traj-1 routed again: it becomes the most recent, so traj-2 is evicted next
    scorer.pre_request(CycleState(), _request("traj-1"), pods["ep1"])
    scorer.pre_request(CycleState(), _request("traj-3"), pods["ep2"])

    assert scorer.holder("traj-1") == "ep1"
    assert scorer.holder("traj-2") is None
    assert scorer.holder("traj-3") == "ep2"
