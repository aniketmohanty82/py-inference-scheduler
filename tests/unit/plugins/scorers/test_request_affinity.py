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

from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.framework import CycleState, Endpoint, LLMRequest, build_scorer
from py_inference_scheduler.plugins import RequestAffinityScorer


def _pods(*names: str) -> dict[str, Endpoint]:
    return {name: Endpoint(name=name) for name in names}


def _route(scorer: RequestAffinityScorer, request_id: str, endpoint: Endpoint) -> None:
    scorer.pre_request(CycleState(), LLMRequest(request_id=request_id), endpoint)


def _scores(
    scorer: RequestAffinityScorer, request_id: str, pods: dict[str, Endpoint]
) -> dict[str, float]:
    return scorer.score(CycleState(), LLMRequest(request_id=request_id), pods)


def test_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        RequestAffinityScorer(capacity=0)


def test_is_registered():
    assert isinstance(build_scorer("request_affinity"), RequestAffinityScorer)


def test_unseen_request_id_casts_no_vote():
    scores = _scores(RequestAffinityScorer(), "traj-1", _pods("ep1", "ep2"))
    assert scores == {"ep1": 0.0, "ep2": 0.0}


def test_votes_for_the_last_endpoint_and_follows_a_move():
    scorer = RequestAffinityScorer()
    pods = _pods("ep1", "ep2", "ep3")

    _route(scorer, "traj-1", pods["ep2"])
    assert _scores(scorer, "traj-1", pods) == {"ep1": 0.0, "ep2": 1.0, "ep3": 0.0}

    _route(scorer, "traj-1", pods["ep3"])
    assert _scores(scorer, "traj-1", pods) == {"ep1": 0.0, "ep2": 0.0, "ep3": 1.0}


def test_holder_outside_the_candidates_casts_no_vote():
    scorer = RequestAffinityScorer()
    _route(scorer, "traj-1", Endpoint(name="ep9"))
    assert _scores(scorer, "traj-1", _pods("ep1", "ep2")) == {"ep1": 0.0, "ep2": 0.0}


def test_votes_are_per_request_id():
    scorer = RequestAffinityScorer()
    pods = _pods("ep1", "ep2")
    _route(scorer, "traj-1", pods["ep1"])
    assert _scores(scorer, "traj-2", pods) == {"ep1": 0.0, "ep2": 0.0}


def test_forgets_the_least_recently_routed_first():
    scorer = RequestAffinityScorer(capacity=2)
    pods = _pods("ep1", "ep2")
    _route(scorer, "traj-1", pods["ep1"])
    _route(scorer, "traj-2", pods["ep2"])
    _route(scorer, "traj-1", pods["ep1"])  # traj-1 becomes the most recent
    _route(scorer, "traj-3", pods["ep2"])  # so traj-2 is the one forgotten

    assert _scores(scorer, "traj-1", pods)["ep1"] == 1.0
    assert _scores(scorer, "traj-2", pods) == {"ep1": 0.0, "ep2": 0.0}
    assert _scores(scorer, "traj-3", pods)["ep2"] == 1.0


def test_scheduler_returns_a_turn_to_the_endpoint_that_served_the_last_one():
    config = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "p": {
                "scorers": [
                    {"type": "request_affinity", "weight": 2.0},
                    {"type": "least_queue", "weight": 1.0},
                ],
                "picker": {"type": "max_score"},
            }
        },
    }
    scheduler = Scheduler.new_with_config(SchedulerConfig.from_dict(config))
    eps = [Endpoint(name=name, attributes={"queue_len": 0}) for name in ("a", "b")]

    first = scheduler.run(LLMRequest(request_id="traj"), candidates=eps)[0].endpoint.name
    for ep in eps:
        ep.attributes["queue_len"] = 1 if ep.name == first else 0
    second = scheduler.run(LLMRequest(request_id="traj"), candidates=eps)[0].endpoint.name

    assert second == first
