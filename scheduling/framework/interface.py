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

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .types import CycleState, Endpoint, LLMRequest, ProfileRunResult, ScoredEndpoint


class FilterPlugin(Protocol):
    def filter(
        self,
        request: LLMRequest,
        cycle_state: CycleState,
        endpoints: Sequence[Endpoint],
    ) -> Sequence[Endpoint]:
        ...


class ScorerPlugin(Protocol):
    def score(
        self, request: LLMRequest, endpoints: Sequence[Endpoint]
    ) -> Mapping[str, float]:
        ...


class PickerPlugin(Protocol):
    def pick(
        self,
        request: LLMRequest,
        cycle_state: CycleState,
        endpoints: Sequence[ScoredEndpoint],
    ) -> ScoredEndpoint | None:
        ...


class ProfileHandler(Protocol):
    def pick(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        profiles: Mapping[str, SchedulerProfile],
        profile_results: Mapping[str, ProfileRunResult | None],
    ) -> Mapping[str, SchedulerProfile]:
        ...

    def process_results(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        profile_results: Mapping[str, ProfileRunResult | None],
    ) -> str | None:
        ...


@dataclass
class WeightedScorer:
    scorer: ScorerPlugin
    weight: float = 1.0


@dataclass
class SchedulerProfile:
    name: str
    filters: list[FilterPlugin] = field(default_factory=list)
    scorers: list[WeightedScorer] = field(default_factory=list)
    picker: PickerPlugin | None = None
    flow_control: dict[str, Any] = field(default_factory=dict)

    def with_filters(self, *fs: FilterPlugin) -> SchedulerProfile:
        self.filters.extend(fs)
        return self

    def with_scorers(self, *ss: WeightedScorer) -> SchedulerProfile:
        self.scorers.extend(ss)
        return self

    def with_picker(self, p: PickerPlugin) -> SchedulerProfile:
        self.picker = p
        return self

    def run(
        self,
        request: LLMRequest,
        cycle_state: CycleState,
        endpoints: Sequence[Endpoint],
    ) -> ProfileRunResult:
        # 1. run filters
        filtered = endpoints
        for f in self.filters:
            filtered = f.filter(request, cycle_state, filtered)

        # map to quickly find endpoints by name
        endpoints_map = {e.name: e for e in endpoints}

        # 2. run scorers
        total_scores = {}
        for ws in self.scorers:
            scores = ws.scorer.score(request, filtered)
            for name, score in scores.items():
                total_scores[name] = (
                    total_scores.get(name, 0.0) + score * ws.weight
                )

        # create ScoredEndpoint list preserving endpoint info
        scored = [
            ScoredEndpoint(
                endpoint=endpoints_map[name], score=total_scores.get(name, 0.0)
            )
            for name in endpoints_map
        ]
        # sort descending
        scored.sort(key=lambda sp: sp.score, reverse=True)

        # 3. run picker
        if self.picker and len(scored) > 0:
            picked = self.picker.pick(request, cycle_state, scored)
        else:
            picked = scored[0] if len(scored) > 0 else None

        return ProfileRunResult(
            profile_name=self.name, scores=total_scores, endpoint=picked
        )
