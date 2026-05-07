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

import pathlib
from typing import Any, Sequence

import yaml

from scheduling.core.config import SchedulerConfig
from scheduling.framework import (
    CycleState,
    Endpoint,
    LLMRequest,
    ProfileRunResult,
    SchedulerProfile,
    SchedulingResult,
    ScoredEndpoint,
)


class Scheduler:
    def __init__(self) -> None:
        self.profiles = {}
        self.profile_handler = None
        self.config_path = None
        self.last_mtime = 0

    @classmethod
    def from_config(cls, config_path: str) -> Scheduler:
        instance = cls()
        instance.config_path = config_path
        instance._maybe_reload_config()
        return instance

    @classmethod
    def new_with_config(cls, config: SchedulerConfig) -> Scheduler:
        instance = cls()
        instance.profile_handler = config.profile_handler
        instance.profiles = config.profiles
        return instance

    def get_flow_control_config(self) -> dict[str, Any]:
        """Returns the flow_control configuration from the primary profile, or an empty dict."""
        self._maybe_reload_config()
        if hasattr(self, "profiles") and self.profiles:
            primary_profile = next(iter(self.profiles.values()))
            return getattr(primary_profile, "flow_control", {})
        return {}

    def _maybe_reload_config(self) -> None:
        if self.config_path is None:
            return
        mtime = pathlib.Path(self.config_path).stat().st_mtime
        if mtime > self.last_mtime:
            print(f"Reloading scheduler config from {self.config_path}")
            with pathlib.Path(self.config_path).open(encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                raise ValueError("Parsed configuration is not a valid dictionary.")
            config = SchedulerConfig.from_dict(config_dict)
            self.profile_handler = config.profile_handler
            self.profiles = config.profiles
            self.last_mtime = mtime

    def schedule(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> SchedulingResult:
        if not candidates:
            raise ValueError("no scheduling candidates provided")

        cycle_state = CycleState()
        profile_results: dict[str, ProfileRunResult | None] = {}

        # ask profile handler which profiles to run
        selected = self.profile_handler.pick(
            cycle_state, request, self.profiles, profile_results
        )
        assert selected is not None  # noqa: S101

        def run_profile(
            profile_name: str, profile: SchedulerProfile
        ) -> ProfileRunResult | None:
            try:
                return profile.run(request, cycle_state, candidates)
            except Exception as e:  # noqa: BLE001
                print(f"Error running profile {profile_name}: ")
                print(repr(e))
                return None

        for name, profile in selected.items():
            profile_results[name] = run_profile(name, profile)

        primary = self.profile_handler.process_results(
            cycle_state, request, profile_results
        )

        # Build SchedulingResult
        result = SchedulingResult(
            profile_results={k: v or ProfileRunResult() for k, v in profile_results.items()},
            primary_profile_name=primary,
        )

        selected_eps = (
            result.profile_results[primary].endpoint_list[:1]
            if primary in result.profile_results
            else []
        )
        print(f"Selected endpoint {selected_eps}")
        if selected_eps:
            for w in self.profiles[primary].scorers:
                if hasattr(w.scorer, "pre_request"):
                    w.scorer.pre_request(cycle_state, request, selected_eps[0].endpoint)

        return result

    def run(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[ScoredEndpoint]:
        scheduler_output = self.schedule(request, candidates)
        profile_name = scheduler_output.primary_profile_name
        profile_results = scheduler_output.profile_results.get(profile_name)

        print(f"Profile {profile_name} results: {profile_results}")
        if profile_results and profile_results.endpoint_list:
            return profile_results.endpoint_list[:1]
        print("No endpoint selected, defaulting to framework routing logic")
        return []
