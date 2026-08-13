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

"""wants_offload tag/offload-all semantics (vllm stubbed: engine-only dep)."""

import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def overschedule(monkeypatch):
    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        config = types.ModuleType("vllm.config")
        config.VllmConfig = object
        vllm_logger = types.ModuleType("vllm.logger")
        vllm_logger.init_logger = lambda name: __import__("logging").getLogger(name)
        sched = types.ModuleType("vllm.v1.core.sched.scheduler")
        sched.Scheduler = object
        request_mod = types.ModuleType("vllm.v1.request")
        request_mod.Request = object
        for name, mod in {
            "vllm": vllm,
            "vllm.config": config,
            "vllm.logger": vllm_logger,
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.core": types.ModuleType("vllm.v1.core"),
            "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
            "vllm.v1.core.sched.scheduler": sched,
            "vllm.v1.request": request_mod,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)
    from py_inference_scheduler.datalayer.connectors.mooncake import overschedule

    return overschedule


def _request(extra_args=None, *, pooling=False):
    if pooling:
        return SimpleNamespace(sampling_params=None)
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args))


def test_untagged_request_not_offloaded(overschedule):
    assert overschedule.wants_offload(_request()) is False


def test_tagged_request_offloaded(overschedule):
    request = _request({overschedule.OFFLOAD_TAG_KEY: "1"})
    assert overschedule.wants_offload(request) is True


def test_offload_all_covers_untagged(overschedule):
    assert overschedule.wants_offload(_request(), offload_all=True) is True


def test_offload_all_never_touches_pooling(overschedule):
    assert overschedule.wants_offload(_request(pooling=True), offload_all=True) is False


def test_offload_all_env_gate(overschedule, monkeypatch):
    monkeypatch.delenv(overschedule.ENV_OFFLOAD_ALL, raising=False)
    assert overschedule.offload_all_from_env() is False
    monkeypatch.setenv(overschedule.ENV_OFFLOAD_ALL, "1")
    assert overschedule.offload_all_from_env() is True
    monkeypatch.setenv(overschedule.ENV_OFFLOAD_ALL, "0")
    assert overschedule.offload_all_from_env() is False
