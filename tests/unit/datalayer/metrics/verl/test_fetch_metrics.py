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

from types import SimpleNamespace

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.verl.fetch_metrics import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint


async def test_a_failed_scrape_keeps_the_last_capacity():
    reports = iter([{"kv_cache_size": 157088}, {"kv_cache_size": 0, "error": "timeout"}])

    async def get_routing_stats() -> dict[str, object]:
        return next(reports)

    server = SimpleNamespace(get_routing_stats=SimpleNamespace(remote=get_routing_stats))
    engine = Endpoint(name="e1", attributes={"replica_obj": server})
    for _ in range(2):
        await fetch_worker_metrics(engine, InflightStore())

    assert engine.attributes["kv_cache_size"] == 157088
    assert engine.attributes["routing_stats"]["error"] == "timeout"
