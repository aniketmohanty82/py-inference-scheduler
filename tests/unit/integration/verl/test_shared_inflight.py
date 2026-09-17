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

import asyncio
import time

import ray

from integration.verl.shared_inflight import SharedInflightLedger


async def test_degrades_to_none_without_ray():
    """Outside a cluster the ledger is inert and callers keep local counts."""
    assert not ray.is_initialized()
    ledger = SharedInflightLedger()
    ledger.increment("e1")
    ledger.decrement("e1")
    assert await ledger.snapshot() is None


async def test_ledgers_in_separate_instances_share_counts():
    """Two ledger instances (one per AgentLoopWorker in prod) see one total.

    Increments are fire-and-forget, so the assertion polls until the actor
    has applied them rather than reading immediately.
    """
    ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    try:
        worker_a = SharedInflightLedger()
        worker_b = SharedInflightLedger()
        worker_a.increment("e1")
        worker_b.increment("e1")
        worker_b.increment("e2")
        worker_b.increment("e2")
        worker_b.decrement("e2")

        expected = {"e1": 2, "e2": 1}
        deadline = time.monotonic() + 15
        snap: dict[str, int] | None = None
        while time.monotonic() < deadline:
            snap = await worker_a.snapshot()
            if snap == expected:
                break
            await asyncio.sleep(0.05)
        assert snap == expected
    finally:
        ray.shutdown()
