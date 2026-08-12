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

import logging
import os
import pathlib
import threading
from typing import Any, Sequence
from uuid import uuid4

import yaml

from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.core.scheduler import Scheduler
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.poller import MetricsPoller
from py_inference_scheduler.datalayer.metrics.vime.vllm import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)

ENV_CONFIG = "ROUTER_CONFIG_PATH"
ENV_METRICS_INTERVAL_MS = "RLS_METRICS_INTERVAL_MS"
ENV_SESSION_HEADER = "RLS_SESSION_HEADER"
DEFAULT_SESSION_HEADER = "x-rls-session-id"


class _GatewayLoadView(InflightStore):
    """Inflight counts mirrored from the gateway's own active_counts.

    The gateway increments on route() and decrements on release(), so we
    mirror its counters on every select_worker call instead of bookkeeping
    our own. increment/decrement stay unused no-ops from the parent.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loads: dict[str, int] = {}
        self._loads_lock = threading.Lock()

    def merge(self, loads: dict[str, int]) -> None:
        with self._loads_lock:
            self._loads.update(loads)

    def prune(self, urls: Sequence[str]) -> None:
        with self._loads_lock:
            for url in urls:
                self._loads.pop(url, None)

    def get(self, endpoint_name: str) -> int:
        with self._loads_lock:
            return self._loads.get(endpoint_name, 0)

    def get_all(self) -> dict[str, int]:
        with self._loads_lock:
            return dict(self._loads)


class SchedulerRoutingPolicy:
    """rllm-model-gateway RoutingPolicy backed by py-inference-scheduler.

    Loaded by the gateway's ``_load_policy`` (argument-free), so configuration
    comes from env: ROUTER_CONFIG_PATH (scheduler yaml, required),
    RLS_METRICS_INTERVAL_MS (poller interval, default 100) and
    RLS_SESSION_HEADER (header name carrying the gateway session id into
    LLMRequest for the sticky_session scorer, default x-rls-session-id).

    Workers are duck-typed gateway WorkerInfo objects (``.url`` attribute);
    rllm is deliberately not imported. select_worker never raises: any
    scheduler failure falls back to least-loaded so training cannot stall
    on a policy bug.
    """

    def __init__(self) -> None:
        config_path = os.environ.get(ENV_CONFIG)
        if not config_path:
            raise ValueError(f"{ENV_CONFIG} must point to a scheduler yaml config")
        with pathlib.Path(config_path).open(encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        if not isinstance(config_dict, dict):
            raise TypeError("Parsed configuration is not a valid dictionary.")
        config = SchedulerConfig.from_dict(config_dict)
        logger.info("Loaded scheduler config: %s", config)

        self._scheduler = Scheduler.new_with_config(config)
        self._session_header = os.environ.get(ENV_SESSION_HEADER, DEFAULT_SESSION_HEADER)
        self._lock = threading.Lock()
        self._endpoints: dict[str, Endpoint] = {}
        self._loads = _GatewayLoadView()
        interval_ms = int(os.environ.get(ENV_METRICS_INTERVAL_MS, "100"))
        self._poller = MetricsPoller(
            self.list_endpoints, self._loads, fetch_worker_metrics, interval_ms=interval_ms
        )
        self._poller.start()

    def list_endpoints(self) -> list[Endpoint]:
        with self._lock:
            return list(self._endpoints.values())

    # -- RoutingPolicy protocol ---------------------------------------------

    def select_worker(self, workers: list[Any], session_id: str | None, active_counts: dict[str, int]) -> Any:
        try:
            with self._lock:
                candidates = self._sync_endpoints(workers, active_counts)
                headers = {self._session_header: session_id} if session_id else {}
                request = LLMRequest(
                    request_id=uuid4().hex,
                    target_model=None,
                    headers=headers,
                    body=None,
                )
                scored = self._scheduler.run(request, candidates)
            if scored:
                winner_url = scored[0].endpoint.name
                for worker in workers:
                    if worker.url == winner_url:
                        return worker
                logger.warning("Winner %s not in offered worker set", winner_url)
        except Exception:
            logger.exception("Selection failed, falling back to least-loaded")
        return min(workers, key=lambda w: active_counts.get(w.url, 0))

    def on_worker_change(self, workers: list[Any]) -> None:
        """Gateway add/remove signal: replace the endpoint registry."""
        with self._lock:
            current = {str(w.url) for w in workers}
            removed = [url for url in self._endpoints if url not in current]
            for url in removed:
                self._endpoints.pop(url, None)
                logger.info("Dropping worker %s", url)
            self._loads.prune(removed)
            for worker in workers:
                self._track(str(worker.url))

    # -- Internal ------------------------------------------------------------

    def _track(self, url: str) -> Endpoint:
        endpoint = self._endpoints.get(url)
        if endpoint is None:
            endpoint = Endpoint(
                name=url,
                attributes={
                    "url": url,
                    "worker_type": "regular",
                    "queue_len": 0,
                    "routing_stats": {},
                },
            )
            self._endpoints[url] = endpoint
            logger.info("Tracking worker %s", url)
        return endpoint

    def _sync_endpoints(self, workers: Sequence[Any], active_counts: dict[str, int]) -> list[Endpoint]:
        candidates: list[Endpoint] = []
        loads: dict[str, int] = {}
        for worker in workers:
            url = str(worker.url)
            endpoint = self._track(url)
            load = int(active_counts.get(url, 0))
            endpoint.attributes["queue_len"] = load
            loads[url] = load
            candidates.append(endpoint)
        self._loads.merge(loads)
        return candidates
