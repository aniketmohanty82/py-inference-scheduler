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
import logging
from urllib.parse import unquote

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from integration.slime.server import (
    WorkerRegistry,
    make_lifespan,
    register_worker_routes,
    schedule_and_proxy,
)
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.flow_control import FlowControlManager
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.vime.vllm import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint

logger = logging.getLogger(__name__)


class VimeWorkerRegistry(WorkerRegistry):
    """vLLM/vllm-router deregisters engines by url and lists their worker_type."""

    def remove_by_url(self, url: str) -> bool:
        worker_id = self._url_to_id.get(url)
        if worker_id is None:
            return False
        return self.remove(worker_id)

    def urls(self) -> list[str]:
        return [str(ep.attributes["url"]) for ep in self._by_id.values()]

    def list_workers_as_dicts(self) -> list[dict[str, str]]:
        return [
            {
                "url": str(ep.attributes["url"]),
                "id": ep.name,
                "worker_type": str(ep.attributes.get("worker_type", "regular")),
            }
            for ep in self._by_id.values()
        ]


def _routing_body(body: dict) -> object:
    """Obtain prompt token_ids from the generate payload for prefix cache routing."""
    return body.get("token_ids", [])


def create_app(scheduler: Scheduler, flow_poll_interval_s: float = 0.1) -> FastAPI:
    """Build the vime router FastAPI app around a configured Scheduler."""
    registry = VimeWorkerRegistry()
    inflight = InflightStore()
    scheduling_lock = asyncio.Lock()

    async def _scraped_endpoints() -> list[Endpoint]:
        # vime has no background poller: the flow-control watcher scrapes itself.
        endpoints = registry.endpoints()
        if endpoints:
            session = app.state.http
            await asyncio.gather(
                *[fetch_worker_metrics(ep, inflight, session) for ep in endpoints]
            )
        return endpoints

    flow_control = FlowControlManager(
        scheduler.get_flow_control_plugins,
        _scraped_endpoints,
        poll_interval_s=flow_poll_interval_s,
    )

    app = FastAPI(title="vime sampling router", lifespan=make_lifespan(flow_control))
    app.state.flow_control = flow_control

    # adds /workers endpoints
    register_worker_routes(app, registry)

    @app.get("/list_workers")
    async def list_worker_urls() -> JSONResponse:
        # vime's abort path probes /list_workers to enumerate engines.
        return JSONResponse(content={"urls": registry.urls()})

    @app.delete("/workers/{worker_ref:path}")
    async def delete_worker(worker_ref: str) -> JSONResponse:
        # vime deregisters by the worker's (percent-encoded) url, not our id.
        ref = unquote(worker_ref)
        if registry.remove_by_url(ref) or registry.remove(ref):
            logger.info("Deregistered worker %s", ref)
            return JSONResponse(content={"status": "success"})
        return JSONResponse(status_code=404, content={"status": "not_found"})

    @app.post("/inference/v1/generate")
    async def generate(request: Request) -> Response:
        return await schedule_and_proxy(
            request,
            registry=registry,
            inflight=inflight,
            scheduling_lock=scheduling_lock,
            scheduler=scheduler,
            fetch_metrics=fetch_worker_metrics,
            routing_body=_routing_body,
            generate_path="/inference/v1/generate",
            flow_control=flow_control,
        )

    return app
