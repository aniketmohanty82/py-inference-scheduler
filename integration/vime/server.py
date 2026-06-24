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

"""vime router (vLLM dialect).

vime is a fork of slime that swaps the rollout backend to vLLM + vllm-router.
The engine-neutral router machinery is reused from ``integration.slime.server``
(``safe_json``, ``lifespan``, ``schedule_and_proxy``, the base ``WorkerRegistry``);
everything vLLM-specific lives here: deregistration by url + ``/list_workers``,
the ``token_ids`` routing key, the ``/inference/v1/generate`` endpoint, and the
vLLM Prometheus scrape.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import unquote

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from datalayer.metrics.verl.datastore import InflightStore
from datalayer.metrics.vime.vllm import fetch_worker_metrics
from integration.slime.server import (
    WorkerRegistry,
    lifespan,
    safe_json,
    schedule_and_proxy,
)
from scheduling import Scheduler

logger = logging.getLogger(__name__)

# vime's rollout posts generation here; the engine serves the same path.
_GENERATE_PATH = "/inference/v1/generate"


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
    """Return the token ids vime routes on (the /inference/v1/generate payload)."""
    return body.get("token_ids", [])


def create_app(scheduler: Scheduler) -> FastAPI:
    """Build the vime router FastAPI app around a configured Scheduler."""
    registry = VimeWorkerRegistry()
    inflight = InflightStore()
    scheduling_lock = asyncio.Lock()

    app = FastAPI(title="vime sampling router", lifespan=lifespan)

    @app.post("/workers")
    async def add_worker(request: Request) -> JSONResponse:
        body = safe_json(await request.body())
        url = body.get("url")
        if not url:
            return JSONResponse(status_code=400, content={"error": "missing 'url'"})
        worker_id = registry.add(str(url), str(body.get("worker_type", "regular")))
        logger.info("Registered worker %s -> %s", worker_id, url)
        return JSONResponse(content={"status": "success", "id": worker_id, "url": url})

    @app.get("/workers")
    async def list_workers() -> JSONResponse:
        return JSONResponse(content={"workers": registry.list_workers_as_dicts()})

    @app.get("/list_workers")
    async def list_worker_urls() -> JSONResponse:
        # vime's abort path probes /list_workers to enumerate engines.
        return JSONResponse(content={"urls": registry.urls()})

    @app.delete("/workers/{worker_ref:path}")
    async def delete_worker(worker_ref: str) -> JSONResponse:
        # vime deregisters by url-encoded url (vllm_engine.py); fall back to id.
        ref = unquote(worker_ref)
        if registry.remove_by_url(ref) or registry.remove(ref):
            logger.info("Deregistered worker %s", ref)
            return JSONResponse(content={"status": "success"})
        return JSONResponse(status_code=404, content={"status": "not_found"})

    @app.post(_GENERATE_PATH)
    async def generate(request: Request) -> Response:
        return await schedule_and_proxy(
            request,
            registry=registry,
            inflight=inflight,
            scheduling_lock=scheduling_lock,
            scheduler=scheduler,
            fetch_metrics=fetch_worker_metrics,
            routing_body=_routing_body,
            generate_path=_GENERATE_PATH,
        )

    return app
