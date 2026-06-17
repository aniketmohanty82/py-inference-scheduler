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

"""External HTTP router for slime (v0.3.0), backed by ``scheduling.Scheduler``.

slime skips launching its own sgl-router when ``--sglang-router-ip/port`` point
here. Its engines then self-register and POST generations to us over the
sgl-router ``/workers`` + ``/generate`` HTTP surface. This module exposes
that surface and delegates the routing decision to the py-inference-scheduler
engine. slime owns the rollout lifecycle (batching, partial rollout, aborts);
the router only owns which worker serves this request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from datalayer.metrics.slime.sglang import fetch_worker_metrics
from datalayer.metrics.verl.datastore import InflightStore
from scheduling import Scheduler
from scheduling.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)

# Headers that must not be forwarded verbatim when proxying to a worker.
_HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}


class WorkerRegistry:
    """In-memory directory of registered SGLang workers (the routing pool).

    The engines (owned by slime) announce themselves via ``POST /workers``; the
    router never creates or manages them, it only keeps their addresses so the
    scheduler has candidates to choose from.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Endpoint] = {}
        self._url_to_id: dict[str, str] = {}

    def add(self, url: str, worker_type: str = "regular") -> str:
        existing = self._url_to_id.get(url)
        if existing is not None:
            return existing
        worker_id = uuid.uuid4().hex
        self._by_id[worker_id] = Endpoint(
            name=worker_id,
            attributes={"url": url, "worker_type": worker_type, "routing_stats": {}},
        )
        self._url_to_id[url] = worker_id
        return worker_id

    def remove(self, worker_id: str) -> bool:
        ep = self._by_id.pop(worker_id, None)
        if ep is None:
            return False
        self._url_to_id.pop(str(ep.attributes.get("url")), None)
        return True

    def list(self) -> list[dict[str, str]]:
        return [{"url": str(ep.attributes["url"]), "id": ep.name} for ep in self._by_id.values()]

    def endpoints(self) -> list[Endpoint]:
        return list(self._by_id.values())


def _safe_json(raw: bytes) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _routing_body(body: dict) -> object:
    """The content the scorers route on: prefer token ids, fall back to text."""
    ids = body.get("input_ids")
    if ids:
        return ids
    return body.get("text", "")


def create_app(scheduler: Scheduler) -> FastAPI:
    """Build the slime router FastAPI app around a configured ``Scheduler``."""
    registry = WorkerRegistry()
    inflight = InflightStore()
    # Serialises scheduling decisions so each request in slime's synchronous
    # burst sees the prior request's inflight increment (and a fresh metrics
    # scrape) instead of all piling onto one worker.
    scheduling_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # aiohttp's default connector caps the (shared) session at 100 total connections
        # and queues the rest, which throttles a rollout burst to ~100/num_engines in-flight.
        # limit=0 removes the cap so requests fan out across all engines.
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            app.state.http = session
            yield

    app = FastAPI(title="py-inference-scheduler (slime router)", lifespan=lifespan)

    # ---- worker registry (sgl-router /workers API used by slime v0.3.0) ----
    @app.post("/workers")
    async def add_worker(request: Request) -> JSONResponse:
        body = _safe_json(await request.body())
        url = body.get("url")
        if not url:
            return JSONResponse(status_code=400, content={"error": "missing 'url'"})
        worker_id = registry.add(str(url), str(body.get("worker_type", "regular")))
        logger.info("Registered worker %s -> %s", worker_id, url)
        return JSONResponse(content={"status": "success", "id": worker_id, "url": url})

    @app.get("/workers")
    async def list_workers() -> JSONResponse:
        return JSONResponse(content={"workers": registry.list()})

    @app.delete("/workers/{worker_id}")
    async def delete_worker(worker_id: str) -> JSONResponse:
        if registry.remove(worker_id):
            logger.info("Deregistered worker %s", worker_id)
            return JSONResponse(content={"status": "success"})
        return JSONResponse(status_code=404, content={"status": "not_found"})

    # ---- generation: the only scheduled endpoint ----
    @app.post("/generate")
    async def generate(request: Request) -> Response:
        raw = await request.body()
        llm_req = LLMRequest(request_id=uuid.uuid4().hex, body=_routing_body(_safe_json(raw)))

        endpoints = registry.endpoints()
        if not endpoints:
            return JSONResponse(status_code=503, content={"error": "no workers registered"})

        session: aiohttp.ClientSession = request.app.state.http

        # Metrics scrape + scheduling happen ON the request path under a lock:
        # slime fires the whole step as one synchronous burst
        async with scheduling_lock:
            await asyncio.gather(*[fetch_worker_metrics(ep, inflight, session) for ep in endpoints])
            selected = scheduler.run(llm_req, candidates=endpoints)
            if not selected:
                return JSONResponse(status_code=503, content={"error": "no worker selected"})
            winner = selected[0].endpoint
            inflight.increment(winner.name)

        worker_url = str(winner.attributes["url"])
        fwd_headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        try:
            async with session.post(
                f"{worker_url}/generate", data=raw, headers=fwd_headers
            ) as resp:
                content = await resp.read()
                media_type = resp.headers.get("content-type", "application/json")
                return Response(content=content, status_code=resp.status, media_type=media_type)
        except Exception:
            logger.exception("Proxy to worker %s failed", worker_url)
            return JSONResponse(status_code=502, content={"error": "worker request failed"})
        finally:
            inflight.decrement(winner.name)

    return app
