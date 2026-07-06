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
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from datalayer.metrics.datastore import InflightStore
from datalayer.metrics.slime.sglang import fetch_worker_metrics
from scheduling import Scheduler
from scheduling.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)

# Headers that must not be forwarded verbatim when proxying to a worker.
_HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}


class WorkerRegistry:
    """In-memory directory of registered SGLang workers."""

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

    def list_workers_as_dicts(self) -> list[dict[str, str]]:
        return [{"url": str(ep.attributes["url"]), "id": ep.name} for ep in self._by_id.values()]

    def endpoints(self) -> list[Endpoint]:
        return list(self._by_id.values())


def _safe_json(raw: bytes) -> dict:
    """Parse request bytes to a dict; {} if empty/invalid/non-dict (never raises)."""
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
    """Build the slime router FastAPI app around a configured Scheduler."""
    registry = WorkerRegistry()
    inflight = InflightStore()
    scheduling_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # limit=0 removes aiohttp's default 100-connection cap so requests fan out across engines.
        connector = aiohttp.TCPConnector(limit=0)
        # total=None lifts aiohttp's default 300s cap per generation
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            app.state.http = session
            yield

    app = FastAPI(title="slime sampling router", lifespan=lifespan)

    # worker registry
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
        return JSONResponse(content={"workers": registry.list_workers_as_dicts()})

    @app.delete("/workers/{worker_id}")
    async def delete_worker(worker_id: str) -> JSONResponse:
        if registry.remove(worker_id):
            logger.info("Deregistered worker %s", worker_id)
            return JSONResponse(content={"status": "success"})
        return JSONResponse(status_code=404, content={"status": "not_found"})

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        raw = await request.body()
        llm_req = LLMRequest(request_id=uuid.uuid4().hex, body=_routing_body(_safe_json(raw)))

        endpoints = registry.endpoints()
        if not endpoints:
            return JSONResponse(status_code=503, content={"error": "no workers registered"})

        session: aiohttp.ClientSession = request.app.state.http

        # Metrics scrape + scheduling happen ON the request path under a lock
        # [check verl_hook.py:L90-95 for why they happen in the same task]
        async with scheduling_lock:
            await asyncio.gather(*[fetch_worker_metrics(ep, inflight, session) for ep in endpoints])
            selected = scheduler.run(llm_req, candidates=endpoints)
            if not selected:
                return JSONResponse(status_code=503, content={"error": "no worker selected"})
            winner = selected[0].endpoint
            inflight.increment(winner.name)

        worker_url = str(winner.attributes["url"])
        # Forward client headers to the worker, minus hop-by-hop ones aiohttp resets itself.
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
