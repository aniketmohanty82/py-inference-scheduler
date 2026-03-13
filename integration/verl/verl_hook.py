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


import asyncio
import logging
import ray
import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AsyncLLMServerManager,
    AgentLoopWorker,
    AgentLoopManager
)

from scheduling.types import LLMRequest, Endpoint
from scheduling.inflight_store import InflightStore
from scheduling.ray_request_scheduler import RayRequestScheduler

logger = logging.getLogger(__name__)

class InferenceSchedulerServerManager(AsyncLLMServerManager):
    """
    Subclass of verl's AsyncLLMServerManager that delegates routing
    to the native py-inference-scheduler engine.
    Compatible with verl v0.7.1.
    """
    def __init__(
        self, 
        config: DictConfig, 
        servers: List[Tuple[str, ray.actor.ActorHandle]], 
        load_balancer_handle: ray.actor.ActorHandle,
        *args, 
        **kwargs
    ):
        super().__init__(config, servers, load_balancer_handle, *args, **kwargs)
        self.ray_request_scheduler = RayRequestScheduler()
        self.inflight_store = InflightStore()
        self.endpoints = []
        self._lb_acquired_requests = set()

        # Reconstruct endpoints from the new (id, handle) tuple structure
        for server_id, handle in servers:
            ep = Endpoint(
                name=server_id,
                attributes={
                    "replica_obj": handle,
                    "routing_stats": {}
                }
            )
            self.endpoints.append(ep)

        self._metrics_task = None

    async def poll_worker_metrics_loop(self):
        """Asynchronously scrapes metrics to update endpoint queue sizes (50ms interval)"""
        import aiohttp

        self._worker_urls = None

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    if self._worker_urls is None:
                        discovered_urls = []
                        for ep in self.endpoints:
                            actor = ep.attributes["replica_obj"]
                            try:
                                ip, port = await actor.get_server_address.remote()
                                discovered_urls.append(f"http://{ip}:{port}/metrics")
                            except Exception as e:
                                logger.error(f"Failed to fetch IP for {ep.name}: {e}")
                                discovered_urls.append(None)
                        self._worker_urls = discovered_urls
                        logger.info(f"Natively mapped Worker Metrics IPs: {self._worker_urls}")

                    # Map exactly to self._worker_urls index to pair with endpoints
                    async def fetch_worker_metrics(session, idx, url):
                        if url is None:
                            return
                        try:
                            async with session.get(url, timeout=0.200) as response:
                                text = await response.text()
                                stats = {
                                    "num_waiting_reqs": 0,
                                    "num_running_reqs": 0,
                                    "kv": 0.0,
                                    "queue_len": 0
                                }

                                for line in text.split('\n'):
                                    try:
                                        if line.startswith("vllm:num_requests_waiting"):
                                            stats["num_waiting_reqs"] = int(float(line.split(" ")[-1]))
                                        elif line.startswith("vllm:num_requests_running"):
                                            stats["num_running_reqs"] = int(float(line.split(" ")[-1]))
                                        elif line.startswith("vllm:kv_cache_usage_perc"):
                                            stats["kv"] = float(line.split(" ")[-1])
                                    except IndexError:
                                        continue

                                endpoint = self.endpoints[idx]
                                local_inflight = self.inflight_store.get(endpoint.name)
                                stats["num_running_reqs"] = max(stats["num_running_reqs"], local_inflight)

                                stats["queue_len"] = stats["num_waiting_reqs"] + stats["num_running_reqs"]
                                endpoint.attributes["routing_stats"] = stats

                        except asyncio.TimeoutError:
                            logger.debug(f"Timeout connecting to {url}")
                        except Exception as e:
                            logger.error(f"Failed to scrape {url}: {e}")

                    tasks = [fetch_worker_metrics(session, i, url) for i, url in enumerate(self._worker_urls)]
                    await asyncio.gather(*tasks)

                except Exception as e:
                    logger.error(f"Metrics poll error: {e}")

                await asyncio.sleep(0.05)

    async def _acquire_server(self, request_id: str, prompt_ids: Optional[List[int]] = None) -> Tuple[str, ray.actor.ActorHandle]:
        """Overrides Verl's Native Global Load Balancer with py-inference-scheduler logic"""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(verl_metrics_polling_loop(self.endpoints, self.inflight_store))

        req = LLMRequest(request_id=request_id, body=prompt_ids)
        selected_endpoints = self.ray_request_scheduler.run(req, candidates=self.endpoints)

        # fall back to verl LB
        if not selected_endpoints:
            logger.warning("py-inference-scheduler returned no endpoints, falling back to verl global LB.")
            self._lb_acquired_requests.add(request_id)
            return await super()._acquire_server(request_id)

        winning_endpoint: Endpoint = selected_endpoints[0].endpoint
        stats = winning_endpoint.attributes.get('routing_stats', {})
        logger.info(f"[{request_id[:6]}] Routed to {winning_endpoint.name} (stats: {stats})")
        
        return winning_endpoint.name, winning_endpoint.attributes["replica_obj"]

    def _release_server(self, server_id: str, request_id: Optional[str] = None) -> None:
        """Decrements local inflight tracking and notifies global LB if it originated there"""
        self.inflight_store.decrement(server_id)
        if request_id and request_id in self._lb_acquired_requests:
            super()._release_server(server_id)
            self._lb_acquired_requests.remove(request_id)

    async def generate(
        self, 
        request_id: str, 
        *, 
        prompt_ids: List[int], 
        sampling_params: Dict[str, Any], 
        image_data: Optional[List[Any]] = None, 
        video_data: Optional[List[Any]] = None
    ):
        """Overrides Verl's generate to manage lifecycle with scheduler"""
        server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)
        self.inflight_store.increment(server_id)
        
        # vLLM requires a fresh request_id per generation to prevent KV cache collisions. verl has sticky request_ids for multi-turn rollouts.
        vllm_request_id = uuid.uuid4().hex
        try:
            return await server.generate.remote(
                request_id=vllm_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data
            )
        finally:
            self._release_server(server_id, request_id)


class PyInferenceAgentLoopWorker(AgentLoopWorker):
    """
    Overrides the Ray worker actor to inject the custom ServerManager
    before calling super().__init__
    Compatible with verl v0.7.1.
    """
    def __init__(
        self, 
        config: DictConfig, 
        servers: List[Tuple[str, ray.actor.ActorHandle]], 
        load_balancer_handle: ray.actor.ActorHandle,
        reward_loop_worker_handles: Optional[List[ray.actor.ActorHandle]] = None
    ):
        # Inject our manager
        self.server_manager = InferenceSchedulerServerManager(config, servers, load_balancer_handle)
        super().__init__(config, servers, load_balancer_handle, reward_loop_worker_handles)


class PyInferenceAgentLoopManager(AgentLoopManager):
    """
    The main hook entrypoint loaded by ray_trainer.py
    Overrides the worker actor class that verl spawns across the cluster.
    """
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(PyInferenceAgentLoopWorker)
        super().__init__(*args, **kwargs)
