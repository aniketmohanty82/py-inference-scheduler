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

from scheduling.framework import LLMRequest, Endpoint
from datalayer.verl.metrics import verl_metrics_polling_loop
from datalayer.verl.datastore import InflightStore
from scheduling import Scheduler

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
        self.ray_request_scheduler = Scheduler()
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

    async def _acquire_server(self, request_id: str, prompt_ids: Optional[List[int]] = None) -> Tuple[str, ray.actor.ActorHandle]:
        """Overrides Verl's Native Global Load Balancer with py-inference-scheduler logic"""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(verl_metrics_polling_loop(self.endpoints, self.inflight_store))

        for ep in self.endpoints:
            ep.attributes["queue_len"] = self.inflight_store.get(ep.name)

        req = LLMRequest(request_id=request_id, body=prompt_ids)
        selected_endpoints = self.ray_request_scheduler.run(req, candidates=self.endpoints)

        # fall back to verl LB
        if not selected_endpoints:
            logger.warning("py-inference-scheduler returned no endpoints, falling back to verl global LB.")
            self._lb_acquired_requests.add(request_id)
            return await super()._acquire_server(request_id)

        winning_endpoint: Endpoint = selected_endpoints[0].endpoint
        logger.info(f"[{request_id[:6]}] Routed to {winning_endpoint.name}")
        
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
        # Yield CPU to check for metrics poller
        await asyncio.sleep(0)

        server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)
        self.inflight_store.increment(server_id)
        
        # vLLM requires a fresh request_id per generation to prevent KV cache collisions. verl has sticky request_ids for multi-turn rollouts.
        vllm_request_id = uuid.uuid4().hex

        # vLLMAsyncServer ignores ignore_eos from config, so we must pass it explicitly.
        if isinstance(sampling_params, dict):
            sampling_params["ignore_eos"] = True
        elif hasattr(sampling_params, "ignore_eos"):
             setattr(sampling_params, "ignore_eos", True)

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