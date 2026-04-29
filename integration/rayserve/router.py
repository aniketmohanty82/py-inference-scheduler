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
import random
import struct
import time
from typing import Any, Callable
import uuid

from ray import serve
from ray.actor import ActorHandle
from ray.llm._internal.serve.core.ingress.builder import (  # noqa: PLC2701
    LLMServingArgs,
    make_fastapi_ingress,
)
from ray.llm._internal.serve.core.server.builder import build_llm_deployment  # noqa: PLC2701
from ray.serve._private.common import (
    DeploymentHandleSource,
    DeploymentID,
    RunningReplicaInfo,
)
from ray.serve.llm import LLMConfig
from ray.serve.request_router import (
    PendingRequest,
    ReplicaID,
    ReplicaResult,
    RequestRouter,
    RunningReplica,
)

from datalayer.rayserve.engine import MetricsAwareLLMServer
from scheduling.core.scheduler import Scheduler
from scheduling.framework import (
    Endpoint,
    LLMRequest,
)

NAMESPACE_FOR_UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "llmd.inference.scheduler")
_DEFAULT_USE_TOKEN_BUDGET = False


class IGWRouter(RequestRouter):
    def __init__(  # noqa: PLR0913
        self,
        deployment_id: DeploymentID,
        handle_source: DeploymentHandleSource,
        *,
        self_actor_id: str | None = None,
        self_actor_handle: ActorHandle | None = None,
        use_replica_queue_len_cache: bool = False,
        get_curr_time_s: Callable[[], float] | None = None,
        create_replica_wrapper_func: Callable[[RunningReplicaInfo], RunningReplica] | None = None,
        **kwargs: object,
    ) -> None:
        RequestRouter.__init__(
            self,
            deployment_id=deployment_id,
            handle_source=handle_source,
            self_actor_id=self_actor_id,
            self_actor_handle=self_actor_handle,
            use_replica_queue_len_cache=True,
            get_curr_time_s=get_curr_time_s,
            create_replica_wrapper_func=create_replica_wrapper_func,
            **kwargs,
        )
        self.scheduler = Scheduler()
        self._rollout_request_stats: dict[str, dict[str, int]] = {}  # rollout_request_id -> {isl, max_osl}
        self._replica_token_usage: dict[ReplicaID, int] = {}  # replica_id -> token_usage_at_replica
        self._request_at_replica: dict[str, tuple[ReplicaID, int]] = {}  # request_id -> (replica_id, tokens)
        self._admission_queue: list[asyncio.Future] = []  # FIFO queue for blocked requests

        self._last_drip_at = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._budgeted_requests: set[str] = set()
        self._replica_kv_cache_size: dict[ReplicaID, int] = {}

    async def _get_routing_stats(
        self, replicas: list[RunningReplica], pending_request: PendingRequest
    ) -> list[dict]:
        """Fetch routing stats (KV usage, etc.) from a list of replicas."""
        futures = [
            r._get_replica_wrapper(pending_request)._actor_handle.record_routing_stats.remote()
            for r in replicas
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        return [res if not isinstance(res, Exception) else {} for res in results]

    def _parse_to_llm_request(self, pending_request: PendingRequest) -> LLMRequest:
        """Converts Ray request to LLMRequest."""
        if pending_request:
            req_id = pending_request.metadata.request_id
        else:
            req_id = str(uuid.uuid4())

        if not pending_request or not pending_request.args:
            return LLMRequest(request_id=req_id, body="", target_model="qwen-32b")
        request_args = pending_request.args[0]

        if hasattr(request_args, "messages"):
            body = request_args.messages
        elif hasattr(request_args, "prompt"):
            body = request_args.prompt
        else:
            body = request_args

        # make configurable in LLMConfig
        target_model = getattr(request_args, "model", "qwen-32b")

        return LLMRequest(request_id=req_id, body=body, target_model=target_model)

    def _get_rollout_request_id(self, body: Any) -> tuple[str, int]:
        """Generates a rollout request ID and approximate character length from the request body.
        We don't require character length to be accurate, it is used as a fail-safe and for edge cases.
        rollout_request_id -> id used to track ISL/OSL of a single prompt across rollouts/steps. Calculated using promptIDs
        request_id -> id used to uniquely identify a request (vLLM encourages this). 
        """
        if not body:
            return "", 0

        try:
            # Case 1: Pre-tokenized prompt ids (List[int])
            if isinstance(body, list) and len(body) > 0 and isinstance(body[0], int):
                prompt_bytes = struct.pack(f"{len(body)}i", *body)
                return uuid.uuid5(NAMESPACE_FOR_UUID, prompt_bytes.hex()).hex, len(body) * 4

            # Case 2: Raw string or bytes prompt
            if isinstance(body, (str, bytes)):
                prompt_str = body if isinstance(body, str) else body.hex()
                return uuid.uuid5(NAMESPACE_FOR_UUID, prompt_str).hex, len(body)

            # Case 3: List of messages (Dicts/Pydantic models)
            if isinstance(body, list):
                extracted_text = ""
                for m in body:
                    if isinstance(m, dict):
                        extracted_text += str(m.get("content", ""))
                    else:
                        extracted_text += str(getattr(m, "content", ""))
                char_len = len(extracted_text)
                return uuid.uuid5(NAMESPACE_FOR_UUID, extracted_text).hex, char_len

        except Exception as e:
            print(f"[ROUTER ERROR] Failed to parse request ID securely: {e}")

        return "", 0

    async def _wait_for_space(self) -> None:
        """Wait until at least one replica has space freed up."""
        fut = self._loop.create_future()
        self._admission_queue.append(fut)
        await fut

    def _get_available_replicas(
        self,
        candidate_replicas: list[RunningReplica],
        tokens_required: int,
    ) -> list[RunningReplica]:
        """Return subset of replicas that can fit the required tokens."""
        res = []
        for r in candidate_replicas:
            budget = self._replica_kv_cache_size.get(r.replica_id)
            if budget is not None and budget > 0:
                if (
                    self._replica_token_usage.get(r.replica_id, 0) + tokens_required
                    <= budget
                ):
                    res.append(r)
        return res

    async def choose_replicas(  # noqa: PLR0912
        self,
        candidate_replicas: list[RunningReplica],
        pending_request: PendingRequest | None = None,
    ) -> list[list[RunningReplica]]:
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        if not candidate_replicas:
            return []

        llm_req = self._parse_to_llm_request(pending_request)
        rollout_request_id, char_len = self._get_rollout_request_id(llm_req.body)
        request_id = llm_req.request_id

        # for health check empty requests
        if char_len == 0:
            print("No pending request or args, defaulting to random choice")
            index = random.randint(0, len(candidate_replicas) - 1)
            return [[candidate_replicas[index]]]

        try:
            metrics_results = await self._get_routing_stats(candidate_replicas, pending_request)

            candidates = []
            for replica, routing_stats in zip(candidate_replicas, metrics_results):
                queue_len = 0
                if self.replica_queue_len_cache:
                    cached_val = self.replica_queue_len_cache.get(replica.replica_id)
                    if cached_val is not None:
                        queue_len = cached_val

                # Discover and cache KV Cache size if not already known
                if replica.replica_id not in self._replica_kv_cache_size:
                    kv_cache_size = routing_stats.get("kv_cache_size", -1)
                    if kv_cache_size > 0:
                        self._replica_kv_cache_size[replica.replica_id] = kv_cache_size
                        print(
                            f"[BUDGET] Discovered KV Cache size for replica {replica.replica_id}: {kv_cache_size} tokens"
                        )

                if isinstance(routing_stats, Exception):
                    print(f"Failed to fetch metrics via RPC for {replica_id}: {routing_stats}")
                    routing_stats = {}  # noqa: PLW2901

                candidates.append(
                    Endpoint(
                        name=str(replica.replica_id),
                        attributes={
                            "queue_len": queue_len,
                            "routing_stats": routing_stats,
                        },
                    )
                )

            fc = self.scheduler.get_flow_control_config()
            use_token_budget = fc.get("use_token_budget", _DEFAULT_USE_TOKEN_BUDGET)

            tokens_required = 0
            is_budgeted = False

            if (
                use_token_budget
                and request_id
                and request_id not in self._budgeted_requests
            ):
                default_osl = fc.get("default_osl", 1024)
                drip_threshold_kv = fc.get("drip_threshold_kv", 0.1)
                drip_interval_s = fc.get("drip_interval_s", 2.0)
                stats = self._rollout_request_stats.get(rollout_request_id)

                if stats:
                    tokens_required = stats["isl"] + stats["osl"]
                    is_budgeted = True
                else:
                    # Fallback for first contact with prompt
                    tokens_required = (char_len // 4) + default_osl
                    is_budgeted = True

                if is_budgeted:
                    while True:
                        available_replicas = self._get_available_replicas(
                            candidate_replicas, tokens_required
                        )

                        if available_replicas:
                            candidate_replicas = available_replicas
                            candidates = [
                                c
                                for c in candidates
                                if c.name in [str(r.replica_id) for r in candidate_replicas]
                            ]
                            break
                        else:
                            new_metrics = await self._get_routing_stats(
                                candidate_replicas, pending_request
                            )
                            for i, res in enumerate(new_metrics):
                                candidates[i].attributes["routing_stats"] = res

                            now = time.time()
                            best_drip_replica = None
                            for i, r in enumerate(candidate_replicas):
                                stats = candidates[i].attributes.get("routing_stats", {})
                                physical_kv = stats.get("kv", 1.0)

                                if (
                                    physical_kv < drip_threshold_kv
                                    and (now - self._last_drip_at) > drip_interval_s
                                ):
                                    best_drip_replica = r
                                    break

                            if best_drip_replica:
                                self._last_drip_at = now
                                print(
                                    f"[BUDGET] Drip Admission: req={request_id} to {best_drip_replica.replica_id} (Physical KV: {physical_kv:.2f})"
                                )
                                candidate_replicas = [best_drip_replica]
                                candidates = [
                                    c
                                    for c in candidates
                                    if c.name == str(best_drip_replica.replica_id)
                                ]
                                is_budgeted = True
                                break

                        await self._wait_for_space()

            selected_endpoints = self.scheduler.run(llm_req, candidates)

            index = -1
            for i, replica in enumerate(candidate_replicas):
                if (
                    len(selected_endpoints) > 0
                    and str(replica.replica_id) == selected_endpoints[0].endpoint.name
                ):
                    index = i
                    break
            if index == -1:
                index = random.randint(0, len(candidate_replicas) - 1)

            target_replica = candidate_replicas[index]

            # Commit the reservation for the selected replica
            if is_budgeted:
                self._replica_token_usage[target_replica.replica_id] = (
                    self._replica_token_usage.get(target_replica.replica_id, 0) + tokens_required
                )
                self._request_at_replica[request_id] = (
                    target_replica.replica_id,
                    tokens_required,
                )
                self._budgeted_requests.add(request_id)
                print(
                    f"[BUDGET] Admission committed for req={request_id} to replica={target_replica.replica_id} (Usage: {self._replica_token_usage[target_replica.replica_id]})"
                )

            return [[target_replica]]
        except Exception as e:
            print(f"Error of: {e!r} during scheduling: {e}, defaulting to random choice")
            index = random.randint(0, len(candidate_replicas) - 1)
            return [[candidate_replicas[index]]]

    def on_request_routed(
        self,
        pending_request: PendingRequest,
        replica_id: ReplicaID,
        result: ReplicaResult,
    ):
        llm_req = self._parse_to_llm_request(pending_request)
        request_id = llm_req.request_id
        rollout_request_id, char_len = self._get_rollout_request_id(llm_req.body)
        routed_at = time.time()
        is_streaming = pending_request.metadata.is_streaming

        def _on_done(_):
            elapsed = time.time() - routed_at
            if request_id in self._request_at_replica:
                replica_id_to_reclaim, tokens = self._request_at_replica.pop(request_id)
                self._replica_token_usage[replica_id_to_reclaim] = max(
                    0, self._replica_token_usage.get(replica_id_to_reclaim, 0) - tokens
                )
                self._budgeted_requests.discard(request_id)
                print(
                    f"[BUDGET] Released req={request_id} from replica={replica_id_to_reclaim}. "
                    f"Elapsed={elapsed:.3f}s. Usage={self._replica_token_usage[replica_id_to_reclaim]}"
                )

                if self._admission_queue:
                    waiter = self._admission_queue.pop(0)
                    if not waiter.done():
                        waiter.set_result(True)

        result.add_done_callback(_on_done)

        if char_len > 0:
            if not is_streaming:
                original_get_async = result.get_async

                async def patched_get_async():
                    response = await original_get_async()
                    if response and hasattr(response, "usage") and response.usage:
                        self._rollout_request_stats[rollout_request_id] = {
                            "isl": response.usage.prompt_tokens,
                            "osl": response.usage.completion_tokens,
                        }
                    return response

                result.get_async = patched_get_async
            else:
                original_anext = result.__anext__

                async def patched_anext():
                    chunk = await original_anext()
                    if chunk and hasattr(chunk, "usage") and chunk.usage:
                        self._rollout_request_stats[rollout_request_id] = {
                            "isl": chunk.usage.prompt_tokens,
                            "osl": chunk.usage.completion_tokens,
                        }
                    return chunk

                result.__anext__ = patched_anext


# Hooking into Ray Serve's Request Router

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-32b",
        model_source="Qwen/Qwen2.5-32B-Instruct",
    ),
    engine_kwargs=dict(
        enable_prefix_caching=True,
        tensor_parallel_size=2,
    ),
    deployment_config=dict(
        # max_ongoing_requests=10,
        autoscaling_config=dict(
            min_replicas=1,
            max_replicas=1,
        ),
        request_router_config=dict(
            # Note our custom IGWRouter here
            request_router_class=IGWRouter,
        ),
        ray_actor_options=dict(num_cpus=1),
    ),
    runtime_env={
        "env_vars": {
            "NCCL_NET_PLUGIN": "/usr/local/gib/lib64/libnccl-net_internal.so",
            "NCCL_CROSS_NIC": "0",
            "NCCL_NET_GDR_LEVEL": "PIX",
            "NCCL_P2P_NET_CHUNKSIZE": "131072",
            "NCCL_NVLS_CHUNKSIZE": "524288",
            "NCCL_IB_ADAPTIVE_ROUTING": "1",
            "NCCL_IB_QPS_PER_CONNECTION": "4",
            "NCCL_IB_TC": "52",
            "NCCL_IB_FIFO_TC": "84",
            "NCCL_TUNER_CONFIG_PATH": "/usr/local/gib/configs/tuner_config_a3u.txtpb",
        }
    },
)


def build_custom_openai_app(builder_config: dict[str, object]) -> object:
    # Same internal logic as build_openai_app, but we map our deployment_cls
    builder_config = LLMServingArgs.model_validate(builder_config)
    llm_configs = builder_config.llm_configs

    llm_deployments = [
        build_llm_deployment(c, deployment_cls=MetricsAwareLLMServer) for c in llm_configs
    ]

    ingress_cls_config = builder_config.ingress_cls_config
    ingress_options = ingress_cls_config.ingress_cls.get_deployment_options(llm_configs)
    ingress_cls = make_fastapi_ingress(ingress_cls_config.ingress_cls)

    return serve.deployment(
        ingress_cls,
        **ingress_options,
    ).bind(llm_deployments=llm_deployments, **ingress_cls_config.ingress_extra_kwargs)


app = build_custom_openai_app(
    {
        "llm_configs": [llm_config],
    }
)

if __name__ == "__main__":
    serve.run(app, blocking=True)
