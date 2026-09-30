from __future__ import annotations

import logging

from py_inference_scheduler.datalayer.metrics.verl.vllm import get_vllm_routing_stats

logger = logging.getLogger(__name__)


class VllmEnginePatch:
    """Expose vLLM routing stats on verl's server actor over Ray RPC."""

    @classmethod
    def apply(cls) -> None:
        try:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import (  # type: ignore[import-not-found]
                vLLMHttpServer,
            )
        except Exception as e:  # noqa: BLE001 - vLLM internals raise more than
            # ImportError on CPU-only nodes (e.g. triton AttributeError on the
            # Ray head); any import failure means "no vLLM here", so skip.
            logger.info(
                "Skipping vLLM patch (normal on head node if vLLM is not importable): %s", e
            )
            return

        # Only the RPC surface. The server actor runs vLLM's front-end stat
        # logger and its uvicorn in one process, so vLLM's default per-process
        # registry already serves per-engine /metrics. Earlier revisions forced
        # PROMETHEUS_MULTIPROC_DIR here: one directory shared by every engine on
        # the node made each /metrics an aggregate of all of them (gauges from
        # the last writer, counters summed), and setting it after
        # prometheus_client was imported served an empty registry.
        vLLMHttpServer.get_routing_stats = get_vllm_routing_stats
