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

        # vLLM serves each engine's /metrics from its own registry. Forcing one
        # PROMETHEUS_MULTIPROC_DIR on the engines of a node makes every port
        # report the node aggregate, so no environment is propagated here.
        vLLMHttpServer.get_routing_stats = get_vllm_routing_stats
