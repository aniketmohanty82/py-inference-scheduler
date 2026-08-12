# Pinned upstream revisions

| repo | ref | why |
|---|---|---|
| rllm-org/rllm | `1d1109a655e291b3001d8526d7c9ecc5b9328226` (main, 2026-08-12) | v0.3.0.pre; pins verl==0.8.0 + vllm==0.22.1 (matches our Mooncake connectors); rllm-model-gateway is an editable path dep inside this monorepo |

`rllm-gateway-routing-policy.diff` applies onto that SHA (`git apply` in the image
build). It adds the `RLLM_GATEWAY_ROUTING_POLICY` env var to the gateway server's
env map and plumbs `rllm.gateway.routing_policy` through `GatewayManager` into both
subprocess and thread launch modes. Local mirror of the same commit exists as branch
`gateway-routing-policy` in the working clone; push to a GitHub fork is deferred
until explicitly approved, at which point the image recipe can switch from
clone+apply to a fork ref.

Gateway RoutingPolicy protocol frozen at this SHA
(`rllm-model-gateway/src/rllm_model_gateway/session_router.py`):

```python
class RoutingPolicy(Protocol):
    def select_worker(
        self,
        workers: list[WorkerInfo],
        session_id: str | None,
        active_counts: dict[str, int],
    ) -> WorkerInfo: ...
    def on_worker_change(self, workers: list[WorkerInfo]) -> None: ...
```

Notes fixed at this SHA:
- `_load_policy()` instantiates the class with NO arguments — our policy must be
  constructible from env vars alone.
- verl backend => gateway mode "process" (subprocess inherits trainer env), chosen in
  `rllm/trainer/unified_trainer.py`; the gateway path requires an `agent_flow` +
  (`evaluator` or `hooks`) invocation — a bare `workflow_class` run uses
  UnifiedWorkflowEngine, i.e. the token-in-token-out path that BYPASSES the gateway.
