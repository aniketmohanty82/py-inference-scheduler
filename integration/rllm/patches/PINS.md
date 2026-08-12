# Pinned upstream revisions

| repo | ref | why |
|---|---|---|
| rllm-org/rllm | `1d1109a655e291b3001d8526d7c9ecc5b9328226` (main, 2026-08-12) | v0.3.0.pre; pins verl==0.8.0 + vllm==0.22.1 (matches our Mooncake connectors); rllm-model-gateway is an editable path dep inside this monorepo |

`rllm-gateway-routing-policy.diff` applies onto that SHA (`git apply` in the image
build). It carries four changes:
1. `RLLM_GATEWAY_ROUTING_POLICY` env var in the gateway server's env map +
   `rllm.gateway.routing_policy` plumbed through `GatewayManager` (both modes).
2. Loopback-pin skip when `DOCKER_HOST` is set or
   `RLLM_GATEWAY_NO_LOOPBACK_PIN=1` (off-host sandboxes need a routable IP).
3. `kubernetes` sandbox backend (`rllm/sandbox/backends/kubernetes_backend.py`,
   Modal-shaped, exec-based; env knobs `RLLM_K8S_NAMESPACE`,
   `RLLM_K8S_NODE_SELECTOR`, `RLLM_K8S_IMAGE_PULL_SECRET`,
   `RLLM_K8S_POD_TIMEOUT`) + dispatch/resource/CLI/no-snapshot/no-tunnel
   integration edits. VALIDATED live on rls-ab-west 2026-08-12 (exec,
   runuser split, chunked uploads, timeout, teardown; requires
   `k8s/rbac-sandbox.yaml` in-cluster).
4. `kubernetes` python client added to the image install. Local mirror of the same commit exists as branch
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

Gate-1 validation (2026-08-12, local): patched gateway loaded a custom policy from
`RLLM_GATEWAY_ROUTING_POLICY`, delivered the URL-path session id to
`select_worker`, and honored its choice; full-stack resolution
(`rllm[verl]` + py-inference-scheduler) succeeds against PyPI **only with
`--override integration/rllm/image/overrides.txt`** (replicates rllm's
project-level uv overrides; without them verl's `numpy<2` vs vllm's
`numpy>=2`-via-opencv is unsatisfiable).

Notes fixed at this SHA:
- `_load_policy()` instantiates the class with NO arguments — our policy must be
  constructible from env vars alone.
- verl backend => gateway mode "process" (subprocess inherits trainer env), chosen in
  `rllm/trainer/unified_trainer.py`; the gateway path requires an `agent_flow` +
  (`evaluator` or `hooks`) invocation — a bare `workflow_class` run uses
  UnifiedWorkflowEngine, i.e. the token-in-token-out path that BYPASSES the gateway.
