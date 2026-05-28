---
name: verl-integration
description: >-
  Troubleshoot and fix issues encountered while following the veRL README to integrate py-inference-scheduler with the veRL framework. Integration should begin by reading the veRL README. Use when pods fail to start, metrics are missing or zero, or job submission fails. Focuses on GRPO training.
---

# veRL Scheduler Integration

Integration should begin by reading the [veRL README](integration/verl/README.md). This file deals with specific issues and debugging steps you may be led to while following the README.

## Supported Environment

- **Compatible Training Type**: This integration is designed for standard GRPO using `AgentLoopManager`. It does not work for training types that do not use `AgentLoopManager`.
- **Verified Images**: 
  - vLLM Backend: `verlai/verl:vllm011.latest`
  - SGLang Backend: `verlai/verl:sgl059.latest`
- **Version Constraints**: Stick to `verl==0.7.1`. If you must use different vLLM/SGLang version, use ones released after the ones specified above.

## Troubleshooting Pod Startup Failures

If pods are stuck in `PENDING`, `ContainerCreating`, `ImagePullBackOff` or are terminating:

- **Check for Missing ConfigMap**: Ensure you created the `scheduler-config` ConfigMap *before* applying the Ray cluster YAML.
- **Check for Stale Secrets**: Verify that your `imagePullSecrets` or HuggingFace secrets are not stale or expired. Refresh them or use a rotating key.
- **Inspect Pod Events**: Run `kubectl describe pod {pod_name}` to see why it is failing to start. If events do not show any issues, check the raw logs per container using `kubectl logs {pod_name} -c {container_name}`.

## Troubleshooting Job Submission Failures

If the job fails instantly or post engine initialization:

- **Check Environment Variables**: Verify you have the correct variables in [runtime-env.yaml](integration/verl/examples/runtime-env.yaml) (e.g., `ROUTER_CONFIG_PATH`, `PROMETHEUS_MULTIPROC_DIR`).
- **Avoid Context Waste**: Do not read full Ray job submission logs. Ask the user for the traceback or pipe logs to a `.txt` file to search.
- **Process Boundary Issues**: The monkey patch might fail to apply across Ray process boundaries. In isolated Ray environments, different Ray workers might handle the engine patch and the engine initialization, leaving the actor unpatched. It is best to use the official veRL vLLM and SGLang images as mentioned here and in the [veRL README](integration/verl/README.md). At the least, use vLLM >= 0.11.0 and SGLang >= 0.5.9
- **Trace Failures**: 
  - Errors related to `AgentLoopManager` or `AsyncLLMServerManager` usually indicate bugs in the package code.
  - Earlier trace errors usually point to image and dependency conflicts.
- **Check Ray Dashboard**: If you find no logs that show errors in the terminal or worker files, request the user to view the logs on the Ray Dashboard (usually hosted at `http://localhost:8265`). It provides a visual interface to check for errors or potential problems in the job or actors.

## Troubleshooting Metrics (routing_stats)

If metrics (specifically the `routing_stats` object) are missing or stuck at 0, follow these steps to diagnose and fix:

1.  **Query the Endpoint**: Run `curl http://{host}:{port}/metrics` from the pod to see if metrics are available at all. The actual endpoints to check will be mentioned in the logs as training starts as part of the "Selected endpoint..." and "Scorer...." logs.
    - If not available, it is likely that the engines in the images are not applying these functions (`patched_get_env_vars()` or `patched_launch()`) or applying them at the right time, rather than an issue with the functions themselves.
2.  **Check Aggregation Directory**: Verify that `PROMETHEUS_MULTIPROC_DIR: "/tmp/metrics"` is in `runtime-env.yaml` and mounted as an `emptyDir` volume in the cluster YAML.
3.  **Check Worker Logs**: If the root cause is still unclear, inspect the raw logs on head and worker pods at `/tmp/ray/session_latest/logs/`.