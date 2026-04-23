# KV Saturation Flow Control

The KV Saturation system is a flow control mechanism designed to maximize sampling throughput in memory-intensive sampling workloads by preventing KV cache preemptions.

Currently, this feature is only available in the **Ray Serve** implementation.

## Core Concepts

### 1. Token Budgeting
We have found that one of the causes of throughput collapse in vLLM is the "Recompute Tax" incurred when a GPU runs out of KV cache and must preempt (evict) a running request. Re-admitting that request requires recomputation of the preempted tokens, which creates bubbles in the GPU decode batch.

The KV Saturation system reduces this by ensuring a request is **only** admitted to a replica if that replica has enough theoretical capacity to serve the request from start to finish.

*   **Learning Phase:** During the first encounter with a prompt, the router records the actual Input Sequence Length (ISL) and Output Sequence Length (OSL).
*   **Virtual Reservation:** For all subsequent requests with the same fingerprint, the router calculates the full memory footprint of the request (ISL + OSL). It maintains a virtual counter of tokens occupied on each replica and will park new requests in an asynchronous queue if the addition of the new request would exceed the hardware's KV token budget.

### 2. The Drip Mechanism
In batch-heavy workloads, requests often finish in "cliffs"—where utilization drops significantly because the router waits for a full completion before admitting the next batch. This creates significant GPU idle time.

The **Drip** functionality acts as a controlled "pressure valve" to smooth out these utilization cliffs. This is used only when the virtual KV Cache for a replica is already exhausted.If a replica's physical KV cache usage falls below a specific threshold (decided by the user for now), the router will "drip" one extra request into that replica, even if the virtual budget is technically full. This ensures that as one batch finishes, there are still requests partially through their prefill/decode phase, raising the floor of average utilization.

## Configuration

The system is configured via the `flow_control` block in `scheduler_config.yaml`.

### Example Configuration

```yaml
profiles:
  default_profile:
    flow_control:
      use_token_budget: true      # Toggle to use KV saturation flow control
      default_osl: 1024           # OSL estimate for the very first contact (profiling)
      drip_threshold_kv: 0.2      # Drip if physical KV usage < 20%
      drip_interval_s: 2.0        # Limit drips to one every 2 seconds per replica
```

#### Strict Budgeting Mode (No Drip)
To enforce a mathematically strict budget where preemptions are theoretically impossible, you should disable the drip mechanism by setting the threshold to zero and the interval to a maximum value:

```yaml
    flow_control:
      use_token_budget: true
      drip_threshold_kv: 0.0
      drip_interval_s: 99999999.0
```

