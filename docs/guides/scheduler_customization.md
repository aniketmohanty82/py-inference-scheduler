# Scheduler Customization Guide

This guide explains how the `py-inference-scheduler` engine works and how you can customize its behavior by designing custom **Scheduler Profiles** in your `scheduler.yaml`.

---

## 1. Scheduling Pipeline Architecture

The scheduler uses a modular pipeline to make routing decisions for each incoming request. A **Scheduler Profile** defines the specific plugins used at each stage of this pipeline:

```mermaid
graph TD
    Request([Incoming Request]) --> FlowControl[1. Flow Control]
    FlowControl --> Filter[2. Filters]
    Filter --> Scorer[3. Scorers]
    Scorer --> Picker[4. Picker]
    Picker --> Route([Selected Replica])
```
*   **Flow Control**: Blocks or throttles routing to a replica that doesn't meet flow control policy.
*   **Filters**: Eliminate replicas that do not meet hard constraints (e.g., matching model names, healthy status).
*   **Scorers**: Assign a numerical score to each remaining replica. Multiple scorers can be combined with weights. The scheduler normalizes scores to `[0, 1]` before applying weights.
*   **Picker**: Selects a single replica from the scored candidates. By default, it picks the highest-scoring replica.

---

## 2. YAML Configuration Schema

The scheduler is configured via a `scheduler.yaml` file. 

*   **Reference Example**: For a production-ready reference, see [scheduler.yaml](../integration/verl/examples/scheduler.yaml).

### Schema Template

```yaml
profile_handler:
  type: <profile_handler_type_name>
  # (Optional config parameters for the handler)

profiles:
  <profile_name>:
    # (Optional) Filters run sequentially to eliminate replicas.
    # Omit if you do not need hard filtering.
    filters:
      - type: <filter_type_name>
        # (Filter-specific parameters)
        
    # Scorers assign normalized, weighted scores to remaining replicas.
    # While technically optional, a profile should typically have at least one scorer.
    scorers:
      - type: <scorer_type_name>
        weight: <float>
        # (Scorer-specific parameters)
        
    # (Optional) Custom picker to choose the final replica from scored candidates.
    # If omitted, the scheduler defaults to selecting the highest-scoring replica.
    picker:
      type: <picker_type_name>
      # (Picker-specific parameters)
      
    # (Optional) One flow-control plugin to gate admission before routing.
    # Omit if you do not want flow-control/preemption gating.
    flow_control:
      type: <flow_control_type_name>
      # (Flow-control-specific parameters)
```

---

## 3. Built-in Plugin Reference

### Profile Handlers (for P/D Disaggregation)
Profile Handlers determine which profile(s) to run for a given request.
*   **`single_profile`**: The default handler. It simply runs all defined profiles and returns the first one that successfully selects an endpoint.

### Filters
Filters eliminate replicas based on hard rules.
*   **`simple`**: Keeps only replicas that have a specific attribute matching a target value.
    *   `key` (string, required): The attribute key to check (e.g., `"model_name"`).
    *   `value` (object, optional): The value to match. If omitted, it acts as a no-op.

### Scorers
Scorers assign scores to replicas. Multiple scorers are normalized and weighted.

#### A. Backpressure-Based (Discourages routing to overloaded replicas)
*   **`least_queue`**: Scores replicas based on their active Ray Serve queue length. Encourages routing to replicas with the fewest active requests.
*   **`waiting_queue`**: Scores replicas based on the number of *waiting* requests in the vLLM engine.
*   **`running_queue`**: Scores replicas based on the number of *running* requests in the vLLM engine.
*   **`kv_cache`**: Scores replicas based on physical KV cache memory utilization. Encourages routing to replicas with more free KV cache.
*   **`queue_length`**: A generic scorer that reads a custom attribute.
    *   `attribute_key` (string, default: `"waiting_queue_size"`): The attribute to read.

#### B. Cache-Based (Encourages routing to maximize cache hits)
*   **`prefix_cache`**: Scores replicas based on how much of the request's prompt matches the prefix cache already loaded on the replica. Crucial for maximizing vLLM/SGLang chunked prefix cache hits.
    *   `block_size` (int, default: `64`): Token block size for hashing.
    *   `max_prefix_blocks` (int, default: `256`): Max blocks to index.
    *   `lru_capacity_per_server` (int, default: `31250`): Cache capacity per replica.
    *   `min_match_ratio` (float, default: `0.0`): Minimum fraction of prompt blocks the best replica must have cached for prefix scores to be used; below it, the request is treated as novel and routed to the least-loaded replicas. The default of `0` disables the threshold, so the least-loaded fallback only fires when no replica has any matching block.

*   **`request_affinity`**: Scores the replica that served the previous request with the same `request_id` at `1.0` and every other replica at `0.0`; a `request_id` with no history scores every replica `0.0`, which the profile treats as a tie. Built for multi-turn RL rollouts, where every turn resubmits the whole context and the KV for it lives on the engine that ran the last turn. Unlike a sticky pin, a filter can still drop the holder, and the next turn then follows the replica that actually served this one. Give it a weight above the sum of the load scorers so the holder wins whenever it is a candidate.
    *   `capacity` (int, default: `20000`): Number of request ids remembered; the least recently routed is evicted first.

#### C. Generic Scorers (for benchmarking against current RL sampling routing)
*   **`round_robin`**: Cycles through replicas sequentially.
*   **`jitter`**: Scores every replica with a uniform random value in `[0, 1)`. At a small weight it only decides exact ties, which otherwise always go to the first replica in candidate order.
*   **`constant`**: Assigns a static score to all replicas.
    *   `value` (float, required): The score to assign.

### Pickers
Pickers choose the final replica from the scored list.
*   **`max_score`**: Always selects the replica with the highest combined score (default).
*   **`random`**: Introduces entropy by picking randomly from the top $N$ replicas.
    *   `max_num` (int, default: `1`): The size of the top-candidate pool to pick from.

### Flow Control (Gatekeeping)
Flow control plugins prevent replica overload and mid-decoding preemptions by controlling the flow to affected replicas.
*   **`simple_backpressure`**: Admits a request only to replicas below both saturation thresholds, read from live engine metrics. When every replica is over, the integration's flow-control manager parks the request and re-admits it, one at a time at an AIMD-paced rate, as fresh metrics show capacity. Stateless: completions do not drive re-admission, metrics do. Validated on a 32B agentic RL rollout at `kv_threshold: 0.90`, `waiting_threshold: 4` (preemptions -34 to -40% per million tokens, engine queue wait -74%).
    *   `kv_threshold` (float, default: `0.95`): KV-cache utilization at or above which a replica is inadmissible.
    *   `waiting_threshold` (int, default: `6`): waiting-request count at or above which a replica is inadmissible.
*   **`kv_saturation`**: Estimates the KV cache impact of incoming requests. If routing a request to a replica would exceed its physical KV cache capacity (causing vLLM to preempt/drop other active requests), it blocks admission.
    *   `enable_drip` (bool, default: `false`): Enables slow "drip" admission when all replicas are saturated, rather than blocking completely.
    *   `drip_threshold_kv` (float, default: `0.1`): Max physical KV utilization for drip eligibility.
    *   `drip_interval_s` (float, default: `2.0`): Minimum time between drip admissions.
    *   `default_osl` (int, default: `1024`): Default output sequence length estimate used before stats are learned.
    *   *More Info*: For a detailed deep-dive into how KV saturation budgeting works and its mathematical model, see the [KV Saturation Guide](./kv_saturation.md).

---

## 4. Example Configuration

Here is an example of a sophisticated `scheduler.yaml` that combines prefix caching, backpressure scoring, and KV saturation protection:

```yaml
profile_handler:
  type: single_profile

profiles:
  hybrid_policy:
    scorers:
      # Heavily favor prefix cache hits for speed
      - type: prefix_cache
        weight: 10.0
        block_size: 64
      # Secondarily favor replicas with lower KV cache utilization
      - type: kv_cache
        weight: 3.0
      # Trivial backpressure check
      - type: least_queue
        weight: 1.0
    picker:
      type: max_score
    flow_controls:
      # Protect against KV saturation and preemption storms
      - type: kv_saturation
        enable_drip: true
        drip_threshold_kv: 0.15
        default_osl: 512
```
