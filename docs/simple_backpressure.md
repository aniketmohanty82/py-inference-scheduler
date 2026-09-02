# Simple Backpressure Flow Control

`simple_backpressure` holds requests on the scheduler side while every replica is saturated, instead of pushing them into the model servers' waiting queues where they force KV-cache preemptions. It is metric-driven and model-server agnostic: the only inputs are the `kv` (KV-cache utilization) and `num_waiting_reqs` values that every integration already publishes into `routing_stats`, whether they come from a vLLM or SGLang Prometheus scrape or from an in-process stat logger.

## How it works

1. **Gate.** On every scheduling decision the plugin passes through only the replicas below both saturation thresholds (same criteria as the `saturation` filter). The configured routing policy (filters → scorers → picker) then runs over that admissible subset.
2. **Queue.** When *all* replicas are saturated, the plugin returns no candidates and the integration's `FlowControlManager` parks the request in a FIFO queue — outside any scheduling lock, so unaffected traffic keeps flowing.
3. **Watch.** While requests are parked, a background watcher polls fresh endpoint metrics every `poll_interval_s` (slime reuses its `MetricsPoller` snapshots; vime scrapes; Ray Serve queries replica actors). Request completions do **not** drive re-admission — only metrics do.
4. **Drain with AIMD.** Each watcher tick where the gate is open admits up to `window` waiters, then grows the window additively (`aimd_increase`). A tick that finds the gate shut again shrinks it multiplicatively (`aimd_decay`). The window seeds from the number of admissible replicas when a parked episode first reopens and resets once the queue drains. This makes the drain rate feedback-controlled: a fleet-wide capacity release (say KV dropping from 99% to 50%) ramps admissions quickly instead of trickling one request per tick, while a re-saturation halves the rate immediately.

Unlike `kv_saturation`, this plugin keeps no per-request budgets, learns nothing from responses, and never inspects request bodies. It is stateless: config hot-reload can swap it freely because all queue state lives in the manager.

## Configuration

Enable it per profile in `scheduler.yaml`:

```yaml
profile_handler:
  type: single_profile
profiles:
  backpressure:
    flow_control:
      type: simple_backpressure
      kv_threshold: 0.95       # saturated at >= 95% KV-cache utilization
      waiting_threshold: 6     # or >= 6 waiting requests
    scorers:
      - type: waiting_queue
        weight: 5.0
      - type: kv_cache
        weight: 1.0
    picker:
      type: max_score
```

### Plugin knobs (YAML)

| Knob | Default | Meaning |
| --- | --- | --- |
| `kv_threshold` | `0.95` | A replica is saturated at or above this KV-cache utilization (0–1]. |
| `waiting_threshold` | `6` | A replica is saturated at or above this many waiting requests. |

Both thresholds are inclusive; an endpoint with no `routing_stats` yet counts as healthy.

### Manager knobs (per integration)

| Knob | Default | Where | Meaning |
| --- | --- | --- | --- |
| `poll_interval_s` | `0.1` | `--flow-poll-interval-s` (slime, vime) / `FLOW_CONTROL_POLL_S` env (Ray Serve) | Watcher tick while requests are parked. |
| `aimd_increase` | `1` | `FlowControlManager` constructor | Additive window growth per open tick. |
| `aimd_decay` | `0.5` | `FlowControlManager` constructor | Multiplicative window shrink when the gate shuts (floor 1). |
| `max_admissions_per_tick` | `0` (uncapped) | `FlowControlManager` constructor | Hard cap on admissions per tick. |

## Integration support

| Integration | Queueing wired | Watcher metric source |
| --- | --- | --- |
| Ray Serve (`integration/rayserve`) | yes | `record_routing_stats` actor calls per tick |
| slime (`integration/slime`) | yes | `MetricsPoller` background snapshots (no extra I/O) |
| vime (`integration/vime`) | yes | watcher scrapes worker `/metrics` per tick |
| verl, tunix | follow-up | — |

## Caveats

* **Barging.** A fresh request that finds capacity is admitted immediately even while older requests are parked; strict FIFO across both paths is a follow-up.
* **Burst races.** The gate is stateless, so requests admitted within one metrics-refresh window don't see each other's load. The AIMD window bounds this for the parked queue; concurrent fresh arrivals are spread by the backpressure scorers.
* **No queue timeout.** A parked request waits until capacity returns, the client disconnects, or the server shuts down (parked requests then get 503s). A `queue_timeout_s` knob is a follow-up.
