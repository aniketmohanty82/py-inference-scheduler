# KV Saturation Flow Control

`kv_saturation` admits a request only to a replica whose KV cache has room for it, so engines are never asked to hold more context than fits and preempt running requests to make space.

## How it works

1. **Budget.** Each replica's budget is its KV cache capacity in tokens minus what admitted, unfinished requests have reserved. Capacity comes from the engine: vLLM publishes it in its `vllm:cache_config_info` metric.
2. **Request size.** A request needs its prompt tokens plus its trajectory's output from the previous turn. A trajectory's first turn assumes `default_osl` output tokens. A request larger than a whole replica reserves that replica's full capacity, so it runs on an idle replica.
3. **Placement.** The replica that served the trajectory's last turn still holds most of its context, so it is offered alone when it fits. Otherwise every replica that fits is offered, and the profile's scorers choose among them.
4. **Queueing.** When no replica fits, the request waits until a reservation is freed.
5. **Release.** When a request finishes, its reservation is freed and its output length becomes the estimate for the trajectory's next turn.

A trajectory is identified by the request id the integration passes, which verl keeps for every turn of a trajectory.

## Configuration

```yaml
profiles:
  default:
    flow_control:
      type: kv_saturation
      default_osl: 1024   # output tokens assumed for a trajectory's first turn
    scorers:
      - type: least_queue
        weight: 1.0
    picker:
      type: max_score
```

## Integration support

| Integration | Where requests wait | Order of retries |
| --- | --- | --- |
| verl (vLLM engines) | one queue in the fleet actor shared by every agent-loop worker | longest prompt first, any request that fits |
| Ray Serve | the router | first come, first retried |

In verl, placement moves into the fleet actor while a flow-control plugin is configured, so reservations and the queue are shared by all agent-loop workers.
