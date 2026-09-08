# verl-native SWE A/B - plain-pod sandbox regime

Store-vs-recompute pair on verl 0.8.0's native agent loop (PR-62 harness),
Qwen2.5-7B-Instruct, 12 GRPO steps, batch 16 x n 4, seed 42, 8xH200 single
node, sandboxes as plain pods on `sandbox-pool`. This regime predates the
gVisor/agent-sandbox setup; the gVisor pair is a separate regime (S7).

## Recorded artifacts

| file | what it records |
|---|---|
| `recompute_driver.log.gz` | full trainer stdout, recompute arm, steps 1-12 (the 12 `step:N` lines are verl's emitted step metrics) |
| `recompute_master.txt` | mooncake master `/metrics` scrape at harvest (recompute arm mounts no segments; baseline reference) |

## Not recorded

- Recompute `prompt_tokens_by_source`: the metrics-scraper sidecar saw no
  engine ports (engines bind in ray-worker's netns) and emitted nothing.
  The split does not exist anywhere for this arm.
- Store arm: in flight at commit time; artifacts land here only if the run
  passes the validity gates. Two failed store launches (segment-size OOMs,
  8 mounts x 512gb then x 256gb) are quarantined outside the repo per the
  valid-runs-only rule; their root cause is recorded in
  `integration/verl/k8s/cm-mooncake-config.yaml`.
