# verl-native SWE A/B - plain-pod sandbox regime

Store-vs-recompute pair on verl 0.8.0's native agent loop (PR-62 harness),
Qwen2.5-7B-Instruct, 12 GRPO steps, batch 16 x n 4, seed 42, 8xH200 single
node, sandboxes as plain pods on `sandbox-pool`. This regime predates the
gVisor/agent-sandbox setup; the gVisor pair is a separate regime (S7).

## Recorded artifacts

| file | what it records |
|---|---|
| `recompute_driver.log.gz` | full trainer stdout, recompute arm, steps 1-12 (the 12 `step:N` lines are verl's emitted step metrics) |
| `recompute_master.txt` | mooncake master `/metrics` scrape at recompute harvest (that arm mounts no segments; baseline reference) |
| `store_driver.log.gz` | full trainer stdout, store arm, steps 1-12, exit 0; 1TiB store (8 x 128gb segments), DecodeKVSavingConnector + sha256_cbor |
| `store_bysource_midrun.txt` | mid-run engine counter observations (by_source split, mooncake op counters) with provenance; see header |
| `store_master_postrun.txt` | master `/metrics` after the store run; reads zero (segments unmounted after engines died with the node reboot) - kept as-recorded |

Both arms: zero `rollout-error` rewards. Rewards are ~all zero (recompute
step 2 mean 0.015625 = exactly one solved trajectory, proving the grading
path); Qwen2.5-7B rarely solves these tasks, so GRPO advantages are ~0 and
weights are ~static across steps in both arms.

## VALIDITY: store arm NOT comparable beyond step 1

Recorded `actor/entropy` per step: recompute 0.24-0.30 for all 12 steps;
store 0.287 at step 1, then 3.7-4.3 for steps 2-12. In the same window the
store arm's `response_length/clip_ratio` jumps from 0.016 (step 1, matching
recompute) to 0.19-0.44, per-trajectory LLM time triples, and total tokens
inflate ~25%. Step 1 is the only step with an empty external store (nothing
to load); external loads begin at step 2 after verl's per-step
`reset_prefix_cache(reset_connector=True)`. With weights ~static (zero
advantages), stale KV cannot explain this; the evidence points at the
cross-step external-load path serving wrong KV while reporting success
(`load_get` failed_keys=0). The mooncake transfer-engine P2P handshake
errors in the store driver log ("malformed json ... length: 0", engine
init) are noted as a possible related signal.

Consequence: step-1-only may be compared across arms; steps 2-12 of the
store arm measure a corrupted serving path, and every store-favorable
aggregate (gen time -24%, tool time -35%) is confounded by degenerate
trajectories.

RESOLVED - see `../entropy-diagnosis/`: isolation runs exonerated the node
and the external-load path and convicted vLLM sleep mode's page remapping
under the connector's one-time RDMA registration of the KV pool. The next
pair runs `free_cache_engine=False` in both arms.

## Not recorded

- Recompute `prompt_tokens_by_source`: the metrics-scraper sidecar saw no
  engine ports (verl binds each engine's uvicorn to the node IP on a random
  port) and emitted nothing. The split does not exist anywhere for this arm.
- Store arm FINAL by_source totals: the GPU node rebooted at trainer
  teardown (recorded k8s event) and the worker pod's /tmp - including the
  60s snapshot log - was lost before harvest. Only the mid-run observations
  in `store_bysource_midrun.txt` survive.
- Per-step preemption counts: verl's `num_preempted` fields are -1
  (unpopulated) in every step line of both arms.
- Two failed store launches (segment-size OOMs, 8 mounts x 512gb then
  x 256gb) are quarantined outside the repo per the valid-runs-only rule;
  root cause recorded in `integration/verl/k8s/cm-mooncake-config.yaml`.
