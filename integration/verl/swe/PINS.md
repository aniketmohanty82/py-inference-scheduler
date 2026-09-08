# Vendored sources and pins for the verl-native store-vs-recompute A/B

## Vendored from PR #62

| field | value |
|---|---|
| PR | https://github.com/llm-d-incubation/py-inference-scheduler/pull/62 (OPEN, unmerged) |
| branch | `kfswain/py-rl-scheduler:swe-rl` |
| commit | `b99ebde3d26cae1b4f09734303e6a1dfd2b51db8` ("Guide updates", 2026-09-03 20:54 UTC) |
| fetched as | `git fetch https://github.com/llm-d-incubation/py-inference-scheduler.git pull/62/head:pr-62-swe-rl` |

Vendored: `integration/verl/swe/*`, `integration/verl/helpers/*`,
`integration/verl/examples/{run_swe.sh,swe_agent_loop.yaml,runtime-env-swe.yaml}`,
`integration/verl/hook_compat_check.py`, `configs/swe_{sandbox_example,sandbox_rbac,image_prewarm_job}.yaml`.

Also taken (independent bug fix): `backends/verl/{vllm,sglang}.py` now catch
`Exception`, not `ImportError` — vLLM import raises `AttributeError` from triton
on CPU-only nodes (the Ray head), so the narrow guard let it escape.

The PR is self-described as "much of this is claude made and will be refined";
re-diff against the pinned commit before pulling updates.

## Store-key structure (probed in the live engine, vLLM 0.22.1)

`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py`:

```
PoolKey.to_string() ->
  {model_name}@tp_rank:{n}@pcp{n}@dcp{n}@pp_rank:{n}@group:{group_id}@{chunk_hash}
```

`KeyMetadata` (model_name, tp_rank, pcp_rank, dcp_rank, pp_rank, group_id) is
constructed **once** at `store/worker.py:977`, i.e. per engine process, and
`chunk_hash` is a content hash of the token block (`prefix_caching_hash_algo=sha256_cbor`).

Consequences for multi-step runs:

- **No weight version anywhere in the key.** After a training step updates LoRA
  weights, KV written in step N is stale for step N+1: vLLM's own prefix cache
  is invalidated on weight update, the external store is not.
- **`group_id` is NOT a free namespace** — `store/coordinator.py` uses it for
  attention-group indexing (`(group_id, hash)` exists-sets, `kv_cache_group_ids`).
  Repurposing it would collide with that semantics.
- **`model_name` is the only namespace-shaped component**, but it is fixed for
  the engine's lifetime, so it cannot be bumped per step without restarting
  rollout engines.
- `kv_connector_extra_config` **does** reach the connector
  (`store/connector.py:92`), which is how `save_decode_kv` is delivered — so a
  custom namespace field could be plumbed through a subclass, but it would have
  to be mutated at runtime on the worker side, and no vLLM hook signals a
  weight update.

### Practical read

With **fresh tasks per step** (the chosen multi-step design), prompts differ
every step, so content hashes differ and cross-step key overlap is confined to
the shared system prompt. The exposure is therefore small but not zero, and
must be **measured, not assumed**: instrument per-step `external_kv_transfer`
attributable to keys written in earlier steps. If it is ~0, document and
proceed; if material, fall back to a store flush between steps.

Do not mid-run restart the mooncake master to flush: engines hold mounted
segments, and `integration/rllm/k8s/prelaunch-cleanup.sh` is designed for
*between-run* restarts (with a rollout-Ready wait added after the gate once
passed against a terminating pod and an engine lost the mount race).
