# Entropy-explosion diagnosis: sleep mode x RDMA registration

The plainpod-pair store arm degraded from step 2 onward (entropy 0.29 ->
3.7-4.3, cap-hit ratio 0.19-0.44). Three 3-step isolation runs, back to
back on the same node, image, seed, and dataset, convicted the cause.

## Recorded verdicts (actor/entropy per step, from the driver logs here)

| run | config delta | step 1 | step 2 | step 3 | verdict |
|---|---|---|---|---|---|
| `control_recompute_3step` | recompute, post-reboot node | 0.307 | 0.249 | 0.287 | node exonerated |
| `store_saveonly_3step` | store, `kv_role: kv_producer` (loads impossible) | 0.300 | 4.350 | 4.262 | load path exonerated - explodes without any loads |
| `store_nosleep_3step` | store, `kv_both`, `free_cache_engine=False` (no sleep) | 0.277 | 0.254 | 0.246 | **clean - sleep convicted** |

Cap-hit ratios agree: no-sleep store 0.016/0.031/0.016 (recompute band);
sleeping store variants 0.19-0.44 from step 2.

## Mechanism

`mooncake/store/worker.py` `register_kv_caches()` registers the GPU KV
pool with the transfer engine ONCE at engine start
(`store.register_buffer(cache_storage.data_ptr(), nbytes)`), which pins
physical pages. verl's `--enable_sleep_mode` unmaps and remaps the KV
pool's physical pages around every training phase; the registration is
never refreshed, so after the first wake the pinned mapping and the KV
pool disagree and the engine generates degraded text - with transfers
still reporting success (`load_get` failed_keys=0 in the 12-step run).

Corroborating: every sleeping store run's teardown rebooted the GPU node
(recorded k8s "Node ... has been rebooted" events; 3 of 3), consistent
with violent unwinding of stale pinned registrations. The rllm-convergence
rig proved the same connector clean because its RayServe engines never
sleep.

Also recorded en route: vLLM 0.22.1 `CompletionOutput` has no
`num_preempted` field, so verl's `agent_loop/*/num_preempted` is -1
(unpopulated) on this stack; per-run preemption tracking must come from
`vllm:num_preemptions_total`.

## Consequences

- Valid A/B design on colocated verl: `free_cache_engine=False` in BOTH
  arms (7B at gmu 0.30 coexists with FSDP training on H200).
- Proper fix (separate work): re-register connector buffers after
  `wake_up`, or deregister on sleep.
