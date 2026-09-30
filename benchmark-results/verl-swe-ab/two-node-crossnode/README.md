# Store + flow control sharing KV across two nodes — SWE-bench agent RL, Qwen2.5-32B, 2 nodes / 8 engines, 12 steps

## TLDR

KV written by an engine on one node is served to engines on the other node,
under real training load, with our router deciding when a trajectory leaves
its engine and flow control holding turns back when every engine is full.
Over a 12-step run on two H200 nodes, **half of every KV key each node pulled
from the tier had been written by the other node** (50% on both), the
trainer stepped normally through it (gradient norms, entropy and the
rollout-versus-actor log-prob agreement all sit inside the single-node
arms' range), and both instrument gates, 23 checks in all, pass on the run
and on the 2-step smoke before it.

Cross-node sharing is a saturation phenomenon, and at eight engines this
workload saturates once. In the cold first step (36 turns per trajectory)
the fleet ran at KV occupancy 0.76, flow control parked 2,332 turns, 1,220
turns crossed nodes and the tier served 38–41% of prompt tokens on both
nodes. In the eleven steps after it, affinity kept 95% of turns on their
engine, the local prefix cache served 83–84% of prompt tokens, and the tier
was touched only by the 6–105 turns per step that moved.

Against the two arms the user asked for on the same nodes, verl alone and
verl with vLLM's local CPU offload, the cross-node tier is one of two offload
targets that both work: each cuts the prompt tokens the GPUs recompute from
15.7% to about 6% over 12 steps and each removes most of the saturated
step's sampling cost (per message turn -62% for the tier, -69% for the CPU
offload, against verl alone). With verl pinning every trajectory to one
engine, the local offload is the cheaper of the two here: it needs no lookup
RPC, no RDMA pull and no flow-control parking, and in the light steps it
costs nothing while the tier arm paid 35% more per turn. That surcharge was
our hook's own placement herd and per-decision polling, not the store (NOTE
H); with both fixed, a 2-step run of this arm spends 1.99 s per LLM call in
the light step against 2.00–2.08 s for verl alone (NOTE I). The metrics
scrape has since moved off the request path into a shared poller (NOTE K):
it halves the vLLM server actors' CPU and keeps the light-step gain, but
exposes the saturated step to a synchronised load wave that the old path's
latency had smoothed by accident. What only the
cross-node tier can do, let a trajectory continue on the other node, is
exercised throughout this run and is not needed for capacity at eight
engines and batch 128.

Two defects had to be fixed to get here, both now in the tree: two-node NCCL
hung the optimizer step until `NCCL_CUMEM_ENABLE=0`, and the first 12-step
attempt reproduced the "engine wedge" we had seen twice before with the
Mooncake store, a request-block pin leak in upstream's connector on steps
that schedule no tokens. Both two-node runs sample 20–25% fewer turns per
trajectory than every single-node arm on the same prompts; a two-node
control with verl's own balancer and no tier reproduces that deficit in
step 1 to within a turn, so it belongs to the two-node environment and not
to the KV path (NOTE C).

| 12 steps, 512 trajectories each | node 10.96.4.12 | node 10.96.5.16 |
|---|---|---|
| KV keys pulled from the tier | 160,407 | 149,605 |
| of which written by the other node | 50% | 50% |
| prompt tokens served from the tier | 14.8M (9.9%) | 17.0M (10.6%) |
| prompt tokens from local prefix cache | 125.4M (84.3%) | 133.5M (83.2%) |
| prompt tokens recomputed on GPU | 8.6M (5.8%) | 10.0M (6.2%) |
| tier share in step 1 alone | 40.7% | 38.0% |

| | single-node store arm | this run, 2 nodes | ours vs single node |
|---|---|---|---|
| rollout time/step (`timing_s/gen`), mean of 12 | 854 s | 707 s | -17% |
| sampling time per trajectory, mean of 12 | 56.3 s | 25.3 s | -55% |
| sampling time per message turn | 4.8 s | 2.9 s | -39% |
| turns per trajectory, mean of 12 | 24.3 | 18.3 | **-25%** |
| `critic/score/mean`, mean of 12 | 0.0194 | 0.0103 | |

---

## What we tested

The single-node runs showed a shared KV tier cutting sampling time by 65%
against local-only KV, with all four engines on one machine. The claim under
test here is the one that matters for scale-out: **that the tier lets a
trajectory continue on a different node from the one that computed its
context, and that flow control keeps the fleet stable while it does so.** If
that holds, a rollout is no longer bound to the engines of one machine.

Three things have to happen, and each has its own instrument:

| | what must happen | instrument |
|---|---|---|
| 1 | a trajectory's next turn is routed to an engine on the other node | the hook prints one `MOVE` line per move with both endpoints and `cross_node=1` when the host changes |
| 2 | that engine pulls the context from the tier rather than recomputing it, and the bytes come from the other node's segment | the connector prints `PULLSRC` lines: for a sampled load batch it asks Mooncake where each key's replica lives and counts keys whose owner is another host |
| 3 | the engine's own accounting agrees | `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}` on every engine of both nodes |

Plus one for flow control: `FLOWCONTROL park` and `drop` lines, and zero
fallbacks to verl's balancer.

---

## Setup

### Stack

| | |
|---|---|
| nodes | 2 x a3-ultragpu-8g (spot), GKE `rls-ab-west`, us-west1-c |
| GPUs | 16 x NVIDIA H200, 143 GB each |
| inference engines | 8, at tp=2, four per node |
| trainer | verl 0.9.0, fsdp2 over 16 ranks, Ulysses SP=2, LoRA r32/a32, lr 1e-6 |
| model | Qwen2.5-32B-Instruct |
| vLLM | 0.29.0 |
| trainer network | RoCE via Google's gIB NCCL plugin, `NCCL_CUMEM_ENABLE=0` |
| KV tier | Mooncake store, one master, one 128 GiB RDMA segment per engine, 8 RDMA NICs per node |
| connector | `RLPullPolicyConnector` = upstream `MooncakeStoreConnector` + pull admission (`RLS_MIN_PULL_TOKENS=1024`) + the two fixes in NOTES B and F, `save_decode_cache: true`, `sha256_cbor` hashing |
| router | `PyInferenceAgentLoopManager` + `configs/swe-crossnode-fc.yaml` |
| image | `rllm-verl-mooncake:swe14` = swe13 + the connector fixes, `sha256:3a068fd0f8ab1d9e28eed6ac60aa3eb7233bff64561fbd1e780d8c1d3773e181` |
| sandbox pool | 21 x e2-standard-16, gVisor |

The smoke ran on swe13 with the unpatched connector. The 12-step run ran on
swe13 pods with the fixed connector hot-patched in place; swe14 is that state
baked, and both RayCluster manifests now point at it.

### Workload

Identical to the single-node runs: 128 SWE-bench tasks x 4 generations, so
**512 trajectories per rollout**, `gpu_memory_utilization` 0.38, 15 s shell
command timeout, 20,000-character tool observations, 32-turn cap, 28,672-token
response cap. Same data order, so step *k* here samples the same 128 tasks as
step *k* of every single-node arm; prompt lengths match to the token.

With eight engines and the same batch, each engine carries half the load of
the single-node runs. That is the honest scale-out configuration, and it means
the fleet saturates in the long cold first step and rarely afterwards (NOTE E).

### How the router decides

Every turn of every trajectory is a routing decision. The profile
(`configs/swe-crossnode-fc.yaml`) runs these in order.

| | plugin | what it does | why |
|---|---|---|---|
| 1 | `saturation` filter, queue >= 8 or kv >= 0.98 | removes a saturated engine from the ballot. Falls through when every engine is over | the only way a trajectory leaves the engine holding its KV |
| 2 | `request_affinity`, weight 8.0 | votes for the engine that ran this trajectory's previous turn. No vote on a first turn | the context's KV is there; anywhere else pulls it from the tier |
| 3 | `waiting_queue` 4.0, `kv_cache` 2.0, `least_queue` 1.0 | place first turns and migrating turns by engine queue, KV occupancy and fleet-wide in-flight count | load awareness across both nodes |
| 4 | `jitter` 0.05 | breaks exact ties | otherwise engine 0 wins every tie |
| 5 | `simple_backpressure` flow control, kv 0.90 / waiting 4 | admits a turn only to an engine under both thresholds; when none qualifies the turn is parked and re-admitted with additive back-off | keeps the pool from being oversubscribed by admissions |

The 8 agent-loop workers share one in-flight ledger (a Ray actor), so every
worker sees the same fleet-wide queue. The gate checks that the ledger's sum
matches the engines' own running plus waiting (median ratio 1.03 over 842 busy
snapshots), that all 8 workers see all 8 engines, and that the per-engine
metrics are per engine (0 of 842 busy snapshots with identical numbers).

### The cross-node path

A turn moved from engine A on node X to engine B on node Y arrives at B with a
prompt whose prefix B has never seen. B's connector looks the prompt's block
hashes up in the store, finds the prefix that A saved (at decode and at turn
end), pre-allocates blocks for it and pulls them over RDMA from A's segment on
node X, then decodes without re-prefilling. The `PULLSRC` instrument wraps
that pull: it fetches the replica descriptors for the batch's keys and reads
each one's transport endpoint, whose host is the segment owner. Same-host
pulls ride the same path over the local NIC.

---

## Results

### Gates

Both runs passed both gates in full: the scheduler gate (8 workers with the
shared ledger, full endpoint view, populated scorer inputs, per-engine
attribution, ledger consistency, decisions on all 8 engines with a top share
of 14%, affinity kept share above 60%, zero fallbacks) and the cross-node gate
(moves across both hosts, cross-node moves, flow control live, `PULLSRC` from
both hosts with a non-zero other-host share, tier tokens on all 4 engines of
both nodes). Full text in `smoke/gate.txt` and `run/gate.txt`.

### 12 steps: per-step verl metrics

| metric | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout time/step (`timing_s/gen`) | 1,361 | 1,106 | 1,307 | 458 | 421 | 443 | 1,052 | 450 | 439 | 506 | 412 | 526 | 707 |
| sampling per trajectory, mean (s) | 71.7 | 18.7 | 21.7 | 20.9 | 20.2 | 20.0 | 19.1 | 20.3 | 21.4 | 22.3 | 26.3 | 21.4 | 25.3 |
| sampling per trajectory, max (s) | 558 | 132 | 131 | 183 | 150 | 131 | 217 | 163 | 258 | 166 | 268 | 478 | 236 |
| tool time per trajectory, mean (s) | 84.7 | 33.3 | 30.1 | 28.4 | 27.9 | 30.0 | 30.4 | 27.3 | 28.9 | 29.3 | 24.8 | 26.0 | 33.4 |
| tool time per trajectory, max (s) | 1,102 | 1,040 | 1,236 | 414 | 352 | 391 | 1,013 | 383 | 390 | 459 | 358 | 364 | 625 |
| `num_turns/mean` | 36.3 | 16.8 | 17.0 | 17.0 | 17.3 | 16.2 | 15.8 | 16.1 | 16.0 | 17.1 | 17.3 | 16.2 | 18.3 |
| `response_length/mean` | 8,171 | 3,229 | 3,281 | 3,565 | 3,548 | 3,064 | 3,014 | 3,350 | 3,067 | 3,457 | 3,320 | 3,412 | 3,707 |
| `critic/score/mean` | 0.0156 | 0.0059 | 0.0098 | 0.0098 | 0.0059 | 0.0117 | 0.0156 | 0.0020 | 0.0215 | 0.0059 | 0.0078 | 0.0117 | 0.0103 |
| `actor/entropy` | 0.187 | 0.224 | 0.232 | 0.202 | 0.214 | 0.222 | 0.217 | 0.226 | 0.223 | 0.210 | 0.198 | 0.202 | 0.213 |
| `actor/grad_norm` | 0.0031 | 0.0015 | 0.0029 | 0.0022 | 0.0021 | 0.0031 | 0.0033 | 0.0015 | 0.0046 | 0.0027 | 0.0039 | 0.0031 | 0.0028 |
| rollout vs actor log-prob diff, mean | 0.0033 | 0.0039 | 0.0040 | 0.0035 | 0.0037 | 0.0038 | 0.0037 | 0.0039 | 0.0037 | 0.0036 | 0.0035 | 0.0034 | 0.0037 |
| `timing_s/update_actor` | 190 | 73 | 74 | 80 | 80 | 69 | 72 | 76 | 72 | 78 | 74 | 77 | 85 |
| `timing_s/step` | 1,619 | 1,213 | 1,415 | 575 | 538 | 544 | 1,156 | 560 | 544 | 621 | 520 | 638 | 829 |

The single-node store arm on the same prompts: entropy 0.185 to 0.226 (mean
0.201), gradient norm 0.0008 to 0.0040 (mean 0.0024), rollout-versus-actor
log-prob difference 0.0033 at steps 1 and 2. Training on two nodes is
indistinguishable from training on one by these measures.

Rollout time/step is the tool tail, as in every earlier run: the five steps
above 1,000 s each contain one trajectory that spent 1,013 to 1,236 s in tool
calls while every engine sat idle (NOTE D). The seven steps without such a
trajectory took 412 to 526 s, against 420 to 671 s for the same prompts on one
node.

### 12 steps: routing, flow control and tier pulls

| step | decisions | kept on their engine | moves | cross-node moves | trajectories that crossed | parks | partial drops | peak running, 8 engines | KV occupancy, busy median | keys pulled, node 4.12 | keys pulled, node 5.16 | from the other node |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9,202 | 6,537 | 2,142 | 1,220 | 283 | 2,332 | 3,641 | 146 | 0.76 | 156,834 | 146,736 | 50% / 50% |
| 2 | 4,255 | 3,724 | 0 | 0 | 0 | 0 | 0 | 61 | 0.19 | 0 | 0 | |
| 3 | 4,303 | 3,736 | 67 | 33 | 33 | 0 | 381 | 92 | 0.21 | 808 | 94 | 49% / 51% |
| 4 | 4,306 | 3,780 | 13 | 9 | 9 | 0 | 197 | 72 | 0.21 | 0 | 0 | |
| 5 | 4,381 | 3,847 | 15 | 7 | 7 | 0 | 120 | 77 | 0.25 | 0 | 318 | - / 49% |
| 6 | 4,100 | 3,585 | 10 | 5 | 5 | 0 | 65 | 67 | 0.11 | 0 | 431 | - / 51% |
| 7 | 4,008 | 3,485 | 6 | 4 | 4 | 0 | 40 | 73 | 0.21 | 0 | 215 | - / 51% |
| 8 | 4,093 | 3,559 | 30 | 15 | 15 | 0 | 223 | 71 | 0.19 | 0 | 0 | |
| 9 | 4,047 | 3,510 | 30 | 15 | 15 | 0 | 145 | 76 | 0.14 | 2,028 | 346 | 49% / 51% |
| 10 | 4,326 | 3,769 | 45 | 28 | 28 | 0 | 313 | 79 | 0.23 | 447 | 408 | 51% / 52% |
| 11 | 4,369 | 3,749 | 105 | 64 | 64 | 0 | 420 | 86 | 0.23 | 0 | 799 | - / 48% |
| 12 | 4,111 | 3,569 | 20 | 16 | 16 | 0 | 216 | 82 | 0.22 | 290 | 258 | 53% / 48% |
| all | 55,501 | 46,850 (95%) | 2,483 | 1,416 (57%) | 479 of 6,144 | 2,332 | 5,761 | | | 160,407 | 149,605 | 50% / 50% |

Keys are counted once per engine (the two TP ranks of an engine pull the same
keys, each its own shard). Pulls in steps 2–12 are sampled at one load batch
in fifty, so a step can show zero when its few dozen batches fell between
samples; the engine counters below are exact.

### 12 steps: what each node's engines served

| step | tier share, node 4.12 | tier share, node 5.16 | recomputed, node 4.12 | recomputed, node 5.16 |
|---|---|---|---|---|
| 1 | 40.7% | 38.0% | 6.3% | 6.4% |
| 2 | 1.4% | 0.1% | 6.7% | 6.7% |
| 3 | 2.3% | 2.3% | 5.1% | 6.2% |
| 4 | 0.4% | 0.7% | 6.5% | 5.8% |
| 5 | 1.6% | 0.6% | 5.1% | 6.6% |
| 6 | 1.3% | 1.1% | 5.7% | 5.9% |
| 7 | 0.8% | 3.0% | 6.5% | 6.4% |
| 8 | 1.6% | 1.6% | 6.0% | 7.6% |
| 9 | 3.6% | 0.3% | 4.5% | 6.3% |
| 10 | 2.0% | 4.6% | 4.9% | 6.6% |
| 11 | 0.5% | 1.9% | 8.1% | 5.2% |
| 12 | 1.2% | 0.0% | 4.2% | 3.5% |
| all | 9.9% | 10.6% | 5.8% | 6.2% |

Shares are of prompt tokens (`vllm:prompt_tokens_by_source_total`), the
remainder being local prefix-cache hits. Per-step rows are differences of
per-minute engine snapshots aligned to the step's end and can be off by a
minute of traffic; the "all" row is the engines' own totals. The
recompute share is flat at about 6% whether or not the tier is busy: it is
the part of every turn the cache can never hold, the new tokens.

### The smoke: 2 steps on the unpatched connector

| step | rollout time/step | sampling per trajectory, mean / max | tool time per trajectory, mean / max | `num_turns/mean` | `response_length/mean` | `critic/score/mean` | `update_actor` |
|---|---|---|---|---|---|---|---|
| 1 | 694 s | 139.3 s / 472.6 s | 79.8 s / 418.6 s | 44.38 | 10,400 | 0.0215 | 237 s |
| 2 | 443 s | 16.8 s / 329.8 s | 29.2 s / 389.7 s | 14.10 | 2,654 | 0.0020 | 64 s |

Step 1 of the smoke: 4,526 moves (2,597 cross-node, 406 trajectories), 9,137
parks, 6,624 partial drops, KV occupancy 0.91 over the busy window, 49–50% of
pulled keys from the other node, tier share 53–55% on both nodes. Those
pressure figures are inflated by the pin leak of NOTE B; the cross-node
evidence stands.

### Comparison arms on the same two nodes

Two further 12-step arms ran on the same pods and prompts after the tier run:
verl alone (verl's balancer, KV local to each engine, nothing offloaded;
`verl-alone/`) and verl with vLLM's native CPU KV offload (128 GiB of pinned
host memory per engine, per node, so a turn that lands on the other node
cannot find its context; `cpu-offload/`). The three arms differ only in where
offloaded KV can live: another node's memory over RDMA, this node's host
memory, or nowhere.

| verl alone, 2 nodes | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout time/step (`timing_s/gen`) | 1,138 | 407 | 445 | 463 | 1,163 | 513 | 994 | 583 | 519 | 416 | 1,029 | 426 | 675 |
| sampling per trajectory, mean (s) | 188.0 | 19.5 | 17.1 | 18.1 | 22.6 | 18.7 | 20.4 | 20.4 | 17.3 | 20.8 | 17.5 | 17.2 | 33.1 |
| sampling per trajectory, max (s) | 874 | 171 | 122 | 129 | 235 | 179 | 314 | 508 | 311 | 196 | 258 | 154 | 287 |
| tool time per trajectory, mean (s) | 71.1 | 47.2 | 44.3 | 42.6 | 45.6 | 46.4 | 43.8 | 39.7 | 39.9 | 40.9 | 43.6 | 37.1 | 45.2 |
| tool time per trajectory, max (s) | 978 | 366 | 408 | 418 | 1,128 | 464 | 978 | 420 | 432 | 348 | 1,004 | 383 | 611 |
| `num_turns/mean` | 35.7 | 21.2 | 20.7 | 20.4 | 21.1 | 20.3 | 19.8 | 19.2 | 18.4 | 20.3 | 19.7 | 18.9 | 21.3 |
| `response_length/mean` | 8,081 | 4,333 | 3,753 | 4,065 | 4,082 | 4,003 | 4,127 | 4,015 | 4,195 | 4,639 | 4,021 | 3,909 | 4,435 |
| `critic/score/mean` | 0.0215 | 0.0137 | 0.0176 | 0.0098 | 0.0078 | 0.0059 | 0.0098 | 0.0078 | 0.0137 | 0.0098 | 0.0156 | 0.0098 | 0.0119 |
| `actor/entropy` | 0.194 | 0.217 | 0.224 | 0.216 | 0.208 | 0.213 | 0.209 | 0.217 | 0.225 | 0.213 | 0.219 | 0.222 | 0.215 |
| `timing_s/update_actor` | 187 | 93 | 82 | 89 | 89 | 86 | 91 | 90 | 94 | 101 | 90 | 88 | 98 |

| verl + CPU offload, 2 nodes | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout time/step (`timing_s/gen`) | 707 | 469 | 502 | 447 | 1,028 | 443 | 575 | 613 | 474 | 464 | 494 | 470 | 557 |
| sampling per trajectory, mean (s) | 55.3 | 16.6 | 19.0 | 17.8 | 19.2 | 15.8 | 18.6 | 20.3 | 17.7 | 19.7 | 18.0 | 17.9 | 21.3 |
| sampling per trajectory, max (s) | 535 | 186 | 157 | 178 | 210 | 299 | 462 | 455 | 449 | 356 | 389 | 455 | 344 |
| tool time per trajectory, mean (s) | 97.1 | 44.1 | 39.1 | 40.4 | 37.6 | 37.3 | 35.2 | 34.4 | 33.2 | 37.3 | 35.4 | 34.2 | 42.1 |
| tool time per trajectory, max (s) | 456 | 430 | 463 | 413 | 988 | 397 | 357 | 471 | 393 | 428 | 450 | 379 | 469 |
| `num_turns/mean` | 34.7 | 18.9 | 19.8 | 18.9 | 19.8 | 17.6 | 18.2 | 19.0 | 17.5 | 19.4 | 18.4 | 17.8 | 20.0 |
| `response_length/mean` | 8,495 | 3,774 | 4,155 | 4,035 | 4,323 | 3,569 | 4,236 | 4,255 | 3,445 | 3,999 | 3,952 | 4,032 | 4,356 |
| `critic/score/mean` | 0.0117 | 0.0098 | 0.0078 | 0.0137 | 0.0078 | 0.0215 | 0.0117 | 0.0098 | 0.0117 | 0.0078 | 0.0039 | 0.0117 | 0.0107 |
| `actor/entropy` | 0.193 | 0.219 | 0.211 | 0.209 | 0.204 | 0.223 | 0.188 | 0.190 | 0.194 | 0.195 | 0.205 | 0.209 | 0.203 |
| `timing_s/update_actor` | 200 | 84 | 92 | 90 | 96 | 80 | 96 | 94 | 77 | 89 | 89 | 91 | 98 |
| offload share of prompt tokens, node 4.13 / 5.17 | 50% / 44% | 0.0% / 0.0% | 1.7% / 2.8% | 0.5% / 3.5% | 0.2% / 0.1% | 0.1% / 0.3% | 0.4% / 0.6% | 0.1% / 0.3% | | 0.7% / 2.0% | 2.9% / 0.9% | | 9.2% / 9.2% |

The CPU offload engaged where the tier did: 44–50% of prompt tokens in the
saturated first step, 0–3.5% after it, 9.2% over the run on both nodes, with
6.3% recomputed.

| same two nodes, same prompts | verl alone | verl + CPU offload | our tier + flow control |
|---|---|---|---|
| prompt tokens recomputed on GPU, 12 steps, both nodes | 58.8M (15.7%) | 22.9M (6.3%) | 18.6M (6.0%) |
| prompt tokens served from offloaded KV | 0 | 33.4M (9.2%) | 31.8M (10.2%) |
| sampling per trajectory, step 1 | 188.0 s | 55.3 s | 71.7 s |
| sampling per message turn, step 1 | 10.8 s | 3.3 s | 4.1 s |
| sampling per trajectory, steps 2–12 mean | 19.1 s | 18.2 s | 21.1 s |
| sampling per message turn, steps 2–12 | 2.0 s | 2.1 s | 2.7 s |
| LLM path per LLM call, steps 2–12 (`generate_sequences` ÷ assistant turns) | 2.00 s | 2.06 s | 2.71 s |
| tool time per LLM call, steps 2–12 (`tool_calls` ÷ assistant turns) | 4.28 s | 3.97 s | 3.46 s |
| rollout time/step, 12-step mean | 675 s | 557 s | 707 s |
| steps with a stuck-command tail above 900 s | 4 | 1 | 5 |
| turns per trajectory, step 1 | 35.7 | 34.7 | 36.3 |
| turns per trajectory, steps 2–12 mean | 20.0 | 18.7 | 16.6 |
| `critic/score/mean`, 12-step mean | 0.0119 | 0.0107 | 0.0103 |
| `timing_s/update_actor`, 12-step mean | 98 s | 98 s | 85 s |

Read per regime. Either offload target removes most of the saturated step's
sampling cost against verl alone: -62% per message turn for the cross-node
tier, -69% for the local CPU offload, and both cut the prompt tokens the GPUs
recompute over the run from 15.7% to about 6%. Between the two offload arms
the local one is cheaper on this fleet: with verl pinning every trajectory to
one engine, that engine's own 128 GiB host buffer already holds what it
evicts, and it pays no lookup RPC, no RDMA pull and no flow-control parking.
In the eleven light steps, where eight engines never fill, the CPU offload
costs nothing over verl alone while the tier arm pays 35% more per assistant
turn. NOTE H traces that surcharge to the hook, not the store: per request the
engines are as fast in this arm as in the CPU-offload arm, and the extra time
is a fresh-placement herd that leaves two engines carrying most of a node's
turns plus a metrics scrape on every decision that lands on the vLLM server
actors' event loops. Per message turn divides by `num_turns`, which counts
assistant and tool messages plus the prompt; per LLM call divides by the
assistant turns alone, (`num_turns` − 1) / 2. With the hook fixed, the
light-step LLM path per call is 1.99 s (NOTE I). The
cross-node tier's distinctive property, that a trajectory can continue on
the other node, is exercised throughout this run (half of all pulled keys
came from the other node) but is not needed for capacity at eight engines and
batch 128. Rollout wall time does not separate the arms in any regime because
the tool tail sets it (NOTE D): the CPU-offload arm's lower mean is one stuck
step against four and five. The turn deficit after the first update is NOTE
C's remaining question.

---

### Measured costs on this fleet

Every number below is a measurement from these runs, not an estimate; the
earlier per-token pricing note derived pull and recompute costs under queueing
and is superseded by this table.

| cost | measured | instrument |
|---|---|---|
| store lookup (`batch_is_exist`) | 0.7 ms; 3.2 ms with the master saturated | Mooncake master metrics, 12-step run |
| store put | 3.6 ms per job | same |
| pull into GPU, per TP rank | 107–138 ms for 820–990 MB, 7–8 GB/s | `PULLSRC` timing |
| pinned host-to-device copy | 51.5 GiB/s | worker probe |
| cold prefill of a never-seen prompt, 32B tp=2 + LoRA, idle engine | 0.13 s at 2.1k tokens, 0.54 s at 8.5k, 1.19 s at 17k, 2.14 s at 28k; 16k falling to 13k tokens/s | `scripts/prefill_probe2.sh`, 09-26 |
| repeat of a cached prompt | 31 ms at 2.1k tokens, 126 ms at 28k | same |
| a turn on a cached context, ~560 new tokens | 58 ms at 2.1k, 97 ms at 8.5k, 122 ms at 17k, 159 ms at 28k | same |
| engine queue wait, light step | 1–10 ms mean, both arms | `/metrics` histograms (`scripts/hist_delta.py`) |
| engine queue wait, saturated step | 0.14 s and 0.49 s mean per node, CPU-offload arm, 25 and 85 preemptions | same |
| time to first token, light / saturated step | 70–156 ms / 0.25–0.63 s | same |
| decode per output token, light step | 15–17 ms on a balanced engine, 21 ms on one carrying twice its share | same |
| hook decision (`schedule()`), light / saturated step | 17–21 ms / 33–83 ms mean, of which the metrics refresh 15–19 / 25–54 ms; maxima 96 ms / 628 ms | `SCHEDT` |
| one `/metrics` scrape as the hook issued it | 6–44 ms for the HTTP render, 15–83 ms end to end with parsing, on a loaded engine | curl and a Python probe on a live engine |
| LLM path per call in light steps, agent loop's view | 2.00 s verl alone, 2.06 s CPU offload, 2.71 s ours | `timing_s/agent_loop/generate_sequences` ÷ assistant turns |

A 6k-token context, the mean per-turn context here, costs about 0.39 s to
recompute cold and about 0.2 s to pull at the measured rate, before any
queueing on either path.

## What the instruments showed

**A.** *Sharing follows saturation.* Every move is a turn the saturation
filter or flow control pushed off its engine, and 57% of moves crossed the
node boundary in both directions (764 one way, 652 the other over the run):
with load balanced across the fleet the destination is as likely to be on the
other node as on this one. Step 1 is where the fleet was saturated (KV 0.76
busy median, 146 requests running, 2,332 parks) and it holds 86% of the run's
moves and 98% of its pulls. Steps 2–12 ran at KV 0.11 to 0.25; there the
filter had nothing to remove, affinity kept 95% of turns home and the moves
that did happen were first turns placed by load.

**B.** *Every pull is half remote.* The `PULLSRC` other-host share is 48–53%
in every step and on both nodes, including the sparse steps. That is the
signature of a tier with no locality preference: a moved turn's context is
wherever the previous engine saved it, and the previous engine was on this
node about half the time.

**C.** *The engines confirm it.* Tier tokens appear on all four engines of both
nodes in every busy step, and the recompute share stays at 6% regardless. In
step 1 the tier served 38–41% of prompt tokens per node with the local prefix
cache serving most of the rest; in the light steps the local cache served
83–84% and the tier the moved remainder.

**D.** *Training is unaffected.* Update time on 16 ranks is 69–80 s per step
after the first (single node: 181 s at step 2), gradient norms, entropy and
the rollout-versus-actor agreement sit inside the single-node band, and
nothing in 12 steps of two-node fsdp2 over gIB stalled once cuMem P2P was
off.

---

## Scrutiny

> **NOTE A — four launches hung at the first optimizer step; the fix is
> `NCCL_CUMEM_ENABLE=0`.** Plain NCCL never connects over A3 Ultra's RoCE
> fabric from inside a pod. Google's gIB plugin (installed by the
> `nccl-rdma-installer` DaemonSet, mounted at `/usr/local/gib`, its
> `nccl.conf` and tuner config applied) gives 102 GB/s between the two pods
> and passes 16-rank synthetic collectives in seconds. The trainer still
> wedged on gIB and on sockets alike: all 16 ranks blocked CPU-side in the
> gradient-norm all-reduce of the first optimizer step, one rank's 2-rank
> sequence-parallel communicator stuck in lazy channel setup "via
> P2P/CUMEM", no CUDA work enqueued so no watchdog and no flight-recorder
> dump. With cuMem-based P2P disabled the channels come up "via P2P/IPC" and
> the step completes. Both settings are in `swe-raycluster-2node.yaml`.

> **NOTE B — the "engine wedge" is a block-pin leak in upstream's connector,
> and the smoke's pressure numbers include it.** The first 12-step attempt
> stopped in step 1's tail with one engine at KV occupancy 1.0, nothing
> running, eight turns waiting for capacity, and the scheduler re-looking up
> those turns' full prefixes twelve times a second. Upstream's Mooncake
> connector pins every block of a request for each store job it emits and
> frees them only when both TP ranks report the job done; the worker queues
> store jobs from `wait_for_save()` alone, and vLLM does not call that hook
> on a step that schedules no tokens. A job emitted on such a step, which is
> what a new turn parked on an async pull looks like on an idle engine, is
> never run and never reported, and its blocks stay pinned for the rest of
> the run. Four max-length turns pin a 106k-token pool. Our subclass now
> flags zero-token steps on the scheduler side and queues the jobs on the
> worker side (`rl_pull_policy.py`); a first version keyed on the forward
> context queued jobs twice and tripped upstream's "reported by too many
> ranks" assert, which is why the flag travels in the connector metadata.
> The same signature was behind two unexplained wedges in earlier
> single-node store runs. Because the smoke ran on the unpatched connector,
> part of its step-1 pressure was phantom: with the fix, the same prompts
> produced 2,142 moves and 2,332 parks against the smoke's 4,526 and 9,137,
> and per-trajectory sampling time halved. The 12 fixed steps never showed
> the signature.

> **NOTE C — two-node trajectories are 25% shorter than single-node ones on
> the same prompts, and the cause is not yet established.** 36.3 turns at
> step 1 and 15.8–17.3 after it, against 44.7–46.0 and 21.3–24.5 in all three
> single-node arms; response length and score down in proportion; the smoke
> showed the same at step 2 (14.1). It is not the trainer: learning rate,
> mini-batch and sequence-parallel settings are identical in both logs,
> gradient norms and entropy sit inside the single-node band, and the
> rollout-versus-actor log-prob agreement is as tight. It is not a surfaced
> infrastructure failure: the agent loop's salvage path, which turns a
> sandbox or tool failure into a truncated trajectory, logged nothing in
> either two-node run, on either node. Prompt lengths match to the token.
> What differs is the tier being pulled across nodes, the sixteen-rank update,
> and the west cluster's sandbox pool (built the same day; NOTE D). A 2-step
> control on the same two nodes and pods with verl's own balancer, no tier
> and no hook (`control/`) settles the first step: **36.9 turns, response
> length 8,169, score 0.0195, against this run's 36.3, 8,171 and 0.0156**, and
> that step holds 98% of the run's cross-node pulls. Whatever shortens
> two-node trajectories before any training is shared by an arm that never
> touches the tier, which leaves the sandbox pool or the eight-engine layout.
> The tier's own effect on that step is the one measured on one node:
> sampling per trajectory 71.7 s against the control's 194.5 s. After the
> first update the picture is less tidy. Over steps 2–12 the three two-node
> arms average 20.0 (verl alone), 18.7 (CPU offload) and 16.6 (this run)
> turns per trajectory, against 21.3–24.5 on one node, and the three
> single-node arms agreed on their 12-step means to within 0.3 turns. The
> tier arm's trajectories are therefore about 17% shorter than verl alone's
> after the first update, in an ordering that follows how much KV each arm
> offloads, even though the tier was barely used in those steps (zero to 105
> moves, 0–4.6% of prompt tokens). Nothing measurable explains it: no salvage
> events, identical training metrics, rollout-versus-actor agreement as
> tight as the single-node runs, prompt lengths identical to the token. Two
> more 2-step runs of this arm on 09-25 and 09-26 put the spread on the
> table: step-2 turns of 16.8 (this run), 19.2 and 17.6, against 18.9 and
> 19.2 for the CPU-offload arm's two runs and 21.2 and 20.0 for verl alone's.
> The gap to the CPU-offload arm is inside this arm's own run-to-run spread;
> the gap to verl alone, about 2.5 turns, has held in every sample so far
> and is still unexplained. The experiment that would assign it is a
> two-node Mooncake arm without the hook and flow control.

> **NOTE D — rollout wall time is the tool tail, and this pool was crowded.**
> In five of twelve steps one trajectory spent 1,013 to 1,236 s in tool calls
> after every engine had drained; those steps took 1,052 to 1,361 s against
> 412 to 526 s for the rest. That is the stuck-command loop characterized in
> the single-node analysis. It was aggravated here by 279 sandboxes orphaned
> by the wedged and crashed attempts, which held about 140 of the gVisor
> pool's 334 CPUs for the whole run (934 "insufficient cpu" scheduling
> failures in one hour) and by five gVisor nodes hitting disk pressure during
> step 1 (10 sandbox evictions at 06:35). Rows that scale with wall time are
> not a property of the KV tier.

> **NOTE E — eight engines at batch 128 is half the per-engine load of the
> single-node arms.** The fleet saturates in the cold first step and not in
> the 16-turn steps after it, so cross-node moves and parks concentrate in
> step 1 and the tier serves later steps mostly through local hits. A
> matched-pressure run would double the batch, but 1,024 concurrent
> sandboxes at 500m CPU each do not fit the sandbox pool.

> **NOTE F — a start-up race killed one launch.** Upstream derives the
> lookup-RPC socket path from host and data-parallel rank only, so four
> co-located engines share one IPC path and race each other's unlink-then-bind;
> one engine died with `EADDRINUSE`. The subclass now keys the path on the
> engine's instance id; four distinct socket files per node were confirmed.

> **NOTE G — the smoke's per-minute engine series is sparse and the run's
> was harvested by hand.** The original scraper probed every listening port
> on the pod each minute, stalled up to ten minutes per sweep on NCCL and
> Mooncake listeners, captured engine blocks in 2 of 24 snapshots, and was
> the source of the `SocketHandShakePlugin` errors in the engine logs (HTTP
> into Mooncake's handshake port, rejected and harmless). `scripts/snap13b.py`
> resolves engine ports through the vLLM server processes and produced the
> per-minute series for the 12-step run. The driver itself died on a syntax
> error at the end of the run, because its script was edited on disk while
> bash was still reading it; the trainer had already finished with exit code
> 0 and the artifacts were collected by the same commands by hand.

> **NOTE H — the light-step surcharge is the hook, not the store.** In the
> eleven light steps the tier arm spends about 30% more per LLM call than
> verl alone or the CPU-offload arm (2.71 s against 2.00 and 2.06 s, agent
> loop `generate_sequences` time per assistant turn, steps 2–12 means) while
> its tool time per call is lower (3.46 s against 4.28 and 3.97 s). The
> engines do not see the surcharge. Two 2-step runs on the same pods with the
> full per-request histograms scraped every minute, ours at 22:47 and CPU
> offload at 23:13 UTC on 09-25, put step 2's steady state side by side
> (first busy minute excluded; vLLM stamps these with engine-core time):
>
> | step-2 engine means | ours knk6d | ours wr8ms | CPU offload knk6d | CPU offload wr8ms |
> |---|---|---|---|---|
> | requests | 1,391 | 2,669 | 1,566 | 1,468 |
> | time to first token | 0.070 s | 0.114 s | 0.082 s | 0.078 s |
> | queue wait | 0.003 s | 0.010 s | 0.002 s | 0.001 s |
> | prefill | 0.053 s | 0.084 s | 0.063 s | 0.059 s |
> | decode per output token | 15.2 ms | 21.0 ms | 16.7 ms | 16.5 ms |
> | end to end | 1.34 s | 2.32 s | 1.87 s | 1.80 s |
>
> Time to first token, queue wait and prefill agree to within noise, and the
> one difference, 21 against 15 ms per decode token, follows load, not the
> connector: two engines on node wr8ms carried 45% and 40% of that node's
> step-2 prompt tokens and the other two 7% and 8%, where verl's balancer
> spread the same step 25/25/25/25. Everything in the vLLM server actor's
> event loop and in the request path before `add_request` is invisible to
> these histograms and visible to the agent loop's timer; that is where the
> 0.6 s per call sits. In the saturated step the same instrument favours
> this arm: a third default run's step 1 against the CPU-offload smoke's,
> both minus the first minute:
>
> | step-1 engine means | ours knk6d | ours wr8ms | CPU offload knk6d | CPU offload wr8ms |
> |---|---|---|---|---|
> | requests | 4,371 | 4,183 | 3,194 | 3,238 |
> | time to first token | 0.19 s | 0.21 s | 0.25 s | 0.63 s |
> | queue wait | 0.07 s | 0.10 s | 0.14 s | 0.49 s |
> | decode per output token | 20.4 ms | 19.8 ms | 21.0 ms | 24.4 ms |
> | end to end | 2.25 s | 2.30 s | 2.34 s | 3.10 s |
> | preemptions | 0 | 11 | 25 | 85 |
>
> Flow control and the moves do what they are for: less queueing, fewer
> preemptions, more requests served. Two mechanisms make the light steps
> slower, both ours, both reproduced:
>
> 1. *A per-core burst herd.* verl composes the whole batch before any
>    scheduling task runs and the hook's refresh lock is FIFO, so all 64
>    decisions of a core are scored against one snapshot; the engine that
>    snapshot ranks lowest by even one in-flight request takes the entire
>    batch, and the jitter scorer only breaks exact ties. The first minute of
>    a fresh step placed 118 / 64 / 64 / 34 / 65 / 72 / 57 / 38 first turns on
>    the eight engines (verl: 64 each); affinity then kept every trajectory
>    on that engine and the skew lasted the step (1,509 against 814 completed
>    requests per engine by its end). A third default run's step 2 was worse:
>    four minutes in, one engine of node knk6d had completed 1,007 turns and
>    two others 77 and 81. Offline, with the real scheduler and this
>    profile, a snapshot with one engine one request lower gives
>    64/0/0/0/0/0/0/0; counting each dispatch on the endpoint before the next
>    decision gives 8 per engine (`scripts/herd_test2.py`). verl's balancer
>    cannot herd because its increment is atomic inside `acquire_server`.
> 2. *A metrics scrape on every decision.* Each decision called
>    `get_routing_stats` on all eight vLLM server actors, and each actor
>    rendered and regex-parsed its 72 KB `/metrics` on the asyncio loop that
>    also relays every token: 12–25 scrapes per second per engine in a light
>    step, 15–83 ms each under load, and a burst of 64 decisions serialised
>    behind 64 scrapes (`SCHEDT` maxima of 255–628 ms in step 1). The engine
>    core is idle through it: py-spy put the connectors at 0.2% of
>    `EngineCore` samples and 1.3% of the worker's, in both arms alike.
>
> The fix that stays in `integration/verl/verl_hook.py` is the first one:
> dispatch and release adjust the endpoint's in-flight count at once. The
> fix run (NOTE I) also carried a 0.1 s snapshot-reuse floor on the metrics
> refresh; it was removed afterwards because its own value was unproven and
> it traded freshness for speed, which is the wrong trade for this router.
> The fix run is reported in NOTE I. This retracts the earlier reading that blamed the
> lookup RPC and the decode saves: the lookup costs 0.7 ms and the engine-side
> numbers are equal.

> **NOTE J — known differences between the three two-node arms, and which way
> each one leans.**
>
> | difference | arms | leans |
> |---|---|---|
> | fresh placement by a per-core snapshot that herds, against verl's atomic counter | ours vs both others | against ours, every step (NOTE H) |
> | a metrics scrape on every decision, on the vLLM server actors' loops | ours only | against ours (NOTE H) |
> | flow-control parking counted inside `generate_sequences` | ours only | against ours' step-1 sampling time; it buys the lower queue wait and preemptions above |
> | decode-KV saves from the worker's send thread (`save_decode_cache=true`) | ours only | against ours' worker CPU (11% of samples in a CUDA event wait), invisible engine-side |
> | block hashes as `sha256_cbor`, needed for hashes to match across engines | ours only | a small scheduler CPU cost in ours, not quantified |
> | pod generation: ours and verl alone ran on the first pods (10.96.5.16 / 4.12), CPU offload on the pods recreated for its 700 GiB shm (10.96.5.17 / 4.13) | | same two nodes, same image family; the CPU-offload arm ran last, with the sandbox pool in the same state |
> | 279 orphaned sandboxes on the gVisor pool | all three alike | against every two-node arm relative to single node, not between arms (NOTE D) |
> | shorter trajectories in ours after the first update (16.6 against 18.7 and 20.0 turns) | | fewer tokens per step favours ours' wall time; the per-call comparison is unaffected |

> **NOTE I — the fix run: placement is even, and a third mechanism showed
> itself.** A 2-step run of this arm on 09-26 (`fix-run/`) with the patched
> hook: dispatches counted on the endpoint at once, and a fleet snapshot
> reused while younger than 0.1 s (the latter since removed, see NOTE H).
> First turns placed per engine in the
> first minute of step 1, the same instrument as NOTE H:
>
> | run | node knk6d | node wr8ms |
> |---|---|---|
> | default hook, 09-26 00:05 | 118 / 64 / 64 / 34 | 65 / 72 / 57 / 38 |
> | fixed hook, 09-26 00:55 | 83 / 59 / 57 / 59 | 59 / 57 / 57 / 81 |
> | verl's balancer (CPU-offload arm) | 64 / 64 / 64 / 64 | 64 / 64 / 64 / 64 |
>
> Six of eight engines now sit within two of each other. The two at 81–83
> are one worker's doing and expose a third mechanism, older than the other
> two: the hook learns the engine set by draining verl's balancer, and verl
> starts a step's 64 tasks at once, so 64 drains per worker ran concurrently
> (512 fleet-wide, each holding up to 24 un-released acquires) and skewed the
> balancer's counters while they ran; one worker's last drain saw only the
> two least-counted servers and kept that 2/8 view for its next 61
> decisions, all of which went to those two engines. Earlier runs show the
> same in milder form (5/8 to 7/8 views for 100–200 decisions each).
> Discovery now happens once for the whole fleet, inside the shared ledger
> actor (`integration/verl/shared_inflight.py`), and each worker reads the
> result with one call; that change is in the tree and not yet in a run. The
> hook's decision cost fell to 10–13 ms with the snapshot reuse (17–21 ms
> before in light steps, 33–83 ms in the saturated step). Step 2, where the
> view is complete from the first decision, is the clean test, and it
> passes: the fleet ledger at the burst read 9 / 10 / 10 / 10 / 8 / 9 / 9 / 10
> in-flight per engine (default runs: 17 and 20 on two engines, 0–3 on the
> rest), and the step's prompt tokens split 25 / 18 / 32 / 25% and 30 / 20 /
> 13 / 37% per node against 63 / 6 / 25 / 6% and 37 / 27 / 26 / 10% in run 3.
>
> | step 2, per LLM call | verl alone | CPU offload | ours, default hook | ours, fixed hook |
> |---|---|---|---|---|
> | LLM path (`generate_sequences` ÷ assistant turns) | 2.00–2.08 s | 1.93–2.06 s | 2.66–2.75 s | **1.99 s** |
> | engine end to end, steady state | | 1.80–1.87 s | 1.34–2.39 s | 1.68–1.97 s |
> | engine queue wait | | 1–2 ms | 3–10 ms | 1 ms |
> | time to first token | | 78–82 ms | 70–114 ms | 80–93 ms |
> | dispatch to completion, hook's view (`REQLAT`, kept turns) | | | | 1.5–2.1 s mean, 1.2–1.8 s median |
>
> The surcharge is gone: 1.99 s per call against 2.00–2.08 s for verl alone
> and 1.93–2.06 s for the CPU-offload arm, from 2.66–2.75 s in three default
> runs. Dispatch-to-completion as the hook sees it now matches the engine's
> own end-to-end, so nothing measurable is left between the agent loop and
> the engine. The saturated step moved the same way but less: 44.8 s of LLM
> path per trajectory against 48.5 and 49.7 s in the two default smokes, at
> the same turn count, and its engine-side queue wait, time to first token
> and preemptions stayed in run 3's range. Both gates pass (23 checks). The
> run's second step scored 0.0 where the other light steps scored 0.008 to
> 0.016; the hook does not touch scoring and one step is one sample. What
> this run does not restate is the 12-step three-arm table above, which was
> measured with the default hook; a 12-step repeat of this arm with all
> three fixes is the next run.

> **NOTE K — the metrics poller: off the request path, and what it
> uncovered.** Until 09-30 every scheduling decision scraped all eight
> engines through their vLLM server actors: a `/metrics` render and parse on
> each actor's event loop, 12–25 times a second per engine. The hook now runs
> the datalayer's `MetricsPoller` inside the fleet actor
> (`integration/verl/shared_inflight.py`, which also owns the engine map and
> the in-flight ledger) at `RLS_METRICS_INTERVAL_MS` = 100 ms; each worker
> mirrors that view into its endpoint attributes on the same interval, and a
> decision reads the mirror plus one exact fleet in-flight count taken under
> the scheduling lock. Two 2-step runs on 09-30, `poller-smoke1/` (poller
> alone) and `poller-smoke2/` (poller plus admission accounting):
>
> | | old path, calm runs (run 3, fix run) | poller, smoke 1 | poller + accounting, smoke 2 |
> |---|---|---|---|
> | vLLM server actor CPU, step 1 / step 2 | 34–37% / not sampled | 15–16% / 8–12% | 14–16% / 8–11% |
> | light step: engine end to end, queue wait | 1.68–1.97 s, 1 ms | 1.65–1.74 s, 1 ms | 1.70–1.81 s, 1–2 ms |
> | light step: LLM path per call | 1.99 s (fix run) | 2.11 s | 2.29 s, placement skewed by defect 3 below |
> | saturated step: peak engine queue depth | 2 | 44–56 | 8–9 |
> | saturated step: parks / moves | 3–12 / 324–493 | 6,195 / 3,530 | 11,887 / 4,482 |
> | saturated step: preemptions, both nodes | 11–27 | 224 | 238 |
> | saturated step: LLM path per trajectory | 44.8–48.5 s | 114.3 s | 105.2 s |
> | stale-metrics lines while routing | | 0 | 0 |
>
> Three defects surfaced, all ours, all fixed in the tree:
>
> 1. *Re-admission herd.* Flow control's watcher re-runs the gate before each
>    admission and assumes each re-run sees the previous admission. The
>    per-decision scrape made that true by accident, and throttled admissions
>    by 18–80 ms each. With a mirrored snapshot, eight workers' watchers
>    drained onto whichever engine had just dipped under a threshold: 56
>    waiting on one engine in smoke 1. Fix: the saturation helper counts the
>    router's own dispatches the engine has not reported yet (waiting is the
>    larger of the engine's queue and in-flight beyond running), and each
>    decision reads exact fleet counts. Smoke 2: 8–9 waiting at the peak.
> 2. *First-poll race.* The first decisions of a run scored on empty stats,
>    so every engine looked idle. Fix: the first decision waits for the
>    poller's first snapshot.
> 3. *Count read outside the lock.* The exact-count read in smoke 2 was issued
>    before the scheduling lock and applied under it, overwriting the dispatch
>    counts of decisions in between; a step's burst herded again (249 first
>    turns on one engine, 47–56 on its neighbours). Fix: the read happens
>    under the lock. The launch-time compatibility check now starts 48 first
>    turns at once against idle engines and requires an even spread; with the
>    fix it gives 16 / 16 / 16.
>
> The saturated step needed a same-day control before anything could be
> concluded, because the sandbox pool had changed between the poller smokes
> and every earlier run: 279 orphaned sandboxes were deleted on 09-29, and
> tool calls that took 2.6–3.0 s per turn in last week's calm runs take 2.2–2.3
> s since. Across all two-node runs, every run with tool calls under about
> 2.4 s per turn saturated in step 1 and every run above it stayed calm,
> hook or no hook. So on 09-30 the old metrics path was re-run on the same
> pool (`control-oldpath/`: scrape per decision, count fix, engine-reported
> waiting), followed by verl alone:
>
> | same day, same pool | verl alone | old path, scrape per decision | poller + accounting (smoke 2) | poller alone (smoke 1) |
> |---|---|---|---|---|
> | step 1: tool time per turn | 1.91 s | 2.32 s | 2.21 s | 2.27 s |
> | step 1: LLM path per trajectory / per call | 280.4 s / 13.9 s | 98.9 s / 5.2 s | 105.2 s / 5.3 s | 114.3 s / 5.5 s |
> | step 1: engine queue wait | 7.9–8.7 s | 1.03–1.19 s | 0.97 s | 1.86–2.68 s |
> | step 1: preemptions, peak queue depth | 33, 29 | 183, 17 | 238, 8–9 | 224, 44–56 |
> | step 1: parks / moves | none (no flow control) | 5,003 / 2,777 | 11,887 / 4,482 | 6,195 / 3,530 |
> | step 1: vLLM server actor CPU | not sampled | 63–66% | 14–16% | 15–16% |
> | step 2: LLM path per call | 2.35 s | 2.62 s | 2.29 s (placement skewed by defect 3) | 2.11 s |
> | step 2: engine end to end, queue wait | 1.89–1.98 s, 2 ms | 1.67–1.99 s, 2–6 ms | 1.70–1.81 s, 1–2 ms | 1.65–1.74 s, 1 ms |
> | gates | not applicable | 23 PASS | 23 PASS | 23 PASS |
>
> verl alone on the same pool takes 2.8 times the old path's LLM time through
> the saturated step, with eight seconds of engine queueing per request and
> no parks because it has no flow control; its light step runs at
> 2.35 s per call. The old path saturates the same way on this pool. The saturated step is
> the environment, not the metrics design; the poller with admission
> accounting matches the old path there on queue wait, first-token time and
> preemptions, keeps queues shallower, parks more because its gate closes
> earlier, and does it at a quarter of the server actors' CPU. In the light
> step, on the same day, it is faster per call (2.11–2.29 s against 2.62 s).
> An earlier reading of this note blamed the saturation on the poller
> removing an "accidental brake" of scrape latency; the 15-second fleet
> lines show the poller runs ramp no faster after the burst than the calm
> old-path runs did, and the same-day control retires that explanation.
> Deliberate pacing in flow control remains a capacity question for this
> workload at this tool speed, independent of how metrics are gathered; the
> KV-aware admission plugin already in the tree is the place for it if it is
> wanted.
>
> One observation to carry forward rather than explain: the three 2-step
> runs with even placement (fix run, both poller smokes) scored 0, 1 and 0
> successes out of 512 in their second step, against 4 to 8 in the three
> other 2-step runs (default hook, verl alone, CPU offload), and sampled
> fewer turns per trajectory there (14.7–17.7 against 17.6–20.0). Their
> first-step scores and training metrics (entropy, gradient norm,
> rollout-versus-actor agreement) sit inside the band of the others, so
> the update is not visibly different; this is NOTE C's turn deficit
> again, now with a score attached, and one step per run is not enough to
> separate it from the 0-to-8-in-512 noise of a light step.

---

## Files

| file | what it is |
|---|---|
| `run/gate.txt` | the scheduler gate (10 checks) and cross-node gate (13 checks) on the 12-step run, all PASS |
| `run/per_step.md` | the three per-step tables as generated by `scripts/harvest_x2.py` |
| `run/fcstore_driver.log.gz` | the trainer log with every hook and connector print, 12 steps |
| `run/pullsrc_w1.log.gz`, `run/pullsrc_w2.log.gz`, `run/pullsrc_driver.log.gz` | `PULLSRC` lines from each worker's Ray logs and from the driver log |
| `run/snap13_full_w1.log.gz`, `run/snap13_full_w2.log.gz` | per-minute engine `/metrics` series per node |
| `run/fcstore_scrape_w1.log.gz`, `run/fcstore_scrape_w2.log.gz` | the same series from the run's launch |
| `run/compat_check.txt`, `run/nccl_pair_test.txt`, `run/driver_ticks.txt`, `run/workers.txt`, `run/mooncake_master.log.gz` | preflight and bookkeeping |
| `smoke/` | the same set for the 2-step smoke |
| `control/` | the 2-step verl-alone control (driver log, ticks, scrapes) |
| `verl-alone/` | the 12-step verl-alone arm: driver log, per-step table, ticks, scrapes |
| `cpu-offload/` | the 12-step verl + CPU offload arm: the same set |
| `wedge/` | the wedged engine's full `/metrics`, py-spy dumps of its core and worker, the attempt's driver ticks |
| `default-run3/` | the third 2-step run of this arm with the default hook (09-26 00:00 UTC): gates, driver log, per-minute engine series with the full histogram set, `PULLSRC`, ticks, agent-loop decomposition (`turn_decomp.txt`); the run that supplied NOTE H's step-1 engine table and the 1,007 / 77 / 482 / 81 step-2 split |
| `fix-run/` | the 2-step run with the patched hook (NOTE I), same set plus the hook's `REQLAT` and `SCHEDT` lines |
| `poller-smoke1/`, `poller-smoke2/` | the two 2-step runs with the background metrics poller (NOTE K), the second with admission accounting; same set plus the standalone compatibility-check outputs and the queue gate's verdict; `scripts/poller_local_test.py` is the local Ray check of the fleet actor |
| `control-oldpath/` | the same-day control of NOTE K: the pre-poller hook re-run on 09-30's sandbox pool, with the exact hook files under `hook_files/` |
| `control-verlalone/` | the same-day verl-alone control of NOTE K (no hook, no tier), 2 steps on 09-30's pool |
| `scripts/` | `sched2.sh` (driver, forwards every knob to the head), `p40_arm.sh`, `sched_gate.py`, `crossnode_gate.py`, `analyze_crossnode.py`, `harvest_x2.py`, `snap13b.py` (per-minute engine scrape, full `vllm:` series), `nccl_pair_test.sh`; analysis: `hist_delta.py` (per-request latency means between two snapshots), `step_windows.py` (rollout windows in a scrape), `per_port_load.py` (per-engine load split), `turn_decomp.py` (LLM path vs tool time per trajectory and per call), `pyspy_summary.py`; probes: `prefill_probe2.sh` + `tail_probe2.sh` (cold/warm prefill on an idle engine in a rollout tail), `herd_test2.py` (offline reproduction of the fresh-placement herd, NOTE H) |
