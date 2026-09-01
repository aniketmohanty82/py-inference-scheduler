# Run standards — rllm convergence benchmarks

Agreed 2026-09-01. Every run is judged against these standards. A run that
violates any gate is a shakeout: it may be harvested for diagnosis but enters
no comparison table. Bugged jobs are discarded, not nursed.

## S1 — Environment fidelity (canonical DeepSWE execution)

Sandboxes run each task's own R2E docker image (`task.docker_image`). The
`python:3.11-slim` + repo-upload path is retired. Full-scale jobs start only
once the canonical environment runs error-free end to end; integration errors
during conversion are expected, are polled for continuously, and are fixed
iteratively — every error classified from raw log lines before a fix.

Gate: one known-good task passes `test.sh` in its real image before any
measured run.

## S2 — Recorded, never inferred

Only recorded metrics from reliable runs are used; the same applies to errors
(every claim cites raw log lines). Every stat is delivered with its record and
a reference to where it is recorded, or the answer is "I don't know".
Inference happens only when explicitly requested, and is labeled as such.

Raw artifacts retained per run: driver stream, per-pod session-log tarballs,
full unfiltered /metrics polls, master snapshots, Cloud Logging off-node
backup.

## S3 — Pre-launch gates (every run)

| Gate | Requirement |
|---|---|
| (a) Store state | Mooncake master restarted; segment registry verified empty (store arms) |
| (b) GPU perfection | Every GPU node in the pool available with working RDMA — regardless of the job's node count; any node with a Device-not-active history is healed or replaced first |
| (c) Sandbox pool | Capacity verified; image caches warmed identically on all sandbox nodes |
| (d) Config freeze | Frozen constraint set pinned in the YAML (step limit 25, max_tokens null, 128-wide admission, pod timeout 1500, wedge-fixed image, episode logging, trainer OOM fix) |

## S4 — In-run gates

Sentinel, streamer, error monitor, pod watcher, and full-metrics poller armed
against the *current* attempt, and re-armed after any restart. Store arms: at
T+5min, save_put climbing with zero Device errors; by_source shares sane. Any
tripwire pauses the ledger until the cause is diagnosed from raw logs.

## S5 — Post-run validity and discussion

A run counts only if: zero sandbox retry-exhaustions, counters drained, and
harvest completed before teardown (SUCCEEDED ≠ completed — delivered
trajectory count is the truth).

Differences between comparative store and recompute runs must be recorded and
discussed — unexplained deltas are implementation-error detectors, not noise
to average away.

## S6 — Pairing symmetry

Paired arms run back-to-back on identical nodes, identical image caches,
pristine store state (master restart between arms), same task batch and seed.
The first run on any new node, pool, or env regime is always a shakeout.

## S7 — Comparisons, terminology, docs

Valid runs only in comparison docs. A regime change (e.g. S1) resets the
matrix: old numbers stay as archived evidence, never in current tables.
Canonical terminology everywhere: step / sampling / rollout (= whole sampling
phase) / trajectory / turn / batch / generations. Reports follow the gdoc
template (TLDR-first, metric-definition tables, run-log links, lettered
analysis).

## S8 — Multi-step acceptance

Multi-step runs must record verl step metrics (METRICS-CATALOG §8) or they do
not count.
