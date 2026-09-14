# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backfill W&B history from verl console logs.

verl's wandb history can be lost when the SDK->service channel dies during
long idle gaps between steps (observed on Ray + wandb-core 0.22: run exists,
config/stats synced, zero history rows). The console logger always has the
full per-step metrics, so this parses `step:N - key:value - ...` lines from a
Ray job log and (re)logs them into the W&B run.

Accepts .gz logs, and can create a run rather than resume one - a job run
with trainer.logger=["console"] has no W&B run to backfill into.

Usage (WANDB_API_KEY must be set, e.g. on the Ray head pod):
    python3 backfill_wandb.py --log_file job.log \
        --project swe-rl-scheduler --run_id tgpawp8d
    python3 backfill_wandb.py --log_file store_driver.log.gz \
        --project swe-rl-scheduler --run_name p34-store
    python3 backfill_wandb.py --log_file store_driver.log.gz --project x --dry_run

To push a complete pair - verl metrics AND the engine-scrape metrics, one run
per arm - backfill from the harvested CSV instead of the driver log:
    python3 backfill_wandb.py --csv metrics.csv --project swe-store-ab \
        --prefix p34-
"""

from __future__ import annotations

import argparse
import csv
import gzip
import pathlib
import re

STEP_LINE_RE = re.compile(r"\bstep:(\d+) - (.+)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LEADING_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def parse_step_lines(text: str) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for raw in text.splitlines():
        line = ANSI_RE.sub("", raw)
        m = STEP_LINE_RE.search(line)
        if not m:
            continue
        metrics: dict[str, float] = {}
        for pair in m.group(2).split(" - "):
            # partition, not rpartition: metric keys never contain a colon, but
            # Ray interleaves other actors' output onto the LAST field and that
            # junk does ("...throughput:316.1(vLLMHttpServer...) WARNING:...:
            # Flushed..."). Splitting on the last colon hands back " Flushed..."
            # as the value, so perf/throughput - always the final field -
            # silently vanishes from whichever steps got interleaved.
            key, sep, value = pair.partition(":")
            if not sep:
                continue
            hit = LEADING_NUM_RE.match(value.strip())
            if hit:
                metrics[key.strip()] = float(hit.group())
        if metrics:
            steps[int(m.group(1))] = metrics
    return steps


def report_coverage(steps: dict[int, dict[str, float]]) -> None:
    """Name any key missing from some steps, so a silent drop cannot recur."""
    all_keys = set().union(*(set(v) for v in steps.values()))
    for step in sorted(steps):
        missing = all_keys - set(steps[step])
        if missing:
            print(f"WARNING step {step}: missing {sorted(missing)}")
    print(f"parsed {len(steps)} steps, {len(all_keys)} distinct metrics")


def parse_metrics_csv(path):
    """Read a harvested metrics.csv into {arm: {step: {metric: value}}}.

    A driver log only carries verl's own step metrics. The engine-scrape
    metrics (kv usage, preemptions, the by_source split) exist solely in the
    harvested CSV, so backfilling from the CSV is what gets a complete run
    into W&B rather than the verl subset.
    """
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            metric = row["metric"]
            for col, raw in row.items():
                m = re.fullmatch(r"(.+)_s(\d+)", col or "")
                if not m or raw in ("", None):
                    continue
                out.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[metric] = float(raw)
    return out


def push(steps, *, project, entity, run_id, run_name):
    import wandb

    run = wandb.init(
        project=project, entity=entity, id=run_id, name=run_name,
        resume="allow" if run_id else None,
        settings=wandb.Settings(silent=True),
    )
    for step in sorted(steps):
        wandb.log(steps[step], step=step)
        print(f"  step {step}: {len(steps[step])} metrics")
    run.finish()
    print(f"  done -> {run.url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_file", help="verl driver log (.log or .log.gz)")
    parser.add_argument("--csv", help="harvested metrics.csv; pushes one run per arm")
    parser.add_argument("--project", required=True)
    parser.add_argument("--run_id", default=None, help="existing W&B run id to backfill into")
    # A run launched with trainer.logger=["console"] has no W&B run to resume,
    # so allow creating one from the log alone.
    parser.add_argument("--run_name", default=None, help="create a new run under this name instead")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--dry_run", action="store_true", help="parse and report, log nothing")
    parser.add_argument("--prefix", default="", help="prepended to each run name in --csv mode")
    args = parser.parse_args()
    if not args.log_file and not args.csv:
        raise SystemExit("pass --log_file or --csv")

    if args.csv:
        arms = parse_metrics_csv(args.csv)
        for arm, steps in sorted(arms.items()):
            print(f"{arm}:")
            report_coverage(steps)
            if not args.dry_run:
                push(steps, project=args.project, entity=args.entity,
                     run_id=None, run_name=f"{args.prefix}{arm}")
        return

    if not args.run_id and not args.run_name and not args.dry_run:
        raise SystemExit("pass --run_id to resume a run, or --run_name to create one")

    path = pathlib.Path(args.log_file)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace", encoding="utf-8") as f:
        steps = parse_step_lines(f.read())
    if not steps:
        raise SystemExit("no `step:N - k:v` lines found in log file")
    report_coverage(steps)
    if args.dry_run:
        return
    push(steps, project=args.project, entity=args.entity,
         run_id=args.run_id, run_name=args.run_name)


if __name__ == "__main__":
    main()
