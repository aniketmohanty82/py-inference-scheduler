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

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

KV_USAGE = "vllm:kv_cache_usage_perc"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
PREEMPTIONS = "vllm:num_preemptions_total"
PULLED = "vllm:prompt_tokens_by_source_total[external_kv_transfer]"


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each vLLM series over its labels, but keep ``source`` and ``reason`` apart."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        series, _, value = line.rpartition(" ")
        name, _, labels = series.partition("{")
        if name.endswith(("_bucket", "_created")):
            continue
        split = _label(labels, "source") or _label(labels, "reason")
        key = f"{name}[{split}]" if split else name
        try:
            values[key] = values.get(key, 0.0) + float(value)
        except ValueError:
            continue
    return values


def _label(labels: str, wanted: str) -> str:
    for pair in labels.rstrip("}").split(","):
        key, _, value = pair.partition("=")
        if key.strip() == wanted:
            return value.strip().strip('"')
    return ""


@dataclass
class EngineMetricsPoller:
    """Scrape every sampler's /metrics on a timer, so every arm is measured the same way."""

    addresses: list[str]
    interval_s: float = 2.0
    samples: list[tuple[float, str, dict[str, float]]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="engine-metrics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            row = {}
            for address in self.addresses:
                try:
                    with urllib.request.urlopen(f"http://{address}/metrics", timeout=5) as resp:
                        values = parse_metrics(resp.read().decode())
                except Exception as e:  # noqa: BLE001 - one bad scrape must not end the run
                    logger.warning("metrics scrape of %s failed: %s", address, e)
                    continue
                if not any(a == address for _, a, _ in self.samples):
                    # Once per sampler: which series it exports, so a missing gauge is visible.
                    print(f"OVERSCHED_KEYS {address} {json.dumps(sorted(values))}", flush=True)
                self.samples.append((now, address, values))
                # None, not 0, when a series is absent: a silent default once hid exactly that.
                row[address] = [
                    values.get(k) for k in (KV_USAGE, RUNNING, WAITING, PREEMPTIONS, PULLED)
                ]
            print(
                "OVERSCHED_SCRAPE "
                + json.dumps({"t": round(now, 2), "kv_run_wait_preempt_pulled": row}),
                flush=True,
            )
            self._stop.wait(self.interval_s)

    def window(self, start: float, end: float, pad: float) -> dict[str, dict[str, float]]:
        """Per sampler: gauges averaged over [start, end], counter deltas over the padded window.

        The pad brackets the window with one scrape on each side, so counters
        bumped by the first and last turns are inside the delta.
        """
        summary = {}
        for address in self.addresses:
            ours = [(t, v) for t, a, v in self.samples if a == address]
            points = [(t, v) for t, v in ours if start <= t <= end]
            padded = [(t, v) for t, v in ours if start - pad <= t <= end + pad]
            if len(points) < 2:  # noqa: PLR2004 - a time average needs two points
                continue
            span = points[-1][0] - points[0][0]
            kv = sorted(v.get(KV_USAGE, 0.0) for _, v in points)
            stats = {
                "kv_mean": _time_mean(points, KV_USAGE, span),
                # A batch rollout ramps up and drains; the median and p90 read the plateau.
                "kv_p50": kv[len(kv) // 2],
                "kv_p90": kv[int(0.9 * (len(kv) - 1))],
                "kv_max": kv[-1],
                "running_mean": _time_mean(points, RUNNING, span),
                "waiting_mean": _time_mean(points, WAITING, span),
                # Capacity = no KV blocks; deferred = held back by the step's token budget.
                "waiting_capacity_mean": _time_mean(points, f"{WAITING}_by_reason[capacity]", span),
                "waiting_deferred_mean": _time_mean(points, f"{WAITING}_by_reason[deferred]", span),
            }
            first, last = padded[0][1], padded[-1][1]
            for key, value in last.items():
                if key.endswith(("_total", "_count", "_sum")) or "_total[" in key:
                    stats[f"delta:{key}"] = value - first.get(key, 0.0)
            summary[address] = stats
        return summary


def _time_mean(points: list[tuple[float, dict[str, float]]], key: str, span: float) -> float:
    """Trapezoid time average of one gauge."""
    area = sum(
        (t1 - t0) * (v0.get(key, 0.0) + v1.get(key, 0.0)) / 2
        for (t0, v0), (t1, v1) in zip(points, points[1:])
    )
    return area / span if span > 0 else 0.0
