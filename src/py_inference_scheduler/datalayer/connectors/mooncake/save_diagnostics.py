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

# Save-path evidence for RLPullPolicyConnector, kept free of vLLM imports so it is testable with
# fakes. Every printer catches everything: a diagnostic must never take an engine down.

from __future__ import annotations

import os
import time
from collections import Counter


def track_save_jobs(sched, scheduler_output, meta, born: dict[int, float]) -> None:
    """Stamp new store jobs' births in ``born``; print SAVEJOBS for a step that saves or preempts.

    The line splits the step's new jobs by what made the request save: new (first prefill), resumed
    (re-prefill after a preemption, which upstream saves from token 0 again) or running (a later
    chunk). preempted counts the requests this step's schedule preempted; vLLM's preemption metric
    only counts a preemption once its request emits output.
    """
    try:
        pinned = getattr(sched, "_pinned_saves", None) or {}
        for job in [job for job in born if job not in pinned]:
            del born[job]
        now = time.monotonic()
        jobs = [
            m
            for m in getattr(meta, "requests", ())
            if getattr(m, "store_job_id", None) in pinned and m.store_job_id not in born
        ]
        for m in jobs:
            born[m.store_job_id] = now
        preempted = len(scheduler_output.preempted_req_ids or ())
        if not jobs and not preempted:
            return
        cached = scheduler_output.scheduled_cached_reqs
        new_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        resumed_ids = set(getattr(cached, "resumed_req_ids", None) or ())
        requests = getattr(sched, "_unfinished_requests", None) or {}
        origins = Counter(_origin(m.req_id, new_ids, resumed_ids, requests) for m in jobs)
        blocks = {block for m in jobs for block in pinned[m.store_job_id][0]}
        print(
            f"SAVEJOBS t={time.time():.1f} pid={os.getpid()} emitted={len(jobs)} "
            f"new={origins['new']} resumed={origins['resumed']} running={origins['running']} "
            f"blocks={len(blocks)} preempted={preempted} "
            f"scheduled={len(new_ids) + len(cached.req_ids)}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"SAVEJOBS unavailable: {e!r}", flush=True)


def _origin(req_id: str, new_ids: set[str], resumed_ids: set[str], requests) -> str:
    # The v1 model runner lists a resumed request as cached with a resumed flag, the v2 runner as
    # new; a newly scheduled request that was ever preempted is re-prefilling either way.
    if req_id in resumed_ids:
        return "resumed"
    if req_id not in new_ids:
        return "running"
    request = (requests.get(req_id) or (None,))[0]
    return "resumed" if getattr(request, "num_preemptions", 0) else "new"


def print_pins(sched, born: dict[int, float]) -> None:
    """Print one PINS line: in-flight saves, the blocks they pin, the free pool and turnover.

    A stuck job keeps ids' minimum while next advances; oldest_s is the longest-pinned job's age.
    waiting_on maps ranks-still-to-report to jobs, so a rank that lags shows as jobs waiting on
    fewer ranks than the engine has.
    """
    try:
        pinned = getattr(sched, "_pinned_saves", None) or {}
        blocks = {block for ids, _ in pinned.values() for block in ids}
        pool = getattr(sched, "_gpu_block_pool", None)
        free = pool.get_num_free_blocks() if pool is not None else -1
        total = getattr(pool, "num_gpu_blocks", -1)
        now = time.monotonic()
        oldest = max((now - born[job] for job in pinned if job in born), default=0.0)
        waiting = Counter(remaining for _, remaining in pinned.values())
        print(
            f"PINS t={time.time():.1f} pid={os.getpid()} jobs={len(pinned)} "
            f"blocks={len(blocks)} free={free} total={total} "
            f"ids={min(pinned, default=-1)}-{max(pinned, default=-1)} "
            f"next={getattr(sched, '_next_store_job_id', -1)} oldest_s={oldest:.1f} "
            f"waiting_on={_counts(waiting)}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"PINS unavailable: {e!r}", flush=True)


def _counts(counter: Counter[int]) -> str:
    return ",".join(f"{key}:{n}" for key, n in sorted(counter.items())) or "-"


def install_save_timer(thread) -> bool:
    """Count and time the save jobs a send thread finishes, for print_send_state.

    The thread looks its handler up per job, so the wrapper applies from the next job on.
    """
    if thread is None or hasattr(thread, "rls_save_stats"):
        return False
    handle = thread._handle_request
    stats = {"finished": 0, "slowest_s": 0.0, "started": 0.0}

    def timed(req_meta) -> None:
        stats["started"] = time.monotonic()
        try:
            handle(req_meta)
        finally:
            stats["slowest_s"] = max(stats["slowest_s"], time.monotonic() - stats["started"])
            stats["started"] = 0.0
            stats["finished"] += 1

    thread.rls_save_stats = stats
    thread._handle_request = timed
    return True


def print_send_state(thread) -> None:
    """Print one SENDQ line per rank: queued, live and unreported saves, health and progress.

    Progress comes from install_save_timer: finished counts every job the rank has finished,
    slowest_s is its longest job since the last line, busy_s how long the job it runs now has run.
    Reads without the thread's lock: a send thread stuck while holding it must not stall the
    engine through its own diagnostic.
    """
    try:
        if thread is None:
            return
        live = [job for jobs in list(thread.stored_requests.values()) for job in list(jobs)]
        stats = getattr(thread, "rls_save_stats", None) or {}
        started = stats.get("started", 0.0)
        busy = time.monotonic() - started if started else 0.0
        slowest = stats.get("slowest_s", 0.0)
        stats["slowest_s"] = 0.0
        print(
            f"SENDQ t={time.time():.1f} pid={os.getpid()} rank={thread.tp_rank} "
            f"queued={thread.request_queue.qsize()} live={len(live)} "
            f"unreported={len(thread._completed_saves)} "
            f"pressure={getattr(thread, '_store_pressure_active', None)} alive={thread.is_alive()} "
            f"finished={stats.get('finished', -1)} slowest_s={slowest:.2f} busy_s={busy:.1f} "
            f"live_ids={min(live, default=-1)}-{max(live, default=-1)}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"SENDQ unavailable: {e!r}", flush=True)
