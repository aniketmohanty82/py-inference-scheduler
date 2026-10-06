"""VERIFY-ONLY: deploy the repo's Ray Serve router app on 32B, send one burst per phase, print a summary."""

import asyncio
import glob
import random
import time

import aiohttp
import ray
from ray import serve

ray.init(address="auto")
from integration.rayserve import router  # noqa: E402  builds the app with the verify config

SENTENCE = "The quick brown fox jumps over the lazy dog while the scheduler counts tokens. "
REQUESTS = 192


def deploy(app: object) -> None:
    started = time.time()
    serve.run(app, blocking=False)
    while time.time() - started < 3600:
        apps = serve.status().applications
        if apps and all(a.status == "RUNNING" for a in apps.values()):
            break
        time.sleep(15)
    statuses = {name: a.status for name, a in serve.status().applications.items()}
    print(f"SERVE STATUS {statuses} after {time.time() - started:.0f}s", flush=True)


async def one(session: aiohttp.ClientSession, i: int, repeat: int) -> tuple[bool, float, object]:
    body = {
        "model": "qwen-32b",
        "messages": [{"role": "user", "content": f"Request {i}. Summarize this text: " + SENTENCE * repeat}],
        "max_tokens": 128,
        "temperature": 0,
    }
    started = time.time()
    try:
        async with session.post(
            "http://127.0.0.1:8000/v1/chat/completions", json=body, timeout=aiohttp.ClientTimeout(total=1800)
        ) as resp:
            data = await resp.json(content_type=None)
            return resp.status == 200 and bool(data.get("choices")), time.time() - started, resp.status
    except Exception as e:  # noqa: BLE001
        return False, time.time() - started, repr(e)[:160]


async def burst() -> list[tuple[bool, float, object]]:
    rng = random.Random(7)
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[one(session, i, rng.randint(40, 400)) for i in range(REQUESTS)])


def markers() -> dict[str, int]:
    counts = {"Selected endpoint": 0, "ROUTER ERROR": 0, "ROUTER WARNING": 0, "METRICS ERROR": 0, "Traceback": 0}
    logs = glob.glob("/tmp/ray/session_latest/logs/**/*", recursive=True)
    for path in (p for p in logs if p.endswith((".log", ".out", ".err"))):
        for line in open(path, errors="replace"):
            for marker in counts:
                if marker in line:
                    counts[marker] += 1
    return counts


def last_report() -> str:
    try:
        lines = open("/tmp/verify_rs.log").read().splitlines()
    except FileNotFoundError:
        return "none"
    return lines[-1] if lines else "none"


def phase(name: str, app: object) -> None:
    deploy(app)
    before = markers()
    print(f"PHASE {name} counters before: {last_report()[:500]}", flush=True)
    started = time.time()
    results = asyncio.run(burst())
    wall = time.time() - started
    time.sleep(20)
    after = markers()
    latencies = sorted(r[1] for r in results if r[0])
    ok = len(latencies)
    summary = (
        f"p50={latencies[ok // 2]:.2f}s p90={latencies[int(0.9 * ok)]:.2f}s max={latencies[-1]:.2f}s" if ok else ""
    )
    print(f"PHASE {name} BURST requests={REQUESTS} ok={ok} failed={REQUESTS - ok} wall={wall:.1f}s {summary}", flush=True)
    print(f"PHASE {name} failures: {[r[2] for r in results if not r[0]][:5]}", flush=True)
    print(f"PHASE {name} new log markers: {{{', '.join(f'{k}: {after[k] - before[k]}' for k in after)}}}", flush=True)
    print(f"PHASE {name} counters after: {last_report()[:500]}", flush=True)


phase("A default replica concurrency", router.app)
capped = router.llm_config.model_copy(deep=True)
capped.deployment_config["max_ongoing_requests"] = 2
phase("B replicas capped at 2 in flight", router.build_custom_openai_app({"llm_configs": [capped]}))
