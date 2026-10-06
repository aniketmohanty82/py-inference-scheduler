"""VERIFY-ONLY: deploy the repo's Ray Serve router app, send one burst, print a summary."""

import asyncio
import glob
import random
import time

import aiohttp
import ray
from ray import serve

ray.init(address="auto")
from integration.rayserve import router  # noqa: E402  builds the app with the verify config

serve.run(router.app, blocking=False)
deadline = time.time() + 1800
while time.time() < deadline:
    apps = serve.status().applications
    if apps and all(app.status == "RUNNING" for app in apps.values()):
        break
    time.sleep(10)
print("SERVE STATUS", {name: app.status for name, app in serve.status().applications.items()}, flush=True)

SENTENCE = "The quick brown fox jumps over the lazy dog while the scheduler counts tokens. "
REQUESTS = 192


async def one(session: aiohttp.ClientSession, i: int, repeat: int) -> tuple[bool, float, object]:
    body = {
        "model": "qwen-0.5b",
        "messages": [{"role": "user", "content": f"Request {i}. Summarize this text: " + SENTENCE * repeat}],
        "max_tokens": 128,
        "temperature": 0,
    }
    started = time.time()
    try:
        async with session.post(
            "http://127.0.0.1:8000/v1/chat/completions", json=body, timeout=aiohttp.ClientTimeout(total=900)
        ) as resp:
            data = await resp.json(content_type=None)
            return resp.status == 200 and bool(data.get("choices")), time.time() - started, resp.status
    except Exception as e:  # noqa: BLE001
        return False, time.time() - started, repr(e)[:160]


async def burst() -> list[tuple[bool, float, object]]:
    rng = random.Random(7)
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[one(session, i, rng.randint(10, 160)) for i in range(REQUESTS)])


started = time.time()
results = asyncio.run(burst())
wall = time.time() - started
latencies = sorted(r[1] for r in results if r[0])
ok = len(latencies)
print(
    f"BURST requests={REQUESTS} ok={ok} failed={REQUESTS - ok} wall={wall:.1f}s "
    f"p50={latencies[ok // 2]:.2f}s p90={latencies[int(0.9 * ok)]:.2f}s max={latencies[-1]:.2f}s"
    if ok else f"BURST requests={REQUESTS} ok=0",
    flush=True,
)
print("FAILURES", [r[2] for r in results if not r[0]][:5], flush=True)
time.sleep(15)
logs = [p for p in glob.glob("/tmp/ray/session_latest/logs/**/*", recursive=True) if p.endswith((".log", ".out", ".err"))]
print("log files scanned:", len(logs), flush=True)
markers = {"Selected endpoint": 0, "ROUTER ERROR": 0, "ROUTER WARNING": 0, "METRICS ERROR": 0, "Traceback": 0}
for path in logs:
    for line in open(path, errors="replace"):
        for marker in markers:
            if marker in line:
                markers[marker] += 1
                if marker != "Selected endpoint" and markers[marker] <= 3:
                    print("LOG", path.rsplit("/", 1)[-1], line.strip()[:300], flush=True)
print("MARKERS", markers, flush=True)
try:
    reports = open("/tmp/verify_rs.log").read().splitlines()
except FileNotFoundError:
    reports = []
print("VERIFY_RS reports:", len(reports), flush=True)
for line in reports[-4:]:
    print("LAST", line[:600], flush=True)
