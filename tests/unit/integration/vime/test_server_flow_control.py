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

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("httpx")  # FastAPI's TestClient is built on httpx
from fastapi.testclient import TestClient

from integration.vime.server import create_app
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig


def _scheduler():
    config = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "backpressure": {
                "flow_control": {
                    "type": "simple_backpressure",
                    "kv_threshold": 0.95,
                    "waiting_threshold": 2,
                },
                "scorers": [{"type": "waiting_queue", "weight": 1.0}],
                "picker": {"type": "max_score"},
            }
        },
    }
    return Scheduler.new_with_config(SchedulerConfig.from_dict(config))


def _make_handler(worker_id, stats):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/metrics":
                body = (
                    "# TYPE vllm:num_requests_running gauge\n"
                    'vllm:num_requests_running{model_name="m"} 0.0\n'
                    "# TYPE vllm:num_requests_waiting gauge\n"
                    f'vllm:num_requests_waiting{{model_name="m"}} {stats["waiting"]}\n'
                    "# TYPE vllm:kv_cache_usage_perc gauge\n"
                    f'vllm:kv_cache_usage_perc{{model_name="m"}} {stats["kv"]}\n'
                ).encode()
                self._send(200, body, "text/plain")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            if self.path == "/inference/v1/generate":
                body = json.dumps({"worker": worker_id}).encode()
                self._send(200, body, "application/json")
            else:
                self._send(404, b"", "text/plain")

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class MutableStubWorker:
    """Engine whose /metrics render from a dict the test can flip mid-flight."""

    def __init__(self, worker_id):
        self.stats = {"kv": 0.1, "waiting": 0}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(worker_id, self.stats))
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _client():
    return TestClient(create_app(_scheduler(), flow_poll_interval_s=0.05))


def test_healthy_engine_proxies_straight_through():
    with MutableStubWorker("a") as worker, _client() as client:
        client.post("/workers", json={"url": worker.url})
        resp = client.post("/inference/v1/generate", json={"token_ids": [1, 2, 3]})
        assert resp.status_code == 200
        assert resp.json()["worker"] == "a"


def test_saturated_engine_queues_until_scrape_recovers():
    """vime has no background poller: the flow-control watcher scrapes itself."""
    with MutableStubWorker("a") as worker, _client() as client:
        worker.stats["waiting"] = 9  # saturated: the pre-admit scrape sees this
        client.post("/workers", json={"url": worker.url})
        flow_control = client.app.state.flow_control

        result = {}

        def issue():
            result["resp"] = client.post(
                "/inference/v1/generate", json={"token_ids": [1, 2, 3]}
            )

        caller = threading.Thread(target=issue, daemon=True)
        caller.start()
        assert _wait_for(lambda: flow_control.queue_depth() == 1)
        time.sleep(0.2)
        assert "resp" not in result  # still parked while saturated

        worker.stats["waiting"] = 0  # capacity returns; the watcher's scrape re-admits
        caller.join(timeout=3.0)
        assert not caller.is_alive()
        assert result["resp"].status_code == 200
        assert result["resp"].json()["worker"] == "a"
        assert flow_control.queue_depth() == 0
