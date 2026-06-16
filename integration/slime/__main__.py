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

"""Launch the slime external router.

    python -m integration.slime --host 0.0.0.0 --port 8000 --config scheduler.yaml

Then point slime at it with ``--sglang-router-ip <host> --sglang-router-port 8000``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import uvicorn
import yaml

from integration.slime.server import create_app
from scheduling import Scheduler
from scheduling.core.config import SchedulerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="py-inference-scheduler router for slime")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument("--config", required=True, help="path to scheduler.yaml")
    parser.add_argument("--log-level", default="info", help="uvicorn/log level")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    with pathlib.Path(args.config).open(encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = SchedulerConfig.from_dict(config_dict)
    logging.getLogger(__name__).info("Loaded scheduler config: %s", config)

    app = create_app(Scheduler.new_with_config(config))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
