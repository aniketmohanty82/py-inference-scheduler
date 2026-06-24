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

"""Launch the vime external router.

    python -m integration.vime --host 0.0.0.0 --port 8000 --config scheduler.yaml

Then point vime at it with ``--vllm-router-ip <host> --vllm-router-port 8000``.

Reuses slime's engine-neutral ``run`` launcher with vime's ``create_app``.
"""

from __future__ import annotations

from integration.slime.__main__ import run
from integration.vime.server import create_app


def main() -> None:
    run(create_app, description="sampling router for vime", framework="vime")


if __name__ == "__main__":
    main()
