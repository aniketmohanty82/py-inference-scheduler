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

from integration.slime.server import WorkerRegistry


def test_add_assigns_id_and_lists_url_and_id():
    reg = WorkerRegistry()
    wid = reg.add("http://127.0.0.1:1000")
    listed = reg.list()
    assert listed == [{"url": "http://127.0.0.1:1000", "id": wid}]


def test_add_is_idempotent_per_url():
    reg = WorkerRegistry()
    first = reg.add("http://127.0.0.1:1000")
    second = reg.add("http://127.0.0.1:1000")
    assert first == second
    assert len(reg.list()) == 1


def test_remove_by_id():
    reg = WorkerRegistry()
    wid = reg.add("http://127.0.0.1:1000")
    assert reg.remove(wid) is True
    assert reg.list() == []
    # removing again is a no-op
    assert reg.remove(wid) is False


def test_endpoints_carry_url_attribute():
    reg = WorkerRegistry()
    reg.add("http://127.0.0.1:1000")
    reg.add("http://127.0.0.1:2000")
    eps = reg.endpoints()
    assert len(eps) == 2
    assert {str(ep.attributes["url"]) for ep in eps} == {
        "http://127.0.0.1:1000",
        "http://127.0.0.1:2000",
    }
