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

from scheduling.framework import CycleState, Endpoint, LLMRequest
from scheduling.plugins.scorers.prefix_plugin import (
    PrefixCacheScorer,
    PrefixIndexer,
    _hash_prompt_bytes,
)


def test_indexer_add_get_remove():
    idx = PrefixIndexer()
    hashes = [1, 2, 3]
    idx.add(hashes, "s1")

    # each hash should map to server s1
    for h in hashes:
        got = idx.get(h)
        assert "s1" in got

    assert "s1" in idx.pods()

    # remove and ensure gone
    idx.remove_server("s1")
    for h in hashes:
        assert idx.get(h) == set()
    assert idx.pods() == []


def test_indexer_reset_clears_all_mappings():
    idx = PrefixIndexer()
    # Two servers share hash=2 so we exercise both maps and the shared-entry path.
    idx.add([1, 2, 3], "s1")
    idx.add([2, 3, 4], "s2")
    assert set(idx.pods()) == {"s1", "s2"}
    assert idx.get(2) == {"s1", "s2"}

    idx.reset()

    assert idx.pods() == []
    for h in (1, 2, 3, 4):
        assert idx.get(h) == set()

    # After reset the indexer must still be usable.
    idx.add([5], "s3")
    assert idx.pods() == ["s3"]
    assert idx.get(5) == {"s3"}


def test_indexer_lru_refreshes_recency():
    """Re-adding a hash must refresh its recency so hot blocks survive eviction."""
    idx = PrefixIndexer(lru_capacity_per_server=2)
    idx.add([1, 2], "s1")
    idx.add([1], "s1")  # refresh 1: now 2 is the least recently used
    idx.add([3], "s1")  # evicts 2, not 1

    assert idx.get(1) == {"s1"}
    assert idx.get(2) == set()
    assert idx.get(3) == {"s1"}


def test_prefix_cache_scorer_reset_drops_routing_hints():
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10)
    body = "abcdefghijkl"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
    assert len(hashes) >= 1
    scorer.add_prefixes_for_server("ep1", hashes)

    endpoints = {"ep1": Endpoint(name="ep1"), "ep2": Endpoint(name="ep2")}

    # Pre-reset: ep1 has all the prefix hits, so it scores; ep2 does not.
    pre_scores = scorer.score(CycleState(), req, endpoints)
    assert "ep1" in pre_scores
    assert pre_scores["ep1"] > 0.0

    scorer.reset()

    # Post-reset: no cached prefixes -> no endpoint scores against the prefix
    # index. The "novel prompt" fallback then routes to the least-loaded
    # servers; with both at zero load, both are tied.
    post_scores = scorer.score(CycleState(), req, endpoints)
    assert set(post_scores.keys()) == {"ep1", "ep2"}
    assert scorer.indexer.pods() == []


def test_hash_prompt_bytes_basic():
    body = "abcdefgh"
    # block size 4 -> two blocks
    hashes = _hash_prompt_bytes("mymodel", body.encode("utf-8"), 4, 10)
    assert isinstance(hashes, list)
    assert len(hashes) == 2
    # hashes should be integers
    assert all(isinstance(h, int) for h in hashes)


def test_prefix_cache_scorer_scores():
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10)
    # prepare a request with 3 blocks
    body = "abcdefghijkl"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)

    # compute hashes using same logic
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
    assert len(hashes) == 3

    scorer.add_prefixes_for_server("ep1", hashes[:2])
    scorer.add_prefixes_for_server("ep2", [hashes[2]])

    endpoints = {
        "ep1": Endpoint(name="ep1"),
        "ep2": Endpoint(name="ep2"),
        "ep3": Endpoint(name="ep3"),
    }
    cs = CycleState()
    scores = scorer.score(cs, req, endpoints)

    # No block is universal, so all 3 count; scores are match fractions and the
    # never-indexed ep3 gets nothing.
    assert scores["ep1"] == 2.0 / 3.0
    assert scores["ep2"] == 1.0 / 3.0
    assert "ep3" not in scores


def test_partial_match_is_followed_not_dropped():
    """Any discriminative match is real conversation content: follow the
    largest holder rather than falling back to least-loaded."""
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=16)
    body = "0123456789abcdefghijklmnopqrstuvwxyzABCD"  # 10 blocks
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 16)
    assert len(hashes) == 10

    scorer.add_prefixes_for_server("warm", hashes[:4])
    endpoints = {"warm": Endpoint(name="warm"), "cold": Endpoint(name="cold")}
    scores = scorer.score(CycleState(), req, endpoints)
    assert scores["warm"] == 0.4
    assert "cold" not in scores


def test_above_half_match_keeps_custody():
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=16)
    body = "0123456789abcdefghijklmnopqrstuvwxyzABCD"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 16)

    # Blocks 0-1 are held by BOTH pods (universal -> ignored); custody is judged
    # on the remaining 8, of which home holds 4 (>= half).
    scorer.add_prefixes_for_server("home", hashes[:6])
    scorer.add_prefixes_for_server("warm", hashes[:2])
    endpoints = {"home": Endpoint(name="home"), "warm": Endpoint(name="warm")}
    scores = scorer.score(CycleState(), req, endpoints)
    assert scores["home"] == 0.5
    assert "warm" not in scores  # its only matches were the shared pedestal


def test_universal_blocks_are_discounted():
    """An engine holding only the fleet-wide pedestal must not clear the custody bar."""
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=16)
    body = "0123456789abcdefghijklmnopqrstuvwxyzABCD"  # 10 blocks
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 16)

    # All three engines hold the 8-block pedestal; home also holds the 2-block tail.
    for name in ("home", "e2", "e3"):
        scorer.add_prefixes_for_server(name, hashes[:8])
    scorer.add_prefixes_for_server("home", hashes[8:])

    endpoints = {name: Endpoint(name=name) for name in ("home", "e2", "e3")}
    scores = scorer.score(CycleState(), req, endpoints)

    # Discriminative blocks: the 2-block tail only; home holds 2/2.
    assert scores["home"] == 1.0
    assert "e2" not in scores
    assert "e3" not in scores
