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

import math
import random
from dataclasses import dataclass

# Common words that are single tokens in Qwen-family tokenizers; one string keeps them readable.
_WORDS = (  # noqa: SIM905
    "the of and to in is that for it as was with be by on not he this are or his from at "
    "which but have an they you were her she there been one all we their has would when "
    "who will more if no out so said what up its about into than them can only other new "
    "some could time these two may then do first any my now such like our over man me even "
    "most made after also did many before must through back years where much your way well "
    "down should because each just those people how too little state good very make world "
    "still own see men work long get here between both life being under never day same "
    "another know while last might us great old year off come since against go came right "
    "used take three"
).split()

# Spread of each tool call around the mean: lognormal, so most calls are quick and a few are slow.
_TOOL_LATENCY_SIGMA = 0.5


@dataclass(frozen=True)
class Workload:
    """A batch of identical-shape trajectories: unique prompts, fixed turn schedule.

    tool_latency_s is the mean time a tool call takes between turns; 0 means replies arrive at once.
    """

    trajectories: int
    prompt_words: int
    turns: int
    output_tokens: int
    reply_tokens: int
    seed: int = 0
    tool_latency_s: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a negative mean tool time."""
        if self.tool_latency_s < 0:
            raise ValueError("tool_latency_s must be >= 0")

    def rows(self) -> list[dict]:
        """Rows in verl's non-tensor batch layout; the same seed yields the same rows."""
        rng = random.Random(self.seed)  # noqa: S311 - reproducible load text, not secrets
        # Its own stream, so adding tool time leaves every prompt as it was.
        tool_rng = random.Random(f"tool-latency-{self.seed}")  # noqa: S311
        rows = []
        for i in range(self.trajectories):
            # The index leads the prompt so no two trajectories share a prefix past the template.
            text = f"Request {i}: " + " ".join(rng.choice(_WORDS) for _ in range(self.prompt_words))
            rows.append({
                "raw_prompt": [{"role": "user", "content": text}],
                "extra_info": {
                    "output_tokens": [self.output_tokens] * self.turns,
                    "reply_tokens": [self.reply_tokens] * (self.turns - 1),
                    "tool_latency_s": self._tool_latencies(tool_rng),
                },
                "agent_name": "multiturn_load",
                "index": i,
            })
        return rows

    def _tool_latencies(self, rng: random.Random) -> list[float]:
        if self.tool_latency_s == 0:
            return [0.0] * (self.turns - 1)
        mu = math.log(self.tool_latency_s) - _TOOL_LATENCY_SIGMA**2 / 2
        return [rng.lognormvariate(mu, _TOOL_LATENCY_SIGMA) for _ in range(self.turns - 1)]

    def max_context_tokens(self, template_tokens: int = 64) -> int:
        """Upper bound on a trajectory's final context, for sizing pools and max lengths.

        Only the prompt goes through the chat template; turns and replies are exact.
        """
        replies = (self.turns - 1) * self.reply_tokens
        return self.prompt_words + template_tokens + self.turns * self.output_tokens + replies
