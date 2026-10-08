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

from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer  # type: ignore[import-not-found]

_FILLER = " data"


@register("multiturn_load")
class MultiTurnLoadAgentLoop(AgentLoopBase):
    """Run a row's fixed turn schedule back to back, with no tool wait between turns.

    Turn i generates ``output_tokens[i]`` tokens (the rollout must run with
    ignore_eos so the model cannot end a turn early), then ``reply_tokens[i]``
    filler tokens stand in for the next user message and the next turn starts at
    once. Replies are raw token ids rather than a chat-templated message, so their
    length is exact. The schedule rides on the dataset row, so every arm runs
    identical work.
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:  # noqa: ANN401
        schedule = kwargs["extra_info"]
        output_tokens = [int(n) for n in schedule["output_tokens"]]
        reply_tokens = [int(n) for n in schedule["reply_tokens"]]
        messages = [dict(m) for m in kwargs["raw_prompt"]]
        prompt_ids = await self.loop.run_in_executor(None, lambda: self._prompt_ids(messages))
        filler_id = self.tokenizer.encode(_FILLER, add_special_tokens=False)[0]
        full_ids = list(prompt_ids)
        response_mask: list[int] = []
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        for turn, max_tokens in enumerate(output_tokens):
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_ids,
                    sampling_params={**sampling_params, "max_tokens": max_tokens},
                )
            token_ids = list(output.token_ids)
            full_ids += token_ids
            response_mask += [1] * len(token_ids)
            if turn < len(reply_tokens):
                full_ids += [filler_id] * reply_tokens[turn]
                response_mask += [0] * reply_tokens[turn]
        response_length = self.rollout_config.response_length
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=full_ids[len(prompt_ids) :][:response_length],
            response_mask=response_mask[:response_length],
            num_turns=len(output_tokens) + len(reply_tokens) + 1,
            metrics=metrics,
        )

    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        # Newer transformers return an encoding dict here, older ones the id list itself.
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return [int(token) for token in ids]
