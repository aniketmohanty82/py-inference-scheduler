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

import asyncio
import logging
import os
import uuid

from ray.llm._internal.serve.core.configs.openai_api_models import (  # noqa: PLC2701
    CompletionRequest,
    ErrorResponse,
)
from ray.llm._internal.serve.core.ingress.ingress import (  # noqa: PLC2701
    DEFAULT_LLM_ROUTER_HTTP_TIMEOUT,
    OpenAiIngress,
    OpenAIHTTPException,
    router_request_timeout,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

ENABLE_ENV = "ENABLE_MOONCAKE_RELOCATION"
THRESHOLD_ENV = "MOONCAKE_RELOCATION_THRESHOLD"


# Wire protocol between ingress, router, and monitor - not operator tunables,
# so they live here as the single definition all three import.
# Pull requests carry this marker in their x-request-id; ensures the
# PushMonitor never pushes a pull.
PULL_MARKER = "rls-pull-"
PULL_OF_HEADER = "x-rls-pull-of"
TARGET_HEADER = "x-rls-target-replica"
PUSH_PULL_HEADER = "x-rls-push-pull"

_MONITOR_INTERVAL_S = 0.1


def relocation_enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1"


def relocation_threshold() -> int:
    return int(os.environ.get(THRESHOLD_ENV, "100"))


def should_push(prompt_tokens: int, generated_tokens: int, threshold: int) -> bool:
    """A request with no generated tokens has no decode KV worth moving."""
    if generated_tokens <= 0:
        return False
    return prompt_tokens + generated_tokens >= threshold


class PushMonitor:
    """Aborts any running request on this replica whose p+d crossed the
    threshold. The abort IS the push: the connector already streamed every
    full KV block to the store, and vllm finishes the request cleanly with
    finish_reason abort and its partial output."""

    def __init__(self, engine_client: object, threshold: int) -> None:
        self.engine_client = engine_client
        self.threshold = threshold
        self.pushed: set[str] = set()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(_MONITOR_INTERVAL_S)
            try:
                await self._tick()
            except Exception:  # noqa: BLE001
                # The monitor must never take the engine down with it.
                logger.warning("push monitor tick failed", exc_info=True)

    async def _tick(self) -> None:
        # vllm's live table of running requests on this engine
        states = self.engine_client.output_processor.request_states  # type: ignore[attr-defined]

        # drop our pushed set entries for requests that fully left the engine.
        # removing BEFORE the scan keeps just-aborted requests marked as pushed.
        current_ids = {s.external_req_id for s in states.values()}
        self.pushed &= current_ids

        # we await inside the loop, and the output processor
        # mutates request_states whenever the loop yields.
        for state in list(states.values()):
            # external_req_id is the id the caller supplied - our x-request-id
            req_id = state.external_req_id
            if PULL_MARKER in req_id or req_id in self.pushed:
                continue
            # accumulated ids; None for modes without one.
            detokenizer = state.detokenizer
            generated = (
                len(detokenizer.output_token_ids) if detokenizer is not None else 0
            )
            if should_push(state.prompt_len or 0, generated, self.threshold):
                # Mark before the await so a concurrent tick cannot double-abort.
                self.pushed.add(req_id)
                await self.engine_client.abort(req_id)  # type: ignore[attr-defined]


def merge_completion_responses(push_response: dict, pull_response: dict) -> dict:
    """One client response from the two halves: the pushed request's partial
    output followed by the pull's continuation. No recompute, just formatting
    the final object to look like what was requested."""
    push_choice = push_response["choices"][0]
    pull_choice = pull_response["choices"][0]

    choice = dict(pull_choice)
    choice["index"] = 0
    choice["text"] = (push_choice.get("text") or "") + (pull_choice.get("text") or "")
    if push_choice.get("token_ids") is not None:
        choice["token_ids"] = list(push_choice["token_ids"]) + list(
            pull_choice.get("token_ids") or []
        )

    merged = dict(pull_response)
    merged["id"] = push_response["id"]
    merged["created"] = push_response["created"]
    merged["choices"] = [choice]
    # The pull's prompt_tokens counts p+d1 (its resubmitted prompt); the client
    # sent only p and generated d1+d2.
    merged["prompt_token_ids"] = push_response.get("prompt_token_ids")
    push_usage = push_response.get("usage") or {}
    pull_usage = pull_response.get("usage") or {}
    if push_usage and pull_usage:
        prompt_tokens = push_usage["prompt_tokens"]
        completion_tokens = (
            push_usage["completion_tokens"] + pull_usage["completion_tokens"]
        )
        merged["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return merged


def _strip_token_ids(response: dict) -> None:
    response.pop("prompt_token_ids", None)
    for choice in response.get("choices", []):
        choice.pop("token_ids", None)
        choice.pop("prompt_token_ids", None)


def _with_headers(request: Request, extra: dict[str, str]) -> Request:
    # Same synthetic-request shape as RawRequestInfo.to_starlette_request.
    headers = {**dict(request.headers), **extra}
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


class RelocatingIngress(OpenAiIngress):
    """Completes relocated requests as one response: the PushMonitor aborts a
    request on its replica (push response), this ingress resubmits
    prompt_ids+token_ids routed away from that replica (pull request), and the
    client sees a single merged batch JSON."""

    async def completions(self, body: CompletionRequest, request: Request) -> Response:
        multi_prompt = (
            isinstance(body.prompt, list)
            and bool(body.prompt)
            and isinstance(body.prompt[0], (str, list))
        )
        if (
            not relocation_enabled()
            or bool(body.stream)
            or (body.n or 1) != 1
            or multi_prompt
        ):
            return await super().completions(body, request)

        client_wants_ids = bool(body.return_token_ids)
        push_id = uuid.uuid4().hex
        push_body = body.model_copy(update={"return_token_ids": True})

        async with router_request_timeout(DEFAULT_LLM_ROUTER_HTTP_TIMEOUT):
            push_response = await self._single_response(
                push_body, request, {"x-request-id": push_id}
            )
            push_choice = push_response["choices"][0]
            if push_choice.get("finish_reason") != "abort":
                if not client_wants_ids:
                    _strip_token_ids(push_response)
                return JSONResponse(content=push_response)

            prompt_ids = (
                push_response.get("prompt_token_ids")
                or push_choice.get("prompt_token_ids")
                or []
            )
            generated_ids = push_choice.get("token_ids") or []
            update: dict = {
                "prompt": list(prompt_ids) + list(generated_ids),
                "return_token_ids": True,
            }
            if body.max_tokens is not None:
                update["max_tokens"] = max(1, body.max_tokens - len(generated_ids))
            pull_body = body.model_copy(update=update)
            pull_id = f"{PULL_MARKER}{uuid.uuid4().hex}"
            pull_response = await self._single_response(
                pull_body,
                request,
                {"x-request-id": pull_id, PULL_OF_HEADER: push_id},
            )

            merged = merge_completion_responses(push_response, pull_response)
            if not client_wants_ids:
                _strip_token_ids(merged)
            return JSONResponse(
                content=merged,
                headers={PUSH_PULL_HEADER: f"{push_id},{pull_id}"},
            )

    async def _single_response(
        self, body: CompletionRequest, request: Request, extra_headers: dict[str, str]
    ) -> dict:
        """Non-streaming completions yield exactly one response object."""
        synthetic = _with_headers(request, extra_headers)
        async for response in self._get_response(
            body=body, call_method="completions", raw_request=synthetic
        ):
            if isinstance(response, ErrorResponse):
                raise OpenAIHTTPException(
                    message=response.error.message,
                    status_code=response.error.code,
                    type=response.error.type,
                )
            return response.model_dump()
        raise OpenAIHTTPException(
            message="empty response stream", status_code=500, type="InternalError"
        )
