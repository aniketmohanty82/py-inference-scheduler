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

import inspect

import pytest

pytest.importorskip("ray.llm._internal.serve.core.ingress.builder")

from ray.llm._internal.serve.core.ingress.builder import LLMServingArgs  # noqa: E402

# Keep in sync with the bind call in build_custom_openai_app; that builder
# copies ray's private build_openai_app, so a ray upgrade can change the
# ingress contract under us (2.52 -> 2.56 added model_cards).
BOUND_KWARGS = {"llm_deployments", "model_cards", "lora_paths"}


def _ingress_signature():
    args = LLMServingArgs.model_validate(
        {"llm_configs": [{"model_loading_config": {"model_id": "m", "model_source": "s"}}]}
    )
    cfg = args.ingress_cls_config
    return inspect.signature(cfg.ingress_cls.__init__), set(cfg.ingress_extra_kwargs)


def test_builder_provides_every_required_ingress_arg():
    # bind() is lazy: a missing required arg only surfaces at replica init on
    # the cluster, so pin the contract here instead.
    sig, extra = _ingress_signature()
    required = {
        n
        for n, p in sig.parameters.items()
        if n != "self"
        and p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    provided = BOUND_KWARGS | extra
    assert required <= provided, f"ray ingress now requires {required - provided}"


def test_builder_passes_no_unknown_ingress_args():
    sig, extra = _ingress_signature()
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    unknown = (BOUND_KWARGS | extra) - set(sig.parameters)
    assert has_var_kw or not unknown, f"ray ingress no longer accepts {unknown}"
