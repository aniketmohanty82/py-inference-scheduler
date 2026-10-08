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

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from integration.verl.overscheduling.engine_metrics import EngineMetricsPoller
from integration.verl.overscheduling.workload import Workload

AGENT_LOOP_CONFIG = "integration/verl/overscheduling/agent_loop.yaml"
# Chat template plus the "Request i:" lead, on top of one token per prompt word.
PROMPT_SLACK_TOKENS = 256
MANAGERS = {
    "stock": ("verl.experimental.agent_loop.agent_loop", "AgentLoopManager"),
    "overscheduling": ("integration.verl.verl_hook", "PyInferenceAgentLoopManager"),
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse driver flags; anything else goes to verl's config as Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Rollout-only multi-turn load on standalone verl samplers."
    )
    parser.add_argument("--arm", choices=tuple(MANAGERS), required=True)
    parser.add_argument("--samplers", type=int, required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=2)
    parser.add_argument(
        "--trajectories", default="128", help="comma-separated batch sizes, run in order"
    )
    parser.add_argument("--prompt-words", type=int, default=2000)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--reply-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmup", type=int, default=1, help="unmeasured rollouts first, at the first batch size"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-interval", type=float, default=2.0)
    return parser.parse_known_args()


def _workload(args: argparse.Namespace, trajectories: int, repeat: int = 0) -> Workload:
    return Workload(
        trajectories,
        args.prompt_words,
        args.turns,
        args.output_tokens,
        args.reply_tokens,
        args.seed + repeat,
    )


def _compose(args: argparse.Namespace, overrides: list[str]) -> Any:  # noqa: ANN401
    """Build verl's trainer config with the rollout set up for standalone samplers and this load."""
    import verl  # type: ignore[import-not-found]
    from hydra import compose, initialize_config_dir  # type: ignore[import-not-found]
    from omegaconf import open_dict  # type: ignore[import-not-found]

    world = args.samplers * args.tp
    if world % args.nnodes:
        raise ValueError(
            f"{args.samplers} samplers x tp {args.tp} do not split evenly over {args.nnodes} nodes"
        )
    shape = _workload(args, 1)
    config_dir = Path(verl.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    with open_dict(config):
        config.data.max_prompt_length = shape.prompt_words + PROMPT_SLACK_TOKENS
        config.data.max_response_length = shape.max_context_tokens() - shape.prompt_words
        rollout = config.actor_rollout_ref.rollout
        rollout.name = "vllm"
        rollout.mode = "async"
        rollout.tensor_model_parallel_size = args.tp
        rollout.nnodes = args.nnodes
        # Each sampler still gets tp GPUs on one node: verl sizes a replica as min(this, tp).
        rollout.n_gpus_per_node = world // args.nnodes
        # No trainer syncs weights into standalone samplers, so they load from disk.
        rollout.load_format = "auto"
        rollout.skip_tokenizer_init = False
        rollout.disable_log_stats = False
        # Fixed turn lengths in every arm; the hook reads this flag too, so it lives in config.
        rollout.ignore_eos = True
        rollout.agent.agent_loop_config_path = AGENT_LOOP_CONFIG
        rollout.agent.default_agent_loop = "multiturn_load"
    return config


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    # Element by element: np.array would turn equal-length message lists into a 2-D array.
    columns = {}
    for key in rows[0]:
        column: np.ndarray = np.empty(len(rows), dtype=object)
        for i, row in enumerate(rows):
            column[i] = row[key]
        columns[key] = column
    return columns


def _rollout(
    loops: Any,  # noqa: ANN401 - a verl AgentLoopManager; type unavailable off-cluster
    poller: EngineMetricsPoller,
    args: argparse.Namespace,
    workload: Workload,
    repeat: int,
) -> dict[str, Any]:
    """Run the batch once and return its record: wall time, efficiency, per-sampler stats."""
    from verl.protocol import DataProto  # type: ignore[import-not-found]

    batch = DataProto(
        non_tensor_batch=_columns(workload.rows()),
        meta_info={"global_steps": repeat, "validate": False},
    )
    start = time.time()
    output = loops.generate_sequences(batch)
    end = time.time()
    # One more scrape after the last turn lands, so its counters are in the deltas.
    time.sleep(args.metrics_interval * 1.5)
    gpus = args.samplers * args.tp
    rollout_s = end - start
    generated = int(output.batch["response_mask"].sum().item())
    prompt_width = output.batch["prompts"].shape[1]
    prompt_tokens = output.batch["attention_mask"][:, :prompt_width].sum(-1).float()
    return {
        "arm": args.arm,
        "samplers": args.samplers,
        "gpus": gpus,
        "trajectories": workload.trajectories,
        "repeat": repeat,
        "workload": {
            "prompt_words": workload.prompt_words,
            "turns": workload.turns,
            "output_tokens": workload.output_tokens,
            "reply_tokens": workload.reply_tokens,
            "seed": workload.seed,
        },
        "start": start,
        "end": end,
        "rollout_s": rollout_s,
        "samples_per_s_per_gpu": workload.trajectories / rollout_s / gpus,
        "prompt_tokens_mean": float(prompt_tokens.mean().item()),
        "generated_tokens": generated,
        "generated_tokens_per_s_per_gpu": generated / rollout_s / gpus,
        "verl_timing": {k: float(v) for k, v in output.meta_info.get("timing", {}).items()},
        "samplers_stats": poller.window(start, end, pad=args.metrics_interval * 1.5),
    }


def main() -> None:
    args, overrides = _parse_args()
    module, name = MANAGERS[args.arm]
    # Imported before any sampler starts, as the trainer does: the hook patches verl's server class.
    manager_cls = getattr(importlib.import_module(module), name)
    import ray
    from verl.workers.rollout.llm_server import LLMServerManager  # type: ignore[import-not-found]

    config = _compose(args, overrides)
    ray.init()
    servers = LLMServerManager.create(config, worker_group=None)
    addresses = servers.get_addresses()
    print("OVERSCHED_SAMPLERS " + json.dumps(addresses), flush=True)
    loops = manager_cls.create(config, llm_client=servers.get_client())
    poller = EngineMetricsPoller(addresses, args.metrics_interval)
    poller.start()
    sizes = [int(n) for n in args.trajectories.split(",")]
    # A first rollout pays one-time compile and kernel-tuning stalls; measure only after that.
    # Warm-ups use their own seeds, so no measured rollout repeats their prompts.
    runs = [("OVERSCHED_WARMUP", sizes[0], -1 - w) for w in range(args.warmup)]
    runs += [("OVERSCHED_RESULT", size, repeat) for size in sizes for repeat in range(args.repeats)]
    try:
        for tag, trajectories, repeat in runs:
            # No rollout may reuse KV an earlier one computed: drop local and Mooncake caches,
            # and give each repeat its own prompts in case a store entry survives the drop.
            ray.get([handle.clear_kv_cache.remote() for handle in servers.server_handles])
            record = _rollout(loops, poller, args, _workload(args, trajectories, repeat), repeat)
            print(f"{tag} {json.dumps(record)}", flush=True)
    finally:
        poller.stop()


if __name__ == "__main__":
    main()
