"""Gate-2 smoke: verl-backend RL on r2egym tasks with gateway-path rollouts.

Rollouts must transit the Model Gateway (harness/agent-flow path), NOT verl's
token-in-token-out path: mini-swe-agent runs inside each task sandbox and calls
the gateway via OPENAI_API_BASE, so the gateway's sqlite trace store ends up
holding per-call token IDs - that is the Gate-2 observable.

Usage (from the rllm repo root, config overrides in train_smoke.sh):
    python /path/to/train_smoke.py rllm/backend=verl ...
"""

import os

import hydra
from omegaconf import DictConfig
from rllm.data.dataset import DatasetRegistry
from rllm.eval.agent_loader import load_agent
from rllm.trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset("r2egym_smoke", "train")
    if train_dataset is None:
        raise RuntimeError("r2egym_smoke dataset not found. Run: python prepare_r2egym_subset.py")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=load_agent("mini-swe-agent"),
        # evaluator stays None: SandboxTaskHooks auto-wires FromTaskEvaluation,
        # which runs each task's tests/test.sh inside the live sandbox for reward.
        evaluator=None,
        config=config,
        train_dataset=train_dataset,
        val_dataset=train_dataset,
        # kubernetes = task pods on the cluster (vendored backend); use docker
        # for a workstation run with a local daemon.
        sandbox_backend=os.environ.get("RLS_SANDBOX_BACKEND", "kubernetes"),
        # rllm caps sandboxed flows at 64 concurrent; pressure runs above 64
        # rollouts need this raised to match n_parallel_tasks.
        sandbox_concurrency=int(os.environ["RLS_SANDBOX_CONCURRENCY"])
        if os.environ.get("RLS_SANDBOX_CONCURRENCY")
        else None,
    )
    trainer.train()


if __name__ == "__main__":
    main()
