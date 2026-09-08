# Image for the verl-native SWE store-vs-recompute A/B.
#
# Built on OUR mooncake base rather than PR #62's verlai/verl:vllm011 base:
# the store stack is the irreplaceable half, and vLLM 0.22.1 is what carries
# the nested mooncake store connector our DecodeKVSavingConnector subclasses
# plus the prompt_tokens_by_source counters the comparison is measured on.
#
# Phase-0 finding (probed against a live pod, PINS.md): verl 0.8.0 - already
# in this base via rllm[verl] - fully satisfies PR #62's SWEAgentLoop
# contract, so no verl upgrade and no vLLM downgrade are needed. The
# "verl 0.8.x untested" caveat applies to their routing hook, which this A/B
# does not use.
#
# Build:  integration/verl/image/build.sh <tag>
FROM us-south1-docker.pkg.dev/aniket-gke-dev/llm-images/rllm-verl-mooncake@sha256:0266993e90f84845b2891b87ac3a21c4faa024f7bd597b722084cc1ac358c6b0

USER root

# The SWE loop drives sandboxes over the k8s API and reads parquet datasets.
RUN pip install --no-cache-dir kubernetes pandas pyarrow datasets

COPY . /opt/py-inference-scheduler
RUN pip install --no-cache-dir -e /opt/py-inference-scheduler

# integration/ is repo-root code, not part of the installed package, and the
# SWE loop is addressed by dotted path (integration.verl.swe...) from both the
# agent-loop config and the trainer. Put the repo on sys.path for every
# process in the image rather than per-manifest env.
ENV PYTHONPATH=/opt/py-inference-scheduler

# Fail the build, not a GPU run, if any contract the A/B depends on is absent:
# the store connector (arm B), verl's agent-loop surface (the harness), and
# the vendored SWE modules that bind them together.
RUN python - <<'PYEOF'
import importlib, sys

for mod in (
    "vllm",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data",
    "py_inference_scheduler.datalayer.connectors.mooncake.decode_save",
    "verl",
    "verl.experimental.agent_loop",
    "integration.verl.swe.sandbox",
    "integration.verl.swe.grader",
    "integration.verl.swe.scaffold",
    "kubernetes",
):
    importlib.import_module(mod)

import vllm, verl
from verl.experimental.agent_loop import AgentLoopBase, AgentLoopOutput
from py_inference_scheduler.datalayer.connectors.mooncake.decode_save import (
    DecodeKVSavingConnector,
)

# The loop returns these; a missing field means a silent contract break.
required = {"reward_score", "extra_fields"}
missing = required - set(getattr(AgentLoopOutput, "model_fields", {}))
if missing:
    sys.exit(f"AgentLoopOutput missing fields: {missing}")

print(f"GATE OK  vllm={vllm.__version__}  verl={verl.__version__}")
PYEOF

WORKDIR /opt/py-inference-scheduler
