# Training image for the rllm-convergence 3-arm A/B:
#   ray-llm-mooncake base (ray 2.56 + vllm + mooncake wheels)
#   + rllm @ pinned SHA with the vendored gateway routing-policy patch
#   + verl 0.8.0 / vllm 0.22.1 via rllm[verl] (uv overrides mandatory, see overrides.txt)
#   + cupy (separated-mode nccl checkpoint engine) + docker-py (DockerSandbox)
#   + py-inference-scheduler (scheduler + Mooncake connectors + gateway policy)
#
# Build from the repo root:  integration/rllm/image/build.sh <tag>

FROM us-south1-docker.pkg.dev/aniket-gke-dev/llm-images/ray-llm-mooncake:2.56.0

ARG RLLM_SHA=1d1109a655e291b3001d8526d7c9ecc5b9328226

USER root
RUN pip install --no-cache-dir uv

# flash-attn (pulled in by rllm[verl]) has no cu130/torch2.11 wheel and its sdist
# doesn't declare torch as a build dep: pre-build it alone against the image's
# torch with isolation off (sm90-only for H200, bounded parallelism), so the
# main install below keeps normal build isolation for everything else. Kept
# ahead of the rllm clone so patch iterations reuse this ~40min layer.
RUN uv pip install --system --no-cache ninja packaging
ENV MAX_JOBS=16 \
    FLASH_ATTN_CUDA_ARCHS=90
RUN uv pip install --system --no-cache --no-build-isolation flash-attn==2.8.3

RUN git clone https://github.com/rllm-org/rllm.git /opt/rllm \
    && git -C /opt/rllm checkout ${RLLM_SHA}

COPY integration/rllm/patches/rllm-gateway-routing-policy.diff /tmp/rllm-gateway-routing-policy.diff
RUN git -C /opt/rllm apply /tmp/rllm-gateway-routing-policy.diff

COPY integration/rllm/image/overrides.txt /tmp/overrides.txt
RUN uv pip install --system --no-cache \
    --override /tmp/overrides.txt \
    -e "/opt/rllm[verl]" \
    docker \
    cupy-cuda12x

COPY . /opt/py-inference-scheduler
RUN uv pip install --system --no-cache -e /opt/py-inference-scheduler

# Fail the build if the base image's Mooncake/vLLM integration got clobbered
# or the pinned stack does not co-import.
RUN python - <<'EOF'
import importlib
for mod in (
    "vllm",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector",
    "verl",
    "rllm",
    "rllm_model_gateway",
    "py_inference_scheduler",
    "py_inference_scheduler.datalayer.connectors.mooncake.decode_save",
    "py_inference_scheduler.datalayer.metrics.poller",
    "docker",
    "cupy",
):
    importlib.import_module(mod)
import vllm
print("vllm", vllm.__version__)
EOF
# pip check with the known-acceptable residue filtered out:
#   - numpy<2 vs opencv is upstream rllm's own supported config (their uv
#     override pins numpy>=1.26 to satisfy verl; vllm's image paths unused here)
#   - nixl-cu12 / platformdirs are pre-existing base-image quirks
RUN pip check | grep -v -E \
    "opencv-python-headless .* numpy>=2|nixl .* nixl-cu12|python-discovery .* platformdirs" \
    > /tmp/pip-check-residue.txt || true; \
    if [ -s /tmp/pip-check-residue.txt ]; then cat /tmp/pip-check-residue.txt; exit 1; fi
