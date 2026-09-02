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
    kubernetes \
    cupy-cuda12x

# Bake the frozen r2egym subset + registry into the image: the trainer actor
# resolves verifiers from task_path on WHATEVER pod it lands on, so the task
# dirs must exist at the same absolute path cluster-wide. TASK_LIMIT sets the
# subset size - sampling enumerates the WHOLE registered dataset regardless of
# train_batch_size, so this build arg is the only knob that sets a run's task
# count (128 = pressure cells, 8 = cheap smoke).
ARG TASK_LIMIT=128
COPY integration/rllm/smoke/prepare_r2egym_subset.py /tmp/prepare_r2egym_subset.py
ENV RLLM_HOME=/opt/rllm-home
RUN python /tmp/prepare_r2egym_subset.py --limit ${TASK_LIMIT} --out-dir /opt/rllm-home/tasks/r2egym_smoke \
    && chmod -R a+rwX /opt/rllm-home

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
    "kubernetes",
    "rllm.sandbox.backends.kubernetes_backend",
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

# Wedge fix (diagnosed 08-31 via traceback + py-spy): a request preempted
# while its async store-save is still queued has fewer physical blocks than
# the save expects; prepare_value then raises IndexError, run()'s catch
# skipped the dec_stored_request/task_done completion, and the engine held
# the request forever (both replicas deadlock). A failed save must complete
# the request exactly like the skip paths do - the store is a cache.
# Also log the full traceback. Upstream fix belongs in vLLM's mooncake
# store connector; this patches our image until that lands.
RUN python - << 'PYEOF'
import pathlib, vllm
p = pathlib.Path(vllm.__file__).parent / "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
src = p.read_text()
old = '''            except Exception as e:
                logger.error("Error in %s: %s", self.name, e)'''
new = '''            except Exception as e:
                logger.exception("Error in %s: %s", self.name, e)
                try:
                    req_id = getattr(request_data, "req_id", None)
                    if req_id is not None:
                        if hasattr(self, "dec_stored_request"):
                            self.dec_stored_request(req_id)
                        self.request_queue.task_done()
                except Exception:
                    logger.exception(
                        "Failed to release request after error in %s", self.name
                    )'''
assert old in src, "transfer-thread catch site not found - vllm layout changed"
p.write_text(src.replace(old, new))
print("patched", p)
PYEOF

# No-artificial-cap mode: rllm's OpenAI chat path always sends max_tokens
# and never clamps to the remaining context, so any flat value either
# truncates turns or 400s deep prompts (prompt + max_tokens > window).
# Teach it that max_tokens=None means "omit the field" - vLLM then fills
# in exactly the remaining window per request. Enabled by setting
# rllm.rollout.{train,val}.max_tokens=null in the job config.
RUN python - << 'PYEOF'
import pathlib, rllm
p = pathlib.Path(rllm.__file__).parent / "engine/rollout/openai_engine.py"
src = p.read_text()
old = '''        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))
'''
new = '''        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))
        if max_tokens is None:
            return {}
'''
assert old in src, "max_tokens preparation site not found - rllm layout changed"
p.write_text(src.replace(old, new))
print("patched", p)
PYEOF

# Step-capped trajectories: without a step limit the only trajectory bound
# is cli_harness's 1800s kill, so tail sessions cost 30 min in EVERY arm
# and serving speed never reaches step wall-time. RLLM_MSWEA_STEP_LIMIT=N
# appends '-c mini.yaml -c agent.step_limit=N' (multiple -c specs merge
# recursively; including the default file avoids the replace-config trap).
RUN python - << 'PYEOF'
import pathlib, rllm
p = pathlib.Path(rllm.__file__).parent / "harnesses/mini_swe_agent.py"
src = p.read_text()
old = "import shlex\n"
new = "import os\nimport shlex\n"
assert old in src and "import os" not in src, "import site changed"
src = src.replace(old, new, 1)
old = '''            f"--exit-immediately "
            f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
'''
new = '''            f"--exit-immediately "
            + (
                f"-c mini.yaml -c agent.step_limit={int(os.environ['RLLM_MSWEA_STEP_LIMIT'])} "
                if os.environ.get("RLLM_MSWEA_STEP_LIMIT")
                else ""
            )
            + f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
'''
assert old in src, "invocation site changed"
p.write_text(src.replace(old, new))
import py_compile
py_compile.compile(str(p), doraise=True)
print("patched + compiled", p)
PYEOF
