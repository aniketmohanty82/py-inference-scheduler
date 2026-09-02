#!/usr/bin/env bash
# Build + push the rllm-convergence training image.
# Usage: integration/rllm/image/build.sh [tag] [task_limit]
#   tag        image tag (default: dev)
#   task_limit baked r2egym subset size (default: Dockerfile's 128)
set -euo pipefail

TAG=${1:-dev}
TASK_LIMIT=${2:-}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE=us-south1-docker.pkg.dev/aniket-gke-dev/llm-images/rllm-verl-mooncake:${TAG}

DOCKER_BUILDKIT=1 docker build \
    -f "$REPO_ROOT/integration/rllm/image/Dockerfile" \
    ${TASK_LIMIT:+--build-arg TASK_LIMIT=$TASK_LIMIT} \
    -t "$IMAGE" \
    "$REPO_ROOT"

docker push "$IMAGE"
echo "pushed $IMAGE"
