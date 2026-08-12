#!/usr/bin/env bash
# Build + push the rllm-convergence training image.
# Usage: integration/rllm/image/build.sh [tag]   (default tag: dev)
set -euo pipefail

TAG=${1:-dev}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE=us-south1-docker.pkg.dev/aniket-gke-dev/llm-images/rllm-verl-mooncake:${TAG}

DOCKER_BUILDKIT=1 docker build \
    -f "$REPO_ROOT/integration/rllm/image/Dockerfile" \
    -t "$IMAGE" \
    "$REPO_ROOT"

docker push "$IMAGE"
echo "pushed $IMAGE"
