#!/usr/bin/env bash
# Build + push the verl-native SWE A/B image.
# Usage: integration/verl/image/build.sh [tag]   (default: swe0)
set -euo pipefail

TAG=${1:-swe0}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE=us-south1-docker.pkg.dev/aniket-gke-dev/llm-images/rllm-verl-mooncake:${TAG}

DOCKER_BUILDKIT=1 docker build \
    -f "$REPO_ROOT/integration/verl/image/Dockerfile" \
    -t "$IMAGE" \
    "$REPO_ROOT"

docker push "$IMAGE"
echo "pushed $IMAGE"
