#!/usr/bin/env bash
# Build + push the verl-native SWE A/B image.
# Usage: integration/verl/image/build.sh [tag]   (default: swe0)
set -euo pipefail

TAG=${1:-swe0}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE=us-south1-docker.pkg.dev/aniket-gke-dev/llm-images/rllm-verl-mooncake:${TAG}

# Per-tag Dockerfiles are the norm here (Dockerfile.swe5 ... Dockerfile.swe13);
# fall back to the unsuffixed one for the original build.
DOCKERFILE="$REPO_ROOT/integration/verl/image/Dockerfile.${TAG}"
[ -f "$DOCKERFILE" ] || DOCKERFILE="$REPO_ROOT/integration/verl/image/Dockerfile"
echo "building $IMAGE from $(basename "$DOCKERFILE")"

DOCKER_BUILDKIT=1 docker build \
    -f "$DOCKERFILE" \
    -t "$IMAGE" \
    "$REPO_ROOT"

docker push "$IMAGE"
echo "pushed $IMAGE"
