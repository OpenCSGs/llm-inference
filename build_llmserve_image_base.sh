#!/bin/bash
set -euxo pipefail

DOCKER_REPO="${LLMSERVE_DOCKER_REPO:-sean/llmray}"
VERSION="0.0.2"
DOCKER_TAG="$DOCKER_REPO:base-$VERSION"
DOCKER_FILE="${LLMSERVE_DOCKER_FILE:-deploy/ray/Dockerfile-base}"

sudo docker build . -f $DOCKER_FILE -t $DOCKER_TAG

# sudo docker build . -f $DOCKER_FILE -t $DOCKER_TAG -t $DOCKER_REPO:latest
# sudo docker push "$DOCKER_TAG"
