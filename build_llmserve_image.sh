#!/bin/bash
set -euxo pipefail

#git diff-index --quiet HEAD --
GIT_COMMIT=`git rev-parse HEAD | cut -b 1-12`

DOCKER_REPO="${LLMSERVE_DOCKER_REPO:-sean/llmray}"
VERSION="0.0.2"
DOCKER_TAG="$DOCKER_REPO:$VERSION-$GIT_COMMIT"
DOCKER_FILE="${LLMSERVE_DOCKER_FILE:-deploy/ray/Dockerfile}"

./build_llmserve_wheel.sh

sudo docker build . -f $DOCKER_FILE -t $DOCKER_TAG

# sudo docker build . -f $DOCKER_FILE -t $DOCKER_TAG -t $DOCKER_REPO:latest
# sudo docker push "$DOCKER_TAG"
