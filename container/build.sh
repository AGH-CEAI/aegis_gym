#!/bin/bash
# Usage: setup the following values and run the script
set -e

VERSION=${VERSION:-v0.0.1}
REGISTRY=${REGISTRY:-localhost}
PUSH=${PUSH:-false}

AEGIS_GYM_TAG=${AEGIS_GYM_TAG:-v202603091815}
CEAI_RSL_RL_TAG=${CEAI_RSL_RL_TAG:-v3.3.1}
AEGIS_ROS_TAG=${AEGIS_ROS_TAG:-v202603091730}

echo "> Building ${REGISTRY}/agh-ceai/aegis_gym:${VERSION}"

podman build \
  --target hardware_control \
  -t "${REGISTRY}/agh-ceai/aegis_gym:${VERSION}" \
  --build-arg AEGIS_ROS_TAG=${AEGIS_ROS_TAG} \
  --build-arg CEAI_RSL_RL_TAG=${CEAI_RSL_RL_TAG} \
  --build-arg AEGIS_GYM_TAG=${AEGIS_GYM_TAG} \
  .

echo "> Built: ${REGISTRY}/agh-ceai/aegis_gym:${VERSION}"

if [[ "${PUSH}" == "true" ]]; then
  podman push "${REGISTRY}/agh-ceai/aegis_gym:${VERSION}"
  echo "> Pushed to $REGISTRY"
fi
