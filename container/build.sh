#!/bin/bash
# Usage: setup the following values and run the script
set -e

VERSION=${VERSION:-v1.0.0}
REGISTRY=${REGISTRY:-geonosis.local:5000}
PUSH=${PUSH:-false}

# TODO change default tags before merge
AEGIS_GYM_TAG=${AEGIS_GYM_TAG:-feature/learning_container}
CEAI_RSL_RL_TAG=${CEAI_RSL_RL_TAG:-v3.3.1}
AEGIS_ROS_TAG=${AEGIS_ROS_TAG:-feature/aegis_grpc/simplified_cmake_build}


# Validate registry (must be valid domain/IP:port)
if [[ ! "$REGISTRY" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}(:[0-9]+)?$|^[a-zA-Z0-9.-]+(:[0-9]+)?$ ]]; then
  echo "> Error: Invalid REGISTRY '$REGISTRY'. Use e.g. docker.io, localhost:5000, 192.168.0.100:5000"
  exit 1
fi

# echo "> Building ${REGISTRY}/agh-ceai/aegis_gym:${VERSION} variants"
echo "> Building ${REGISTRY}/agh-ceai/aegis_gym:${VERSION}"

# podman build \
#   --target simulation \
#   -t "${REGISTRY}/agh-ceai/aegis_gym:${VERSION}-simulator" \
#   --build-arg AEGIS_GYM_TAG=${AEGIS_GYM_TAG} \
#   --build-arg CEAI_RSL_RL_TAG=${CEAI_RSL_RL_TAG} \
#   --build-arg AEGIS_GYM_TAG=${AEGIS_GYM_TAG} \
#   .

podman build \
  --target hardware_control \
  -t "${REGISTRY}/agh-ceai/aegis_gym:${VERSION}" \
  --build-arg AEGIS_ROS_TAG=${AEGIS_ROS_TAG} \
  --build-arg CEAI_RSL_RL_TAG=${CEAI_RSL_RL_TAG} \
  --build-arg AEGIS_GYM_TAG=${AEGIS_GYM_TAG} \
  .

# echo "> Built: ${REGISTRY}/agh-ceai/aegis_gym:${VERSION}-simulator and :${VERSION}-hardware"
echo "> Built: ${REGISTRY}/agh-ceai/aegis_gym:${VERSION}"

# Optional: Push if flag set
if [[ "${PUSH}" == "true" ]]; then
  # podman push "${REGISTRY}/agh-ceai/aegis_gym:${VERSION}-simulator"
  # podman push "${REGISTRY}/agh-ceai/aegis_gym:${VERSION}-hardware"
  podman push "${REGISTRY}/agh-ceai/aegis_gym:${VERSION}"
  echo "> Pushed to $REGISTRY"
fi
