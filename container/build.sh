#!/bin/bash
# Usage: setup the following values and run the script

VERSION=${VERSION:-v1.0.0}
REGISTRY=${REGISTRY:-geonosis:5000}
PUSH=${PUSH:-false}

# TODO change default tags before merge
AEGIS_ROS_TAG=${AEGIS_ROS_TAG:-feature/aegis_grpc/simplified_cmake_build}
AEIGS_GYM_TAG=${AEIGS_GYM_TAG:-feature/migrate_poetry_to_uv}


# Validate registry (must be valid domain/IP:port)
if [[ ! "$REGISTRY" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}(:[0-9]+)?$|^[a-zA-Z0-9.-]+(:[0-9]+)?$ ]]; then
  echo "> Error: Invalid REGISTRY '$REGISTRY'. Use e.g. docker.io, localhost:5000, 192.168.0.100:5000"
  exit 1
fi

echo "> Building ${REGISTRY}/ceai/aegis_gym:${VERSION} variants"

podman build \
  --target simulation \
  -t "${REGISTRY}/ceai/aegis_gym:${VERSION}-simulator" \
  --build-arg AEIGS_GYM_TAG=${AEIGS_GYM_TAG} \
  .

podman build \
  --target hardware_control \
  -t "${REGISTRY}/ceai/aegis_gym:${VERSION}-hardware" \
  --build-arg AEGIS_ROS_TAG=${AEGIS_ROS_TAG} \
  --build-arg AEIGS_GYM_TAG=${AEIGS_GYM_TA} \
  .

echo "> Built: ${REGISTRY}/ceai/aegis_gym:${VERSION}-simulator and :${VERSION}-hardware"

# Optional: Push if flag set
if [[ "${}" == "true" ]]; then
  podman push "${REGISTRY}/ceai/aegis_gym:${VERSION}-simulator"
  podman push "${REGISTRY}/ceai/aegis_gym:${VERSION}-hardware"
  echo "> Pushed to $REGISTRY"
fi
