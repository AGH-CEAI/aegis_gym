#!/usr/bin/env bash
set -euxo pipefail

: "${AEGIS_ROS_TAG:?AEGIS_ROS_TAG is required}"

export DEBIAN_FRONTEND=noninteractive

apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgrpc++-dev \
        libprotobuf-dev \
        libprotoc-dev \
        protobuf-compiler \
        protobuf-compiler-grpc \
        python3-grpc-tools \
        python3-grpcio \
        python3-protobuf

git clone \
    --depth 1 \
    --branch "${AEGIS_ROS_TAG}" \
    https://github.com/AGH-CEAI/aegis_ros.git \
    /tmp/aegis_ros

cd /tmp/aegis_ros/aegis_grpc

bash ./install_client.sh

cd /ws

rm -rf /tmp/aegis_ros

apt-get remove -y \
    build-essential \
    cmake \
    libgrpc++-dev \
    libprotobuf-dev \
    libprotoc-dev \
    protobuf-compiler \
    protobuf-compiler-grpc

apt-get autoremove -y
apt-get clean

uv cache clean

rm -rf \
    /var/lib/apt/lists/* \
    /root/.cache/pip \
    /root/.cache/uv \
    /tmp/* \
    /var/tmp/*
