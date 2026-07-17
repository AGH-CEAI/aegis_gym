#!/usr/bin/env bash
set -euxo pipefail

: "${CEAI_RSL_RL_TAG:?CEAI_RSL_RL_TAG is required}"
: "${AEGIS_GYM_TAG:?AEGIS_GYM_TAG is required}"

export DEBIAN_FRONTEND=noninteractive

apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl-mesa0 \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libgl1-mesa-glx \
        libgles2 \
        libglvnd-dev \
        libglvnd0 \
        libglx0 \
        libxrender1 \
        libvulkan-dev \
        libvulkan1 \
        libglib2.0-0 \
        mesa-utils \
        mesa-vulkan-drivers \
        vulkan-tools \
        xvfb

git clone \
    --depth 1 \
    --branch "${CEAI_RSL_RL_TAG}" \
    https://github.com/AGH-CEAI/rsl_rl.git \
    /tmp/rsl_rl

uv pip install --system /tmp/rsl_rl

git clone \
    --depth 1 \
    --branch "${AEGIS_GYM_TAG}" \
    https://github.com/AGH-CEAI/aegis_gym.git \
    /tmp/aegis_gym

cd /tmp/aegis_gym

uv export \
    --extra sim-genesis \
    --extra test \
    --output-file requirements.txt

uv pip install \
    --system \
    --requirement requirements.txt

cd /ws

rm -rf \
    /tmp/rsl_rl \
    /tmp/aegis_gym

apt-get autoremove -y
apt-get clean

uv cache clean

rm -rf \
    /var/lib/apt/lists/* \
    /root/.cache/pip \
    /root/.cache/uv \
    /tmp/* \
    /var/tmp/*
