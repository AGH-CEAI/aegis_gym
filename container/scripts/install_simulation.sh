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

# Anything already installed from source above, or by the torch tier, must not
# be re-installed from the lock, which pins the PyPI build of the same name:
#
#   torch/torchvision  the torch tier installs the CUDA wheel-index build; the
#                      lock pins the PyPI one, a different CUDA build with its
#                      own nvidia-* runtime libraries. Letting it through leaves
#                      a mismatched set (torch+cu128 against torchvision+cu129).
#   rsl-rl-lib         the AGH fork is installed from git above. It survives
#                      today only because its version equals the lock's pin; a
#                      bump on either side would swap in the PyPI build.
#
# So drop that family from the export and let the earlier installs own it;
# everything else stays lock-pinned. The list is derived from the lock rather
# than hard-coded, so it keeps up as the dependency set changes.
mapfile -t OWNED_ELSEWHERE < <(
    uv export \
        --no-emit-project \
        --extra sim-genesis \
        --extra test \
        --no-hashes \
        --no-annotate \
        --no-header \
        | sed -n 's/^\(nvidia-[a-z0-9.-]*\|torch\|torchvision\|triton\|rsl-rl-lib\)==.*/\1/p'
)

NO_EMIT=()
for pkg in ${OWNED_ELSEWHERE[@]+"${OWNED_ELSEWHERE[@]}"}; do
    NO_EMIT+=(--no-emit-package "${pkg}")
done

uv export \
    --no-emit-project \
    ${NO_EMIT[@]+"${NO_EMIT[@]}"} \
    --extra sim-genesis \
    --extra test \
    --output-file requirements.txt

uv pip install \
    --system \
    --requirement requirements.txt

mkdir -p /ws
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
