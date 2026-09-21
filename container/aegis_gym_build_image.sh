#!/bin/bash
# Script generated with Claude Opus 5

set -euo pipefail

# Resolve through symlinks so the build context is the repository, not $PWD.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

TORCH_IMAGE="ceai/aegis_gym_torch"
BASE_IMAGE="ceai/aegis_gym"
TORCH_CONTAINERFILE="${SCRIPT_DIR}/Containerfile.torch"

AEGIS_REPO_URL="https://github.com/AGH-CEAI/aegis_gym.git"
AEGIS_ROS_REPO_URL="https://github.com/AGH-CEAI/aegis_ros.git"
DEFAULT_REGISTRY="geonosis:5000"

IMAGE_VERSION="v0.1.0"
AEGIS_GYM_TAG="devel"
CEAI_RSL_RL_TAG="v3.3.2"
AEGIS_ROS_TAG="humble-devel"

UBUNTU_VERSION="22.04"
CUDA_SHORT_VERSION="cu129"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"

ASSUME_YES=0
NO_CACHE=0
TORCH_ONLY=0
REBUILD_TORCH=0
DO_PUSH=0
REGISTRY=""

usage() {
    cat << 'EOF'
Usage: aegis_gym_build_image [options]

Builds the two base image tiers:
  ceai/aegis_gym_torch:<ver>   ubuntu + torch/torchvision (CUDA wheels)
  ceai/aegis_gym:<ver>         graphics stack, rsl_rl, aegis_gym deps, gRPC client

The torch tier is kept separate so a --no-cache rebuild of the dependency
tier does not redo the multi-GB CUDA wheel install.

Options:
  -v, --version VER      Image version (default: v0.1.0)
  -y, --yes              Accept all defaults, no prompts (for CI)
      --gym-ref REF      aegis_gym branch/tag for the dependency lock (default: devel)
      --rsl-ref REF      AGH-CEAI/rsl_rl tag (default: v3.3.2)
      --ros-ref REF      aegis_ros branch/tag for the gRPC client (default: humble-devel)
      --no-cache         Build the dependency tier ignoring the layer cache
      --rebuild-torch    Rebuild the torch tier from scratch (implies --no-cache
                         for that tier) even when it already exists
      --torch-only       Build the torch tier and stop
  -p, --push[=HOST]      Push both tiers (default registry: geonosis:5000)
  -h, --help             This message

Examples:
  aegis_gym_build_image
  aegis_gym_build_image -y -v latest
  aegis_gym_build_image --no-cache --gym-ref feature/my-branch
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v | --version)   IMAGE_VERSION="${2:?-v/--version requires a value}"; shift ;;
        -y | --yes)       ASSUME_YES=1 ;;
        --gym-ref)        AEGIS_GYM_TAG="${2:?--gym-ref requires a value}"; shift ;;
        --rsl-ref)        CEAI_RSL_RL_TAG="${2:?--rsl-ref requires a value}"; shift ;;
        --ros-ref)        AEGIS_ROS_TAG="${2:?--ros-ref requires a value}"; shift ;;
        --no-cache)       NO_CACHE=1 ;;
        --rebuild-torch)  REBUILD_TORCH=1 ;;
        --torch-only)     TORCH_ONLY=1 ;;
        -p | --push)      DO_PUSH=1 ;;
        --push=*)         DO_PUSH=1; REGISTRY="${1#*=}" ;;
        -h | --help)      usage; exit 0 ;;
        *) echo ">>> Unknown option '$1'. See --help." >&2; exit 1 ;;
    esac
    shift
done

command -v podman > /dev/null 2>&1 || {
    echo ">>> Error: 'podman' not found in PATH." >&2
    exit 1
}

# podman prefers Containerfile over Dockerfile; accept whichever is present.
CONTAINERFILE=""
for candidate in "${SCRIPT_DIR}/Containerfile" "${SCRIPT_DIR}/Dockerfile"; do
    if [[ -f "${candidate}" ]]; then
        CONTAINERFILE="${candidate}"
        break
    fi
done

if [[ -z "${CONTAINERFILE}" ]]; then
    echo ">>> Error: no Containerfile or Dockerfile in ${SCRIPT_DIR}." >&2
    exit 1
fi

[[ -f "${TORCH_CONTAINERFILE}" ]] || {
    echo ">>> Error: ${TORCH_CONTAINERFILE} not found." >&2
    exit 1
}

# --- Settings --------------------------------------------------------------

echo ">>> BUILDING THE ${BASE_IMAGE} base images"

if ((ASSUME_YES == 0)); then
    read -r -p ">>> Image version [${IMAGE_VERSION}]: " reply
    IMAGE_VERSION="${reply:-${IMAGE_VERSION}}"

    read -r -p ">>> aegis_gym ref (locks the dependencies) [${AEGIS_GYM_TAG}]: " reply
    AEGIS_GYM_TAG="${reply:-${AEGIS_GYM_TAG}}"

    read -r -p ">>> rsl_rl tag [${CEAI_RSL_RL_TAG}]: " reply
    CEAI_RSL_RL_TAG="${reply:-${CEAI_RSL_RL_TAG}}"

    read -r -p ">>> aegis_ros ref [${AEGIS_ROS_TAG}]: " reply
    AEGIS_ROS_TAG="${reply:-${AEGIS_ROS_TAG}}"
fi

TORCH_REF="${TORCH_IMAGE}:${IMAGE_VERSION}"
BASE_REF="${BASE_IMAGE}:${IMAGE_VERSION}"

# --- Torch tier ------------------------------------------------------------

build_torch() {
    # $1 = 1 to ignore the layer cache. Nothing in this tier's inputs changes
    # between builds -- the versions are build args with fixed defaults -- so a
    # cached rebuild is a no-op that re-tags the identical image. Asking for a
    # rebuild of a tier that already exists therefore has to mean --no-cache,
    # otherwise the flag cannot do the one job it exists for: redoing a CUDA
    # wheel install that came down corrupt or half-finished.
    local no_cache="${1:-0}"
    local build_cmd=(podman build "${SCRIPT_DIR}"
        --file "${TORCH_CONTAINERFILE}"
        --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}"
        --build-arg "CUDA_SHORT_VERSION=${CUDA_SHORT_VERSION}"
        --build-arg "TORCH_VERSION=${TORCH_VERSION}"
        --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}"
        -t "${TORCH_REF}")
    ((no_cache)) && build_cmd+=(--no-cache)

    echo ">>> Building ${TORCH_REF} (torch ${TORCH_VERSION}, ${CUDA_SHORT_VERSION}$(
        ((no_cache)) && echo ", no cache"
    ))..."
    "${build_cmd[@]}"
    echo ">>> Built ${TORCH_REF}"
}

if podman image exists "${TORCH_REF}"; then
    echo ">>> ${TORCH_REF} already exists."
    if ((REBUILD_TORCH)); then
        build_torch 1
    elif ((ASSUME_YES == 0)); then
        # Default No: this is the expensive layer.
        read -r -p ">>> Rebuild it? (y/N): " reply
        case "$(aegis_answer "${reply}" n)" in
            y) build_torch 1 ;;
            *) echo ">>> Keeping the existing torch image." ;;
        esac
    fi
else
    build_torch "${NO_CACHE}"
fi

if ((TORCH_ONLY)); then
    echo ">>> --torch-only given, stopping here."
    exit 0
fi

# --- Dependency tier -------------------------------------------------------

# Resolve each ref to a commit so a layer is rebuilt only when its branch has
# actually moved. Falls back to a timestamp, which always busts the cache.
resolve_rev() {
    # $1 = repository URL, $2 = ref. Only the bare revision goes to stdout.
    local rev
    if ! rev="$(aegis_resolve_rev "$1" "$2")"; then
        echo ">>> Warning: could not resolve '$2' on the remote," \
            "disabling layer cache." >&2
        rev="$(date +%s)"
    fi
    echo "${rev}"
}

REV="$(resolve_rev "${AEGIS_REPO_URL}" "${AEGIS_GYM_TAG}")"
ROS_REV="$(resolve_rev "${AEGIS_ROS_REPO_URL}" "${AEGIS_ROS_TAG}")"

BUILD_CMD=(podman build "${SCRIPT_DIR}"
    --file "${CONTAINERFILE}"
    --build-arg "TORCH_REF=${TORCH_REF}"
    --build-arg "AEGIS_GYM_TAG=${AEGIS_GYM_TAG}"
    --build-arg "AEGIS_GYM_REV=${REV}"
    --build-arg "CEAI_RSL_RL_TAG=${CEAI_RSL_RL_TAG}"
    --build-arg "AEGIS_ROS_TAG=${AEGIS_ROS_TAG}"
    --build-arg "AEGIS_ROS_REV=${ROS_REV}"
    --build-arg "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    -t "${BASE_REF}")
((NO_CACHE)) && BUILD_CMD+=(--no-cache)

echo ">>> Building ${BASE_REF} from ${TORCH_REF} (${AEGIS_GYM_TAG} @ ${REV:0:8})..."
"${BUILD_CMD[@]}"
echo ">>> Built ${BASE_REF}"

# --- Push ------------------------------------------------------------------

if ((DO_PUSH == 0 && ASSUME_YES == 0)); then
    read -r -p ">>> Push the images to a registry? (y/N): " reply
    case "$(aegis_answer "${reply}" n)" in
        y) DO_PUSH=1 ;;
    esac
fi

((DO_PUSH)) || exit 0

if [[ -z "${REGISTRY}" ]]; then
    if ((ASSUME_YES)); then
        REGISTRY="${DEFAULT_REGISTRY}"
    else
        read -r -p ">>> Registry [${DEFAULT_REGISTRY}]: " reply
        REGISTRY="${reply:-${DEFAULT_REGISTRY}}"
    fi
fi

for ref in "${TORCH_REF}" "${BASE_REF}"; do
    remote="${REGISTRY}/${ref}"
    echo ">>> Tagging as ${remote}..."
    podman tag "${ref}" "${remote}"
    echo ">>> Pushing ${remote}..."
    podman push "${remote}"
    echo ">>> Pushed ${remote}"
done
