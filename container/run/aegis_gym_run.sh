#!/bin/bash
# Script generated with Claude Opus 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
CONTAINERFILE="${SCRIPT_DIR}/Containerfile.prod"

BASE_IMAGE="ceai/aegis_gym"
PROD_IMAGE="ceai/aegis_gym_prod"
FALLBACK_BRANCH="devel"
AEGIS_REPO_URL="https://github.com/AGH-CEAI/aegis_gym.git"
AEGIS_REPO_NAME="aegis_gym"
DEFAULT_REGISTRY="geonosis:5000"

IMAGE_VERSION="v0.1.0"
CONTAINER_NAME="aegis_gym_prod"
AEGIS_GYM_TAG=""
CLEARML_CONF="${HOME}/clearml.conf"
SHM_SIZE=""
DO_BUILD=0
FORCE_BUILD=0
WITH_GUI=0
WITH_CLEARML=1
GPU_MODE="auto"
DRY_RUN=0
DO_PUSH=0
PUSH_AS=""
DO_RUN=1
ASSUME_YES=0
REGISTRY=""
ARGS=()

usage() {
    cat << 'EOF'
Usage: aegis_gym_run [options] [command [arguments]]

Options:
  -b, --build            Build the production image before running
  -B, --rebuild          Build ignoring the layer cache
  -v, --version VER      Image version (default: v0.1.0)
  -r, --ref REF          aegis_gym branch/tag/commit (default: detected, else devel)
  -n, --name NAME        Container name (default: aegis_gym_prod)
  -p, --push[=HOST]      Push the image (default registry: geonosis:5000)
      --push-as REF      Push under a fully specified ref, e.g.
                         ghcr.io/agh-ceai/aegis_gym:v0.0.2
  -y, --yes              Skip the confirmation prompt
      --gpu MODE         nvidia | none | auto (default: auto)
      --gui              Forward X11 so the Genesis viewer can open
                         (default: headless, which is what training wants)
      --clearml PATH     ClearML config to mount (default: ~/clearml.conf)
      --no-clearml       Do not mount a ClearML config
      --shm-size SIZE    Use a private IPC namespace with this /dev/shm size
                         instead of sharing the host's (podman rejects both)
      --no-run           Build and/or push only, do not start the container
      --dry-run          Print the podman command instead of running it
  -h, --help             This message

The command is passed to the in-image dispatcher:
  train [args]   python3 /opt/aegis_gym/train.py [args]
  eval  [args]   python3 /opt/aegis_gym/eval.py  [args]
  hpo   [args]   python3 /opt/aegis_gym/hpo.py   [args]
  shell [cmd]    a bare shell (default when no command is given)

Examples:
  aegis_gym_run train -a=rl -e REACHER_TRAIN --num-envs 4096
  aegis_gym_run --gui eval -e REACHER_TRAIN
  aegis_gym_run shell
  aegis_gym_run -B --no-run --push-as ghcr.io/agh-ceai/aegis_gym:v0.0.2
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b | --build)   DO_BUILD=1 ;;
        -B | --rebuild) DO_BUILD=1; FORCE_BUILD=1 ;;
        -v | --version) IMAGE_VERSION="${2:?-v/--version requires a value}"; shift ;;
        -r | --ref)     AEGIS_GYM_TAG="${2:?-r/--ref requires a value}"; shift ;;
        -n | --name)    CONTAINER_NAME="${2:?-n/--name requires a value}"; shift ;;
        -y | --yes)     ASSUME_YES=1 ;;
        -p | --push)    DO_PUSH=1 ;;
        --push=*)       DO_PUSH=1; REGISTRY="${1#*=}" ;;
        --push-as)      DO_PUSH=1; PUSH_AS="${2:?--push-as requires a value}"; shift ;;
        --gpu)          GPU_MODE="${2:?--gpu requires a mode: nvidia|none|auto}"; shift ;;
        --gui)          WITH_GUI=1 ;;
        --clearml)      CLEARML_CONF="${2:?--clearml requires a path}"; shift ;;
        --no-clearml)   WITH_CLEARML=0 ;;
        --shm-size)     SHM_SIZE="${2:?--shm-size requires a value}"; shift ;;
        --no-run)       DO_RUN=0 ;;
        --dry-run)      DRY_RUN=1 ;;
        -h | --help)    usage; exit 0 ;;
        --)             shift; ARGS+=("$@"); break ;;
        -*)             echo ">>> Unknown option '$1'. See --help." >&2; exit 1 ;;
        # The first non-option token is the dispatcher command; everything
        # after it belongs to that command and is forwarded untouched, so its
        # own flags (-a=rl, -e NAME, --num-envs) are never parsed here.
        *)              ARGS+=("$@"); break ;;
    esac
    shift
done

command -v podman > /dev/null 2>&1 || {
    echo ">>> Error: 'podman' not found in PATH." >&2
    exit 1
}

PROD_REF="${PROD_IMAGE}:${IMAGE_VERSION}"

# --- Colours ---------------------------------------------------------------

# Only colourise when stdout is a terminal, so redirected output and logs
# stay free of escape sequences. NO_COLOR is honoured by convention.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_YELLOW=$'\033[1;33m'
    C_RED=$'\033[1;31m'
    C_RESET=$'\033[0m'
else
    C_YELLOW=""
    C_RED=""
    C_RESET=""
fi

warn() { echo "${C_YELLOW}>>> $*${C_RESET}"; }
err() { echo "${C_RED}>>> $*${C_RESET}" >&2; }

# --- Helpers ---------------------------------------------------------------

in_aegis_repo() {
    # True when $PWD is inside an aegis_gym checkout (repo root or any
    # subdirectory). Being in some other git repo does not count.
    local toplevel url
    toplevel="$(git rev-parse --show-toplevel 2> /dev/null || true)"
    [[ -n "${toplevel}" ]] || return 1
    [[ "$(basename "${toplevel}")" == "${AEGIS_REPO_NAME}" ]] && return 0
    url="$(git -C "${toplevel}" config --get remote.origin.url 2> /dev/null || true)"
    [[ "${url}" == *"${AEGIS_REPO_NAME}"* ]]
}

detect_branch() {
    local branch=""
    if in_aegis_repo; then
        branch="$(git rev-parse --abbrev-ref HEAD 2> /dev/null || true)"
    fi
    if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
        branch="${FALLBACK_BRANCH}"
    fi
    echo "${branch}"
}

nvidia_available() {
    # CDI is how modern podman exposes NVIDIA devices. The spec file is
    # generated once on the host with:
    #   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    [[ -f /etc/cdi/nvidia.yaml || -f /var/run/cdi/nvidia.yaml ]]
}

image_label() {
    # $1 = image ref, $2 = label key. Empty string when absent.
    podman image inspect --format "{{ index .Config.Labels \"$2\" }}" "$1" 2> /dev/null || true
}

remote_rev() {
    # $1 = ref. Empty string when it cannot be resolved.
    git ls-remote "${AEGIS_REPO_URL}" "$1" 2> /dev/null | cut -f1 || true
}

build_image() {
    [[ -f "${CONTAINERFILE}" ]] || {
        err "Error: ${CONTAINERFILE} not found."
        exit 1
    }

    podman image exists "${BASE_IMAGE}:${IMAGE_VERSION}" || {
        err "Error: base image ${BASE_IMAGE}:${IMAGE_VERSION} not found."
        err "       Build it first with: aegis_gym_build_image -v ${IMAGE_VERSION}"
        exit 1
    }

    # Resolve the ref to a commit so the aegis_gym layer is rebuilt only when
    # the branch has actually moved.
    local rev
    rev="$(remote_rev "${AEGIS_GYM_TAG}")"
    if [[ -z "${rev}" ]]; then
        echo ">>> Could not resolve '${AEGIS_GYM_TAG}' on the remote, disabling layer cache."
        rev="$(date +%s)"
    fi

    local build_cmd=(podman build "${SCRIPT_DIR}"
        --file "${CONTAINERFILE}"
        --build-arg "BASE_REF=${BASE_IMAGE}:${IMAGE_VERSION}"
        --build-arg "AEGIS_GYM_TAG=${AEGIS_GYM_TAG}"
        --build-arg "AEGIS_GYM_REV=${rev}"
        --build-arg "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        -t "${PROD_REF}")
    ((FORCE_BUILD)) && build_cmd+=(--no-cache)

    echo ">>> Building ${PROD_REF} (${AEGIS_GYM_TAG} @ ${rev:0:8})..."
    "${build_cmd[@]}"
    echo ">>> Built ${PROD_REF}"
    FORCE_BUILD=0
}

prompt_build_settings() {
    # Shows what is about to be built and lets it be edited, mirroring the
    # create-new flow in dev/aegis_gym_toolbx.sh. Updates the globals it touches.
    local reply
    while true; do
        echo
        echo ">>>   base image : ${BASE_IMAGE}:${IMAGE_VERSION}"
        echo ">>>   prod image : ${PROD_IMAGE}:${IMAGE_VERSION}"
        echo ">>>   container  : ${CONTAINER_NAME}"
        echo ">>>   branch/ref : ${AEGIS_GYM_TAG}"
        echo

        read -r -p ">>> Build with these settings? [Y]es / [e]dit / [a]bort: " reply
        case "${reply:-y}" in
            [yY]) break ;;
            [eE])
                read -r -p ">>> Image version [${IMAGE_VERSION}]: " reply
                IMAGE_VERSION="${reply:-${IMAGE_VERSION}}"
                read -r -p ">>> aegis_gym branch/ref [${AEGIS_GYM_TAG}]: " reply
                AEGIS_GYM_TAG="${reply:-${AEGIS_GYM_TAG}}"
                # The version is part of both image references.
                PROD_REF="${PROD_IMAGE}:${IMAGE_VERSION}"
                ;;
            *)
                echo ">>> Aborted."
                exit 0
                ;;
        esac
    done
}

show_provenance() {
    local tag rev built base current
    tag="$(image_label "${PROD_REF}" aegis.gym.tag)"
    rev="$(image_label "${PROD_REF}" aegis.gym.rev)"
    built="$(image_label "${PROD_REF}" aegis.build.date)"
    base="$(image_label "${PROD_REF}" aegis.base.ref)"

    echo
    echo ">>> Image      : ${PROD_REF}"
    echo ">>> Base image : ${base:-unknown}"
    echo ">>> Branch/ref : ${tag:-unknown}"
    echo ">>> Commit     : ${rev:-unknown}"
    echo ">>> Built at   : ${built:-unknown}"

    # Best-effort staleness check against the remote.
    if [[ -n "${tag}" && -n "${rev}" ]]; then
        current="$(remote_rev "${tag}")"
        if [[ -n "${current}" && "${current}" != "${rev}" ]]; then
            warn "NOTE: '${tag}' has moved to ${current:0:8} since this image was built."
        fi
    fi
    echo
}

# --- Build -----------------------------------------------------------------

if [[ -z "${AEGIS_GYM_TAG}" ]]; then
    AEGIS_GYM_TAG="$(detect_branch)"
fi

if ((DO_BUILD == 0)) && ! podman image exists "${PROD_REF}"; then
    echo ">>> ${PROD_REF} not found locally, it has to be built."
    # Only ask when the build was not requested explicitly with -b/-B.
    if ((ASSUME_YES == 0 && DRY_RUN == 0)); then
        prompt_build_settings
    fi
    DO_BUILD=1
fi

if ((DO_BUILD)); then
    if ((DRY_RUN)); then
        echo ">>> [dry-run] would build ${PROD_REF} from ${BASE_IMAGE}:${IMAGE_VERSION}"
    else
        build_image
    fi
fi

# --- Push ------------------------------------------------------------------

if ((DO_PUSH)); then
    podman image exists "${PROD_REF}" || {
        err "Error: ${PROD_REF} does not exist locally, nothing to push."
        exit 1
    }

    if [[ -n "${PUSH_AS}" ]]; then
        REMOTE_REF="${PUSH_AS}"
    else
        REGISTRY="${REGISTRY:-${DEFAULT_REGISTRY}}"
        REMOTE_REF="${REGISTRY}/${PROD_IMAGE}:${IMAGE_VERSION}"
    fi

    echo ">>> Tagging as ${REMOTE_REF}..."
    podman tag "${PROD_REF}" "${REMOTE_REF}"

    echo ">>> Pushing ${REMOTE_REF}..."
    podman push "${REMOTE_REF}"

    echo ">>> Pushed ${REMOTE_REF}"
fi

((DO_RUN)) || exit 0

# --- Confirm ---------------------------------------------------------------

if ((ASSUME_YES == 0 && DRY_RUN == 0)); then
    while true; do
        show_provenance
        read -r -p ">>> Run it? [Y]es / [r]ebuild / [c]leanup and exit / [a]bort: " ACTION
        case "${ACTION:-y}" in
            [yY]) break ;;
            [rR])
                prompt_build_settings
                build_image
                ;;
            [cC])
                echo ">>> Removing ${PROD_REF}..."
                podman rmi --force "${PROD_REF}"
                echo ">>> Cleanup done."
                exit 0
                ;;
            *)
                echo ">>> Aborted."
                exit 0
                ;;
        esac
    done
fi

# --- Run -------------------------------------------------------------------

if podman container exists "${CONTAINER_NAME}"; then
    err "Error: container '${CONTAINER_NAME}' already exists."
    err "       podman rm -f ${CONTAINER_NAME}   (or pass --name)"
    exit 1
fi

RUN_CMD=(podman run --rm --interactive
    --name "${CONTAINER_NAME}"
    # gRPC bridge to aegis_ros on the host
    --network host)

# Only allocate a TTY when there is one, so scripted and piped runs do not
# trip podman's "input device is not a TTY" warning.
[[ -t 0 && -t 1 ]] && RUN_CMD+=(--tty)

# torch DataLoader workers communicate through /dev/shm, whose in-container
# default is a cramped 64MB. Sharing the host's IPC namespace gives them the
# host's /dev/shm instead. podman refuses --shm-size together with --ipc host,
# so an explicit size means a private namespace instead of the host's.
if [[ -n "${SHM_SIZE}" ]]; then
    RUN_CMD+=(--shm-size "${SHM_SIZE}")
else
    RUN_CMD+=(--ipc host)
fi

if ((WITH_CLEARML)); then
    if [[ -f "${CLEARML_CONF}" ]]; then
        RUN_CMD+=(--volume "${CLEARML_CONF}:/root/clearml.conf:ro")
    else
        warn "No ClearML config at ${CLEARML_CONF}; task reporting will be disabled."
    fi
fi

# X11 sockets and passed-through device nodes both need this under SELinux.
NEEDS_LABEL_DISABLE=0

if ((WITH_GUI)); then
    if [[ -z "${DISPLAY:-}" ]]; then
        warn "DISPLAY is not set; the viewer will not be able to open a window."
    fi
    # The base image is headless by default (EGL, PYGLET_HEADLESS); override
    # both so the Genesis viewer can render through GLX.
    RUN_CMD+=(--env "DISPLAY=${DISPLAY:-}"
        --env QT_X11_NO_MITSHM=1
        --env PYOPENGL_PLATFORM=glx
        --env PYGLET_HEADLESS=
        --volume /tmp/.X11-unix:/tmp/.X11-unix:ro)
    NEEDS_LABEL_DISABLE=1
    [[ -d /dev/dri ]] && RUN_CMD+=(--device /dev/dri)
fi

# --- GPU -------------------------------------------------------------------

case "${GPU_MODE}" in
    auto)
        if nvidia_available; then
            USE_NVIDIA=1
        else
            USE_NVIDIA=0
            warn "No NVIDIA CDI spec found; training will fall back to the CPU."
        fi
        ;;
    nvidia)
        USE_NVIDIA=1
        nvidia_available || {
            err "Error: --gpu nvidia requested but no CDI spec found."
            err "       sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
            exit 1
        }
        ;;
    none) USE_NVIDIA=0 ;;
    *)
        err "Unknown --gpu mode '${GPU_MODE}' (nvidia|none|auto)."
        exit 1
        ;;
esac

if ((USE_NVIDIA)); then
    echo ">>> Using the NVIDIA GPU."
    RUN_CMD+=(--device nvidia.com/gpu=all
        # 'graphics' is what the viewer needs; 'compute' alone gives no OpenGL.
        --env NVIDIA_VISIBLE_DEVICES=all
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
        --env __GLX_VENDOR_LIBRARY_NAME=nvidia)
    NEEDS_LABEL_DISABLE=1
fi

if ((NEEDS_LABEL_DISABLE)); then
    RUN_CMD+=(--security-opt label=disable)
fi

# Note the ${arr[@]+"${arr[@]}"} form: a plain "${arr[@]:-}" would expand an
# empty array to one empty string, which the dispatcher would try to exec.
RUN_CMD+=("${PROD_REF}" aegis-gym ${ARGS[@]+"${ARGS[@]}"})

if ((DRY_RUN)); then
    printf '%q ' "${RUN_CMD[@]}"
    echo
    exit 0
fi

echo ">>> Running ${PROD_REF} as ${CONTAINER_NAME}"
exec "${RUN_CMD[@]}"
