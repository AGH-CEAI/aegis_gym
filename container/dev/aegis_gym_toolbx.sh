#!/bin/bash
# Script generated with Claude Opus 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
CONTAINERFILE="${SCRIPT_DIR}/Containerfile.toolbx"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

DEFAULT_IMAGE="ceai/aegis_gym"
DEFAULT_VERSION="v0.1.0"
FALLBACK_BRANCH="devel"
AEGIS_REPO_URL="https://github.com/AGH-CEAI/aegis_gym.git"
AEGIS_REPO_NAME="aegis_gym"
NAME_PREFIX="aegis_gym_dev-"

NO_CACHE=0
WITH_EDITABLE=1

usage() {
    cat << 'EOF'
Usage: aegis_gym_toolbx [options]

Builds localhost/aegis_gym_dev:<ver> from ceai/aegis_gym:<ver>, creates a
toolbx container from it and enters it. With an existing container it offers
to join, recreate, clean up or create another one.

After creating the container the local aegis_gym checkout is installed
editable (--no-deps), so host edits take effect inside immediately. The
checkout is found from $PWD, so run this from inside your clone.

Options:
      --no-cache      Build the image ignoring the layer cache
      --no-editable   Skip the editable install of the local checkout
  -h, --help          This message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache)    NO_CACHE=1 ;;
        --no-editable) WITH_EDITABLE=0 ;;
        -h | --help)   usage; exit 0 ;;
        *) echo ">>> Unknown option '$1'. See --help." >&2; exit 1 ;;
    esac
    shift
done

for cmd in podman toolbox git; do
    command -v "${cmd}" > /dev/null 2>&1 || {
        echo ">>> Error: '${cmd}' not found in PATH." >&2
        exit 1
    }
done

[[ -f "${CONTAINERFILE}" ]] || {
    echo ">>> Error: ${CONTAINERFILE} not found." >&2
    exit 1
}

# --- Helpers ---------------------------------------------------------------

resolve_rev() {
    # The branch as a commit, so the image layer is rebuilt only when the
    # branch has actually moved. A miss falls back to a timestamp, which always
    # busts the cache. Only the bare revision goes to stdout.
    local branch="$1" rev
    if ! rev="$(aegis_resolve_rev "${AEGIS_REPO_URL}" "${branch}")"; then
        echo ">>> Warning: could not resolve '${branch}' on the remote," \
            "disabling layer cache." >&2
        rev="$(date +%s)"
    fi
    echo "${rev}"
}

install_local_aegis_gym() {
    # toolbx shares $HOME, so the host checkout is visible inside at the very
    # same path. --no-deps: the base image already carries the dependencies.
    #
    # sudo is required: toolbx runs as the host user, who cannot write to the
    # image's root-owned /usr/local/lib/python3*/dist-packages. It also fixes
    # which uv runs -- sudo's secure_path finds the image's /usr/local/bin/uv
    # instead of the host's ~/.local/bin/uv that $HOME sharing puts first.
    local name="$1" repo
    ((WITH_EDITABLE)) || {
        echo ">>> Skipping the editable install (--no-editable)."
        return 0
    }

    repo="$(aegis_repo_path)"
    if [[ -z "${repo}" || ! -f "${repo}/pyproject.toml" ]]; then
        echo ">>> No aegis_gym checkout found at or above $(pwd)."
        echo ">>> The container is ready, but Python will not see your local sources."
        echo ">>> Run this from inside your clone, or install by hand:"
        echo ">>>   sudo uv pip install --system --no-deps --editable /path/to/aegis_gym"
        return 0
    fi

    echo ">>> Installing ${repo} as editable..."
    toolbox run --container "${name}" bash -lc \
        "sudo uv pip install --system --no-deps --editable '${repo}'"

    # No env scrub here on purpose: PYTHONNOUSERSITE is baked into the image,
    # so this sees exactly what the interactive shell will see. Wrapping the
    # command in `env PYTHONNOUSERSITE=1` would have hidden a broken image by
    # fixing up the one invocation that checks it.
    echo ">>> Python will import aegis_gym from:"
    toolbox run --container "${name}" \
        python3 -c 'import aegis_gym; print(aegis_gym.__file__)'

    # The whole point of the development image, so fail loudly rather than let
    # someone train against the host's PyPI torch for an afternoon.
    local torch_file
    torch_file="$(toolbox run --container "${name}" \
        python3 -c 'import torch; print(torch.__file__)' 2> /dev/null || true)"
    if [[ "${torch_file}" == "${HOME}/.local/"* ]]; then
        echo ">>> Error: torch resolves to ${torch_file}," >&2
        echo ">>>        i.e. your host's packages are shadowing the image." >&2
        echo ">>>        The image should set PYTHONNOUSERSITE=1; rebuild it with" >&2
        echo ">>>        aegis_gym_toolbx --no-cache and recreate the container." >&2
        exit 1
    fi
}

enter_toolbox() {
    # No env scrub around toolbox: it forwards only its own fixed list of
    # variables, so anything exported here is dropped on the way in. The
    # shared-$HOME shadowing this used to guard against is handled by
    # PYTHONNOUSERSITE=1 in Containerfile.toolbx, where it actually arrives.
    local name="$1"
    echo ">>> Entering ${name}..."
    exec toolbox enter "${name}"
}

require_base_image() {
    # Split out of build_and_enter so [r]ecreate can check it while the old
    # container is still there. Destroying a working container and only then
    # discovering the base image is gone leaves nothing to fall back on, and
    # getting it back costs a multi-GB rebuild.
    local base_ref="$1" version="$2"
    podman image exists "${base_ref}" || {
        echo ">>> Error: base image ${base_ref} not found." >&2
        echo ">>>        Build it first with: aegis_gym_build_image -v ${version}" >&2
        exit 1
    }
}

build_and_enter() {
    local base_ref="$1" version="$2" branch="$3"
    local derived="localhost/aegis_gym_dev:${version}"
    local name="${NAME_PREFIX}${version}"
    local rev locale
    rev="$(resolve_rev "${branch}")"
    locale="$(aegis_host_locale)"

    require_base_image "${base_ref}" "${version}"

    local build_cmd=(podman build "${SCRIPT_DIR}"
        --file "${CONTAINERFILE}"
        --build-arg "BASE_REF=${base_ref}"
        --build-arg "AEGIS_GYM_TAG=${branch}"
        --build-arg "AEGIS_GYM_REV=${rev}"
        --build-arg "LOCALE=${locale}"
        -t "${derived}")
    ((NO_CACHE)) && build_cmd+=(--no-cache)

    echo ">>> Building ${derived} from ${base_ref} (${branch} @ ${rev:0:8}," \
        "locale ${locale})..."
    "${build_cmd[@]}"

    echo ">>> Creating toolbx container ${name}..."
    toolbox create --image "${derived}" "${name}"

    install_local_aegis_gym "${name}"
    enter_toolbox "${name}"
}

create_new() {
    # One confirmation on the defaults; only ask for details on request.
    local base_image="${DEFAULT_IMAGE}" version="${DEFAULT_VERSION}"
    local branch repo
    branch="$(aegis_detect_branch "${FALLBACK_BRANCH}")"
    repo="$(aegis_repo_path)"

    echo
    echo ">>>   base image : ${base_image}:${version}"
    echo ">>>   container  : ${NAME_PREFIX}${version}"
    echo ">>>   branch     : ${branch}"
    echo ">>>   local repo : ${repo:-<none found, editable install will be skipped>}"
    echo

    read -r -p ">>> Create with these settings? [Y]es / [e]dit / [a]bort: " CONFIRM
    case "$(aegis_answer "${CONFIRM}" y)" in
        y) ;;
        e)
            read -r -p ">>> Base image [${base_image}]: " reply
            base_image="${reply:-${base_image}}"
            read -r -p ">>> Image version [${version}]: " reply
            version="${reply:-${version}}"
            read -r -p ">>> aegis_gym branch [${branch}]: " reply
            branch="${reply:-${branch}}"
            ;;
        *)
            echo ">>> Aborted."
            exit 0
            ;;
    esac

    # The container name is derived from the version, so accepting the defaults
    # when a container for that version already exists would fail inside
    # `toolbox create` -- after the image had been rebuilt. Catch it here and
    # ask for a version that is actually free.
    while podman container exists "${NAME_PREFIX}${version}"; do
        echo ">>> Container ${NAME_PREFIX}${version} already exists."
        read -r -p ">>> Image version for the new container (empty aborts): " reply
        if [[ -z "${reply}" ]]; then
            echo ">>> Aborted."
            exit 0
        fi
        version="${reply}"
    done

    build_and_enter "${base_image}:${version}" "${version}" "${branch}"
}

# --- Existing containers ---------------------------------------------------

mapfile -t EXISTING < <(
    podman ps --all --format '{{.Names}}' \
        --filter "name=^${NAME_PREFIX}" | sort
)

if [[ ${#EXISTING[@]} -eq 0 ]]; then
    echo ">>> No existing ${NAME_PREFIX}* container found."
    create_new
fi

if [[ ${#EXISTING[@]} -eq 1 ]]; then
    TARGET="${EXISTING[0]}"
else
    echo ">>> Found ${#EXISTING[@]} ${NAME_PREFIX}* containers:"
    for i in "${!EXISTING[@]}"; do
        printf '  %d) %-32s %s\n' "$((i + 1))" "${EXISTING[i]}" \
            "$(podman inspect -f '{{.State.Status}}' "${EXISTING[i]}")"
    done
    read -r -p ">>> Select [1]: " SEL
    SEL="${SEL:-1}"
    if ! [[ "${SEL}" =~ ^[0-9]+$ ]] || ((SEL < 1 || SEL > ${#EXISTING[@]})); then
        echo ">>> Invalid selection." >&2
        exit 1
    fi
    TARGET="${EXISTING[SEL - 1]}"
fi

TARGET_VERSION="${TARGET#"${NAME_PREFIX}"}"
TARGET_IMAGE="localhost/aegis_gym_dev:${TARGET_VERSION}"
TARGET_STATE="$(podman inspect -f '{{.State.Status}}' "${TARGET}")"

echo ">>> Container '${TARGET}' exists (state: ${TARGET_STATE})."
read -r -p ">>> [J]oin / [r]ecreate / [c]leanup / [n]ew: " ACTION

case "$(aegis_answer "${ACTION}" j)" in
    j)
        enter_toolbox "${TARGET}"
        ;;
    r)
        BRANCH="$(aegis_detect_branch "${FALLBACK_BRANCH}")"
        read -r -p ">>> aegis_gym branch [${BRANCH}]: " REPLY_BRANCH
        BRANCH="${REPLY_BRANCH:-${BRANCH}}"
        # Before the rm, not after: see require_base_image.
        require_base_image "${DEFAULT_IMAGE}:${TARGET_VERSION}" "${TARGET_VERSION}"
        echo ">>> Removing ${TARGET}..."
        toolbox rm --force "${TARGET}"
        build_and_enter "${DEFAULT_IMAGE}:${TARGET_VERSION}" \
            "${TARGET_VERSION}" "${BRANCH}"
        ;;
    c)
        echo ">>> Removing ${TARGET}..."
        toolbox rm --force "${TARGET}"
        if podman image exists "${TARGET_IMAGE}"; then
            read -r -p ">>> Also remove image ${TARGET_IMAGE}? (y/N): " RM_IMAGE
            case "$(aegis_answer "${RM_IMAGE}" n)" in
                y) podman rmi "${TARGET_IMAGE}" ;;
            esac
        fi
        echo ">>> Cleanup done."
        exit 0
        ;;
    n)
        create_new
        ;;
    *)
        echo ">>> Unknown option '${ACTION}', aborting." >&2
        exit 1
        ;;
esac
