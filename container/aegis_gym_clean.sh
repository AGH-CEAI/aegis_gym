#!/bin/bash
# Script generated with Claude Opus 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

NAME_PREFIX="aegis_gym_dev-"
IMAGE_REPO="localhost/aegis_gym_dev"

# Swept only with -a/--all. Ordered so derived images go before the images
# they were built from, otherwise podman refuses to remove the parent.
EXTRA_REPOS=(
    "ceai/aegis_gym_prod"
    "ceai/aegis_gym"
    "ceai/aegis_gym_torch"
)

ASSUME_YES=0
CLEAN_ALL=0

for arg in "$@"; do
    case "${arg}" in
        -y | --yes) ASSUME_YES=1 ;;
        -a | --all) CLEAN_ALL=1 ;;
        -h | --help)
            echo ">>> Usage: $(basename "$0") [-y|--yes] [-a|--all]"
            echo ">>> Removes all ${NAME_PREFIX}* toolbx containers and ${IMAGE_REPO} images."
            echo ">>> With --all, also offers the prod, base and torch images:"
            printf '>>>   %s\n' "${EXTRA_REPOS[@]}"
            exit 0
            ;;
        *)
            echo ">>> Unknown argument '${arg}'. See --help." >&2
            exit 1
            ;;
    esac
done

for cmd in podman toolbox; do
    command -v "${cmd}" > /dev/null 2>&1 || {
        echo ">>> Error: '${cmd}' not found in PATH." >&2
        exit 1
    }
done

confirm() {
    local reply
    ((ASSUME_YES)) && return 0
    read -r -p ">>> $1 (y/N): " reply
    [[ "$(aegis_answer "${reply}" n)" == "y" ]]
}

remove_images() {
    # $1 = repository reference to sweep.
    local repo="$1"
    local -a images candidates
    mapfile -t candidates < <(
        podman images --format '{{.Repository}}:{{.Tag}}' \
            --filter "reference=${repo}" | sort
    )

    # --filter reference matches IMAGES, then podman prints every NAME each
    # matching image carries. One image with two tags in unrelated
    # repositories -- localhost/ceai/aegis_dev:X and geonosis:5000/ceai/aegis:X
    # sharing an id -- therefore yields a row this sweep never asked about, and
    # `-y --all` would remove it unattended. Keep only rows whose repository is
    # the one named, either bare or under a registry/namespace prefix.
    images=()
    local row
    for row in ${candidates[@]+"${candidates[@]}"}; do
        case "${row%:*}" in
            "${repo}" | */"${repo}") images+=("${row}") ;;
        esac
    done

    if [[ ${#images[@]} -eq 0 ]]; then
        echo ">>> No ${repo} images found."
        return 0
    fi

    echo
    echo ">>> Images (${repo}):"
    printf '  %s\n' "${images[@]}"

    if ! confirm "Remove these ${#images[@]} image(s)?"; then
        echo ">>> Skipped ${repo}."
        return 0
    fi

    local image
    for image in "${images[@]}"; do
        echo ">>> Removing ${image}..."
        # An image still referenced by another container or by a derived image
        # cannot be removed; report it instead of aborting the whole cleanup.
        podman rmi "${image}" || {
            echo ">>>   could not remove ${image} (still in use?)" >&2
            FAILED=1
        }
    done
}

FAILED=0

# --- Containers ------------------------------------------------------------

mapfile -t CONTAINERS < <(
    podman ps --all --format '{{.Names}}' \
        --filter "name=^${NAME_PREFIX}" | sort
)

if [[ ${#CONTAINERS[@]} -eq 0 ]]; then
    echo ">>> No ${NAME_PREFIX}* containers found."
else
    echo ">>> Containers to remove:"
    for name in "${CONTAINERS[@]}"; do
        printf '  %-32s %s\n' "${name}" \
            "$(podman inspect -f '{{.State.Status}}' "${name}")"
    done

    if confirm "Remove these ${#CONTAINERS[@]} container(s)?"; then
        for name in "${CONTAINERS[@]}"; do
            echo ">>> Removing ${name}..."
            # Tolerated the same way image removal is, a few lines below: a
            # container that is still running elsewhere must not abort the
            # sweep under `set -e` and leave every image behind.
            toolbox rm --force "${name}" || {
                echo ">>>   could not remove ${name} (still running?)" >&2
                FAILED=1
            }
        done
    else
        echo ">>> Skipped containers."
    fi
fi

# --- Images ----------------------------------------------------------------

remove_images "${IMAGE_REPO}"

if ((CLEAN_ALL)); then
    for repo in "${EXTRA_REPOS[@]}"; do
        remove_images "${repo}"
    done
fi

if ((FAILED)); then
    echo ">>> Some items were left in place." >&2
    echo ">>> Done."
    # Non-zero so a caller in a script or in CI can tell a partial sweep from
    # a clean one; the message alone was invisible to them.
    exit 1
fi

echo ">>> Done."
