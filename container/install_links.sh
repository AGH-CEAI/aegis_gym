#!/bin/bash
# Script generated with Claude Opus 5

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
BIN_DIR="${HOME}/.local/bin"

# "<path relative to the repo>:<command name>"
LINKS=(
    "run/aegis_gym_run.sh:aegis_gym_run"
    "dev/aegis_gym_toolbx.sh:aegis_gym_toolbx"
    "aegis_gym_clean.sh:aegis_gym_clean"
    "aegis_gym_build_image.sh:aegis_gym_build_image"
)

UNINSTALL=0
FORCE=0
DRY_RUN=0

usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Creates symlinks in ${BIN_DIR} for the aegis_gym container scripts.

Options:
      --prefix DIR   Install into DIR instead of ${BIN_DIR}
  -u, --uninstall    Remove the symlinks this script created
  -f, --force        Overwrite existing files without asking
      --dry-run      Show what would happen, change nothing
  -h, --help         This message

Commands installed:
EOF
    for entry in "${LINKS[@]}"; do
        printf '  %-24s -> %s\n' "${entry#*:}" "${entry%%:*}"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)        BIN_DIR="$2"; shift ;;
        -u | --uninstall) UNINSTALL=1 ;;
        -f | --force)    FORCE=1 ;;
        --dry-run)       DRY_RUN=1 ;;
        -h | --help)     usage; exit 0 ;;
        *) echo "Unknown option '$1'. See --help." >&2; exit 1 ;;
    esac
    shift
done

run() {
    if ((DRY_RUN)); then
        printf '  [dry-run]'
        printf ' %q' "$@"
        echo
    else
        "$@"
    fi
}

# --- Uninstall -------------------------------------------------------------

if ((UNINSTALL)); then
    for entry in "${LINKS[@]}"; do
        target="${BIN_DIR}/${entry#*:}"
        source_path="${REPO_DIR}/${entry%%:*}"

        if [[ ! -L "${target}" ]]; then
            [[ -e "${target}" ]] \
                && echo "Skipping ${target}: not a symlink."
            continue
        fi

        # Only remove links that actually point into this repo.
        if [[ "$(readlink -f "${target}")" == "$(readlink -f "${source_path}")" ]]; then
            echo "Removing ${target}"
            run rm "${target}"
        else
            echo "Skipping ${target}: points elsewhere."
        fi
    done
    echo "Done."
    exit 0
fi

# --- Install ---------------------------------------------------------------

echo "Repository : ${REPO_DIR}"
echo "Target dir : ${BIN_DIR}"
echo

run mkdir -p "${BIN_DIR}"

INSTALLED=0
for entry in "${LINKS[@]}"; do
    rel="${entry%%:*}"
    name="${entry#*:}"
    source_path="${REPO_DIR}/${rel}"
    target="${BIN_DIR}/${name}"

    if [[ ! -f "${source_path}" ]]; then
        echo "  ${name}: source ${rel} not found, skipping."
        continue
    fi

    [[ -x "${source_path}" ]] || run chmod +x "${source_path}"

    if [[ -L "${target}" ]] \
        && [[ "$(readlink -f "${target}")" == "$(readlink -f "${source_path}")" ]]; then
        echo "  ${name}: already up to date."
        INSTALLED=$((INSTALLED + 1))
        continue
    fi

    if [[ -e "${target}" || -L "${target}" ]]; then
        if ((FORCE == 0)); then
            read -r -p "  ${target} exists. Replace? (y/N): " reply
            case "${reply}" in
                [yY] | [yY][eE][sS]) ;;
                *) echo "  ${name}: skipped."; continue ;;
            esac
        fi
        run rm -f "${target}"
    fi

    echo "  ${name} -> ${rel}"
    run ln -s "${source_path}" "${target}"
    INSTALLED=$((INSTALLED + 1))
done

# --- PATH check ------------------------------------------------------------

echo
case ":${PATH}:" in
    *":${BIN_DIR}:"*)
        echo "${BIN_DIR} is on your PATH."
        ;;
    *)
        echo "Warning: ${BIN_DIR} is not on your PATH."
        echo "Add this to your ~/.bashrc or ~/.zshrc, then open a new shell:"
        echo
        echo "    export PATH=\"${BIN_DIR}:\${PATH}\""
        ;;
esac

echo
echo "Installed ${INSTALLED} command(s)."
