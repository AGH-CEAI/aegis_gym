#!/bin/bash
# Shared helpers for the aegis_gym container commands.
#
# Sourced, never executed. Keep it free of side effects: no `set` changes, no
# output at source time, no globals beyond the functions themselves. Each
# caller keeps its own `set -euo pipefail` and its own message prefix.
#
# It exists because repo detection, branch detection and ref resolution were
# copy-pasted into aegis_gym_build_image.sh, run/aegis_gym_run.sh and
# dev/aegis_gym_toolbx.sh, and had already drifted apart three ways.

AEGIS_REPO_NAME="${AEGIS_REPO_NAME:-aegis_gym}"
AEGIS_REPO_URL="${AEGIS_REPO_URL:-https://github.com/AGH-CEAI/aegis_gym.git}"

aegis_repo_path() {
    # Absolute path of the aegis_gym checkout $PWD sits in, empty otherwise.
    # Being in some other git repo does not count.
    local toplevel url
    toplevel="$(git rev-parse --show-toplevel 2> /dev/null || true)"
    [[ -n "${toplevel}" ]] || return 0

    if [[ "$(basename "${toplevel}")" == "${AEGIS_REPO_NAME}" ]]; then
        echo "${toplevel}"
        return 0
    fi

    url="$(git -C "${toplevel}" config --get remote.origin.url 2> /dev/null || true)"
    if [[ "${url}" == *"${AEGIS_REPO_NAME}"* ]]; then
        echo "${toplevel}"
    fi
}

aegis_detect_branch() {
    # $1 = fallback branch. The branch of the checkout $PWD sits in, resolved
    # through aegis_repo_path rather than $PWD so it is the same answer in
    # every caller -- aegis_gym_run used to run `git rev-parse` in $PWD while
    # aegis_gym_toolbx ran it in the repo root.
    local fallback="${1:-devel}"
    local branch="" repo
    repo="$(aegis_repo_path)"
    if [[ -n "${repo}" ]]; then
        branch="$(git -C "${repo}" rev-parse --abbrev-ref HEAD 2> /dev/null || true)"
    fi
    if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
        branch="${fallback}"
    fi
    echo "${branch}"
}

aegis_resolve_rev() {
    # $1 = repository URL, $2 = ref. Prints the commit and returns 0, or
    # returns 1 when the ref cannot be resolved. Callers decide what a miss
    # means; nothing but the revision goes to stdout.
    local url="$1" ref="$2" rows rev

    rows="$(git ls-remote "${url}" "${ref}" 2> /dev/null || true)"

    if [[ -n "${rows}" ]]; then
        # ls-remote can answer with more than one row -- a name that exists as
        # both a branch and a tag, or a ref given as a glob -- and a newline in
        # the value would be carried into --build-arg, the RUN cache key and
        # the aegis.gym.rev label. Prefer a peeled tag (^{}), which is the
        # commit rather than the tag object, and otherwise take the first row.
        rev="$(awk '$2 ~ /\^\{\}$/ { print $1; exit }' <<< "${rows}")"
        [[ -n "${rev}" ]] || rev="$(awk 'NR == 1 { print $1 }' <<< "${rows}")"
        echo "${rev}"
        return 0
    fi

    # ls-remote matches ref names only, so a bare commit id always misses.
    # Checked after the lookup, not before, so a branch that happens to be
    # named like a hex string still resolves as a branch. run/Containerfile.prod
    # handles a bare SHA through its fallback full clone plus checkout, so pass
    # it through instead of degrading to the timestamp below -- that timestamp
    # would otherwise be baked into org.opencontainers.image.revision.
    if [[ "${ref}" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        echo "${ref}"
        return 0
    fi

    return 1
}

aegis_answer() {
    # Normalises a menu reply to a single lowercase letter, so "y" and "yes",
    # "j" and "join", "a" and "abort" all select the same branch. $1 = the
    # reply, $2 = the default to use when the reply is empty.
    local reply="${1:-${2:-}}"
    printf '%s' "${reply:0:1}" | tr '[:upper:]' '[:lower:]'
}

aegis_host_locale() {
    # `toolbox enter` forwards the host's $LANG into the container but not its
    # $LC_ALL, so the image has to carry that very locale. When it does not,
    # zsh silently drops to single-byte C and every wide character breaks --
    # most visibly a powerline prompt, which aborts on $'' and leaves a
    # bare 'toolbx%'. C/POSIX carry no such requirement, so for those the image
    # default is good enough.
    local loc="${LANG:-}"
    case "${loc}" in
        C.* | POSIX.*) echo "en_US.UTF-8" ;;
        *.UTF-8 | *.utf8) echo "${loc}" ;;
        *) echo "en_US.UTF-8" ;;
    esac
}
