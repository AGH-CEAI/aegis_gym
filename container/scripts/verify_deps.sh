#!/usr/bin/env bash
# Checks the container's installed Python packages against a checkout's
# uv.lock, and optionally installs the lock's versions for whatever drifted.
#
# Runs INSIDE the container (toolbx shares $HOME, so the checkout is visible
# there at the host path). dev/aegis_gym_toolbx.sh calls it after creating or
# joining a container; it is also fine to run by hand:
#
#   bash container/scripts/verify_deps.sh ~/path/to/aegis_gym
#   bash container/scripts/verify_deps.sh ~/path/to/aegis_gym --install
#
# Why this exists: the base image installs dependencies once, from the uv.lock
# of the branch cloned at image build time (scripts/install_simulation.sh). The
# development container then adds only the checkout itself, with --no-deps. So
# nothing in the normal create/recreate path ever compares what is installed
# against what the lock asks for, and an image a few weeks old silently runs a
# different dependency set than the one the checkout is locked to.
#
# Exit codes: 0 = in sync, 3 = drift found, 1 = error.

set -euo pipefail

# Same family that install_simulation.sh keeps out of the lock export: these
# are installed from the CUDA wheel index or from git, and the lock pins the
# PyPI build of the same name. Reinstalling them from the lock would put a
# +cu128 torch over the +cu129 one and mismatch the whole CUDA stack. Kept as
# a regex rather than a list so it keeps up as nvidia-* packages come and go.
readonly OWNED_ELSEWHERE_RE='nvidia-[a-z0-9.-]*|torch|torchvision|triton|rsl-rl-lib'

REPO=""
DO_INSTALL=0

usage() {
    cat << 'EOF'
Usage: verify_deps.sh REPO [--install]

Compares the installed packages against REPO's uv.lock and reports every
package whose version differs, plus anything the lock asks for that is not
installed at all. torch, torchvision, triton, the nvidia-* runtimes and
rsl-rl-lib are excluded -- they are owned by the image, not by the lock.

Options:
      --install   Offer nothing, just install the lock's versions for the
                  packages that drifted (--no-deps, so nothing else moves)
  -h, --help      This message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)   DO_INSTALL=1 ;;
        -h | --help) usage; exit 0 ;;
        -*)
            echo ">>> Unknown option '$1'. See --help." >&2
            exit 1
            ;;
        *) REPO="$1" ;;
    esac
    shift
done

[[ -n "${REPO}" ]] || {
    echo ">>> Error: no repository path given. See --help." >&2
    exit 1
}
[[ -f "${REPO}/uv.lock" ]] || {
    echo ">>> Error: ${REPO}/uv.lock not found." >&2
    exit 1
}

# Pin the binary instead of trusting $PATH. toolbx shares $HOME, so a host
# ~/.local/bin/uv sits ahead of the image's own copy and would report on a
# different environment than the one we are about to change.
UV="/usr/local/bin/uv"
[[ -x "${UV}" ]] || UV="$(command -v uv 2> /dev/null || true)"
[[ -n "${UV}" ]] || {
    echo ">>> Error: uv not found (looked for /usr/local/bin/uv and \$PATH)." >&2
    exit 1
}

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# --- What the lock asks for ------------------------------------------------

# A failure here must not be mistaken for "nothing to do": an empty or partial
# export would make every installed package look correct. install_simulation.sh
# guards the same call for the same reason.
if ! (cd "${REPO}" && "${UV}" export \
    --no-emit-project \
    --extra sim-genesis \
    --extra test \
    --no-hashes \
    --no-annotate \
    --no-header) > "${TMP}/want.txt" 2> "${TMP}/export.err"; then
    echo ">>> Error: 'uv export' failed for ${REPO}:" >&2
    sed 's/^/>>>        /' "${TMP}/export.err" >&2
    exit 1
fi

if ! grep -qE "^(${OWNED_ELSEWHERE_RE})==" "${TMP}/want.txt"; then
    echo ">>> Error: the lock listed no torch/nvidia packages to exclude," >&2
    echo ">>>        which means the export is not the one this image" >&2
    echo ">>>        expects. Refusing to compare against it." >&2
    exit 1
fi

# --- Compare ---------------------------------------------------------------

compare() {
    # Writes the drifted requirements to $TMP/todo.txt and prints the table.
    # Returns 3 when anything drifted, 0 when the sets agree.
    "${UV}" pip freeze --system > "${TMP}/have.txt" 2> /dev/null

    # Passed under a different name: a prefix assignment to the readonly
    # OWNED_ELSEWHERE_RE itself is an error, not an override.
    OWNED_RE="${OWNED_ELSEWHERE_RE}" TMP="${TMP}" python3 << 'PY'
import os
import re
import sys

tmp = os.environ["TMP"]
skip = re.compile(r"^(%s)$" % os.environ["OWNED_RE"])


def parse(path):
    """name -> version, for plain `name==version` pins only.

    Anything installed from source -- the rsl_rl fork, the aegis_grpc client,
    the editable checkout itself -- appears as a URL or `-e` line with no
    version to compare, so it is skipped rather than reported as missing.
    """
    found = {}
    with open(path) as fh:
        for line in fh:
            line = line.split(";")[0].strip()
            if not line or line.startswith(("#", "-")):
                continue
            m = re.match(r"^([A-Za-z0-9._-]+)==([^\s]+)$", line)
            if m:
                found[m.group(1).lower().replace("_", "-")] = m.group(2)
    return found


want = parse(os.path.join(tmp, "want.txt"))
have = parse(os.path.join(tmp, "have.txt"))

todo, rows = [], []
for name in sorted(want):
    if skip.match(name):
        continue
    if name not in have:
        rows.append((name, want[name], "not installed"))
        todo.append(f"{name}=={want[name]}")
    elif have[name] != want[name]:
        rows.append((name, want[name], have[name]))
        todo.append(f"{name}=={want[name]}")

with open(os.path.join(tmp, "todo.txt"), "w") as fh:
    fh.write("\n".join(todo) + ("\n" if todo else ""))

if not rows:
    print(">>> All lock-pinned packages match the installed versions.")
    sys.exit(0)

print(f">>> {len(rows)} package(s) differ from the lock:")
width = max(len(r[0]) for r in rows)
for name, wanted, installed in rows:
    print(f"      {name:<{width}}  lock={wanted:<14} installed={installed}")
sys.exit(3)
PY
}

status=0
echo ">>> Verifying dependencies against ${REPO}/uv.lock..."
compare || status=$?
[[ ${status} -eq 0 || ${status} -eq 3 ]] || exit "${status}"

# Advisory: reports genuinely unsatisfiable requirements, including ones no
# lock comparison can see -- the image apt-installs python3-protobuf and
# python3-grpcio in a later layer than the pip install, and those shadow the
# lock's versions. Never fatal; it describes the image, not this checkout.
if ! "${UV}" pip check --system > "${TMP}/check.txt" 2>&1; then
    echo ">>> WARNING: 'uv pip check' reports unsatisfied requirements:"
    grep -v '^Using Python' "${TMP}/check.txt" | sed 's/^/      /'
fi

[[ ${status} -eq 3 ]] || exit 0

# --- Install ---------------------------------------------------------------

if ((!DO_INSTALL)); then
    exit 3
fi

echo ">>> torch, torchvision, triton, nvidia-* and rsl-rl-lib are excluded" \
    "and will not be touched."
echo ">>> Installing the lock's versions..."

# sudo: the caller is the host user, who cannot write to the image's
# root-owned dist-packages. --no-deps: install exactly the drifted pins and
# nothing else, so no transitive resolve can reach into the CUDA stack. A
# package pulled in as a new transitive dependency shows up as "not installed"
# in the table above and is therefore already on this list.
sudo "${UV}" pip install --system --no-deps --requirement "${TMP}/todo.txt"

echo ">>> Re-checking..."
status=0
compare || status=$?
exit "${status}"
