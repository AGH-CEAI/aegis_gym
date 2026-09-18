# Manual test procedure

Step-by-step checks for the four installed commands plus `install_links.sh`.
Roughly 30 minutes if the images below are already present; longer if you
choose to rebuild a base tier from scratch.

## Starting state

Already built and verified on this machine:

| Image / container                                                | Exists              |
| ---------------------------------------------------------------- | ------------------- |
| `ceai/aegis_gym_torch:v0.1.0` and `:latest`                      | yes (same image ID) |
| `ceai/aegis_gym:v0.1.0` and `:latest`                            | yes (same image ID) |
| `ceai/aegis_gym_prod:latest`                                     | yes                 |
| `localhost/aegis_gym_dev:latest` + toolbx `aegis_gym_dev-latest` | yes                 |

The default image version is now **`v0.1.0`**, so `ceai/aegis_gym_prod:v0.1.0`
and the `aegis_gym_dev-v0.1.0` toolbx container do **not** exist yet — steps 3
and 4 build them, which is the point.

Note: `:v0.1.0` and `:latest` share an image ID, so their `aegis.build.date`
label shows the earlier build. That is podman deduplicating identical content,
not a stale image.

---

## 1. `install_links.sh`

```bash
cd ~/ceai/ros_ws/src/aegis_gym/container

./install_links.sh --help          # lists the 4 commands it installs
./install_links.sh --dry-run       # prints 4 ln -s, changes nothing
ls -l ~/.local/bin/aegis_gym_*     # still whatever it was before
```

Install, then confirm idempotency and that the links point into this repo:

```bash
./install_links.sh
ls -l ~/.local/bin/aegis_gym_*     # 4 symlinks -> this container/ directory
./install_links.sh                 # "already up to date." x4
```

Check the safety guard — it must refuse to delete a link it does not own:

```bash
ln -sf /bin/true ~/.local/bin/aegis_gym_clean     # impostor
./install_links.sh --uninstall                    # "Skipping ...: points elsewhere."
rm ~/.local/bin/aegis_gym_clean
./install_links.sh                                # reinstall all 4
```

**Expect:** 4 links, idempotent reruns, the impostor left untouched.

---

## 2. `aegis_gym_build_image`

```bash
cd ~                               # prove it does not depend on $PWD
aegis_gym_build_image --help
```

Interactive run — accept every default with Enter. The torch tier exists, so it
asks whether to rebuild; answer **n** (default), the slow layer is then kept:

```bash
aegis_gym_build_image
```

**Expect:** prompts for version `[v0.1.0]`, gym ref `[devel]`, rsl_rl `[v3.3.2]`,
aegis_ros `[humble-devel]`; `>>> ceai/aegis_gym_torch:v0.1.0 already exists.`;
then a cached build of the dependency tier in seconds; finally a push prompt —
answer **n**.

Non-interactive, as CI would run it:

```bash
aegis_gym_build_image -y                  # no prompts at all
aegis_gym_build_image -y -v test-tag      # builds a separate tag
podman images | grep aegis_gym            # test-tag present
podman rmi ceai/aegis_gym:test-tag ceai/aegis_gym_torch:test-tag
```

Optional (slow, ~20 min): `aegis_gym_build_image -y --torch-only --rebuild-torch`
rebuilds only the CUDA tier.

**Expect:** exit 0 every time; `-y` never prompts.

---

## 3. `aegis_gym_run`

Nothing is built for `v0.1.0` yet, so start with the no-op inspection:

```bash
cd ~/ceai/ros_ws/src/aegis_gym     # so the branch is detected from git
aegis_gym_run --help
aegis_gym_run --dry-run train --env reacher -e TEST_PLAYGROUND_dry
```

**Expect** in the printed podman line: `--network host`, `--ipc host`,
the `clearml.conf` mount, `--device nvidia.com/gpu=all`,
`--security-opt label=disable`, ending in
`ceai/aegis_gym_prod:v0.1.0 aegis-gym train --env reacher -e ...`.

Check the flags that alter it:

```bash
aegis_gym_run --dry-run --gui eval --env reacher       # adds DISPLAY, /tmp/.X11-unix, /dev/dri, PYOPENGL_PLATFORM=glx
aegis_gym_run --dry-run --gpu none shell               # no nvidia device
aegis_gym_run --dry-run --no-clearml shell             # no clearml.conf mount
aegis_gym_run --dry-run --shm-size 8g shell            # --shm-size instead of --ipc host, never both
```

Now build the v0.1.0 production image (a few minutes — clone + install only):

```bash
aegis_gym_run -b --no-run
```

**Expect:** the `[Y]es / [e]dit / [a]bort` prompt is skipped because `-b` was
explicit; image `ceai/aegis_gym_prod:v0.1.0` is built.

Verify the ClearML contract — this is the one that must not regress:

```bash
podman inspect ceai/aegis_gym_prod:v0.1.0 --format '{{.Config.Entrypoint}}'   # -> []
podman inspect ceai/aegis_gym_prod:v0.1.0 --format '{{.Config.Cmd}}'          # -> [/bin/bash]
podman run --rm ceai/aegis_gym_prod:v0.1.0 bash -c 'echo agent-style invocation works'
```

Run it for real:

```bash
aegis_gym_run -y shell python3 -c "import torch,aegis_gym;print(torch.cuda.is_available(), aegis_gym.__file__)"
aegis_gym_run -y train --env reacher --control sim -a rl -B 16 --max-iterations 2 -e TEST_PLAYGROUND_manual
```

**Expect:** `True`, then a real 2-iteration run that appears in ClearML under
`TEST_PLAYGROUND`. Without `-y` you get the provenance summary and the
`[Y]es / [r]ebuild / [c]leanup and exit / [a]bort` prompt first — worth seeing
once.

Name-collision guard:

```bash
podman run -d --name aegis_gym_prod ceai/aegis_gym_prod:v0.1.0 sleep 60
aegis_gym_run -y shell echo hi        # must refuse: container already exists
podman rm -f aegis_gym_prod
```

---

## 4. `aegis_gym_toolbx`

Run it **from inside the clone** — that is how it finds the checkout to install:

```bash
cd ~/ceai/ros_ws/src/aegis_gym
aegis_gym_toolbx --help
aegis_gym_toolbx
```

Answer **Y**. It builds `localhost/aegis_gym_dev:v0.1.0`, creates the toolbx
container `aegis_gym_dev-v0.1.0`, installs your checkout editable via `sudo`,
prints the import path, and enters.

**Expect** the printed path to be your working copy:
`/home/macale/ceai/ros_ws/src/aegis_gym/aegis_gym/__init__.py`

Inside the container:

```bash
python3 -c "import aegis_gym, torch; print(aegis_gym.__file__); print(torch.__version__, torch.cuda.is_available())"
```

**Expect:** your checkout path, `2.8.0+cu129`, `True`. If torch says **cu128**,
the `$HOME` shadowing is back — that means the scrub was bypassed.

Prove the edit is live, then leave:

```bash
python3 -c "import aegis_gym; print(aegis_gym.__file__)"   # note the path
exit
```

Second run now shows the existing-container menu (there will be two containers,
`-latest` from earlier testing and `-v0.1.0`, so you also get the selector):

```bash
aegis_gym_toolbx
```

Try each branch across separate runs: **j** joins, **n** creates another, **r**
recreates (asks for the branch first), **c** removes the container and offers to
remove its image.

Also check `aegis_gym_toolbx --no-editable` (skips the editable install) and
running it from `/tmp` — outside any clone it must warn that no checkout was
found and continue rather than fail.

---

## 5. `aegis_gym_clean`

```bash
aegis_gym_clean --help
aegis_gym_clean                 # answer n, n -- lists, deletes nothing
```

**Expect:** the `aegis_gym_dev-*` containers and `localhost/aegis_gym_dev`
images listed, then "Skipped".

```bash
aegis_gym_clean --all           # answer n to every prompt
```

**Expect** four image groups listed in this order, derived before parent:
`localhost/aegis_gym_dev`, `ceai/aegis_gym_prod`, `ceai/aegis_gym`,
`ceai/aegis_gym_torch`.

When you actually want the dev artifacts gone:

```bash
aegis_gym_clean -y              # no prompts, removes dev containers + dev images
```

Do **not** run `aegis_gym_clean -y --all` unless you want the base and torch
tiers gone too — that is about 11 GB and a full rebuild. (The tiers share
layers, so the real cost is the base chain, not the sum of what
`podman images` prints per tag.)

---

## Quick regression checklist

| Check                           | Expected                                  |
| ------------------------------- | ----------------------------------------- |
| `aegis-gym` with no arguments   | drops to a shell, exit 0 (not 1)          |
| `aegis_gym_run train -a=rl ...` | `-a=rl` accepted, not "Unknown option"    |
| `--ipc host` and `--shm-size`   | never both in one podman line             |
| prod image `Entrypoint`         | `[]`                                      |
| base image torch / torchvision  | both `+cu129`                             |
| base image `rsl_rl`             | `direct_url.json` -> `file:///tmp/rsl_rl` |
| base image `import aegis_gym`   | ModuleNotFoundError (deps-only by design) |
| toolbx `torch.__version__`      | `2.8.0+cu129`, not cu128                  |
