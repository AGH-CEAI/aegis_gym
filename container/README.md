# aegis_gym containers

> [!IMPORTANT]
> The following README and scripts were written manually at first, and then processed fully by Claude Code.
>
> You should approach this directory as a fully vibe coded module for launching **production** (podman) and **development** (toolbx) containers.

Everything `aegis_gym` needs — CUDA-built PyTorch, the Genesis simulator and
its graphics stack, the AGH `rsl_rl` fork, and the gRPC client that talks to
[`aegis_ros`](https://github.com/AGH-CEAI/aegis_ros) — packaged so you do not
install any of it on your host.

There are two ways to use it, and picking the right one first saves an hour:

| You want to…                                             | Use                    | You get                                                                                                                                   |
| -------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Edit the code, run experiments, debug                    | **`aegis_gym_toolbx`** | A [toolbx](https://containertoolbx.org/) shell sharing your `$HOME`, with *your checkout* installed editable — host edits apply instantly |
| Run a training exactly as it will run on a ClearML queue | **`aegis_gym_run`**    | A throwaway `podman run` of a self-contained image with the code baked in at a fixed commit                                               |

If you are new here and about to write code, you want **`aegis_gym_toolbx`**.

---

## Quick start

Four steps, in order. The base images are shared by both paths, so steps 1–2
happen once per machine.

```bash
# 1. put the commands on $PATH (once)
cd ~/ceai_ws/ros_ws/src/aegis_gym/container
./install_links.sh

# 2. build the base images (once; slow -- see below)
aegis_gym_build_image -y

# 3a. development: run from inside your clone
cd ~/ceai_ws/ros_ws/src/aegis_gym
aegis_gym_toolbx

# 3b. or production: run a training
aegis_gym_run train --env reacher -a rl -B 4096 -e TEST_PLAYGROUND_run1
```

**Budget for step 2.** The first build pulls roughly 3 GB of CUDA wheels plus
the Genesis stack, so expect tens of minutes on a good connection. Later builds
hit podman's layer cache and finish in seconds. The finished set costs about
**11 GB** of disk — the tiers share layers, so it is *not* the sum of the
per-tag sizes `podman images` prints.

---

## Prerequisites

| Need                     | Why                                   | Check                     |
| ------------------------ | ------------------------------------- | ------------------------- |
| `podman`                 | builds and runs everything            | `podman --version`        |
| `toolbox`                | the development path                  | `toolbox --version`       |
| `git`                    | resolving refs, cloning into images   | `git --version`           |
| NVIDIA driver            | GPU                                   | `nvidia-smi`              |
| NVIDIA Container Toolkit | exposes the GPU to containers         | `nvidia-ctk --version`    |
| A CDI spec               | how podman finds the GPU              | `ls /etc/cdi/nvidia.yaml` |
| `~/clearml.conf`         | `train.py` / `eval.py` log to ClearML | `clearml-init` writes it  |
| ~11 GB free disk         | the image set                         | `df -h /`                 |

Generate the CDI spec once — and **again after every driver update**. Forgetting
that is the single most common way GPU access breaks here:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

For the Ubuntu 22.04 / 24.04 host packages needed to get a GPU inside toolbx
(newer `crun`, `podman`, `shadow`), see
[`aegis_docker/docs/ubuntu_gpu_toolbx.md`](https://github.com/AGH-CEAI/aegis_docker).

> This project expects a reachable self-hosted ClearML server for training and
> evaluation — see the caution in the [root README](../README.md). Run
> `clearml-init` before your first training run; `aegis_gym_run` mounts the
> resulting `~/clearml.conf` read-only and warns if it is missing.

---

## How the images fit together

```
ubuntu:22.04
 └─ ceai/aegis_gym_torch:<ver>      torch + torchvision, CUDA wheel index
     └─ ceai/aegis_gym:<ver>        graphics stack, rsl_rl fork, deps, gRPC client
         ├─ ceai/aegis_gym_prod:<ver>       production  -> podman run
         └─ localhost/aegis_gym_dev:<ver>   development -> toolbx
```

Two things about this shape are deliberate:

**The torch tier has its own tag** so that rebuilding dependencies — which you
do whenever `uv.lock` changes — never re-downloads the multi-GB CUDA wheels.

**Neither base tier contains `aegis_gym` itself**, only its dependencies,
pinned from `uv.lock`. The production image adds the package at a fixed commit;
the development container installs your working copy instead. That is the whole
difference between the two paths.

Packages that an earlier step already installed — `torch`, `torchvision`,
`triton`, the `nvidia-*` runtimes and the `rsl_rl` fork — are deliberately
excluded from the lock export. Without that, the lock's PyPI `torch==2.8.0`
(a **cu128** build) would replace the cu129 one from the torch tier and leave a
mismatched pair. See [`scripts/install_simulation.sh`](scripts/install_simulation.sh).

---

## The four commands

`./install_links.sh` symlinks these into `~/.local/bin`, pointing back at this
directory. They resolve their own location, so they work from anywhere:

| Command                 | Does                                              |
| ----------------------- | ------------------------------------------------- |
| `aegis_gym_build_image` | Builds the two base tiers                         |
| `aegis_gym_run`         | Builds and runs the production container          |
| `aegis_gym_toolbx`      | Builds, creates and enters the development toolbx |
| `aegis_gym_clean`       | Removes development containers and images         |

`~/.local/bin` must be on your `$PATH`; the installer tells you if it is not.
`./install_links.sh --uninstall` removes them again, and it only deletes links
that actually point into this repo. Every command takes `-h`.

---

## Build the base images

The default image version is **`v0.1.0`**; pass `-v` for another.

```bash
aegis_gym_build_image           # interactive, prompts for refs
aegis_gym_build_image -y        # all defaults, no prompts (CI)
```

It asks for the image version and the `aegis_gym` / `rsl_rl` / `aegis_ros`
refs, builds `ceai/aegis_gym_torch:<ver>` (offering to skip it when it already
exists — say yes to skip; it is the expensive one), then `ceai/aegis_gym:<ver>`,
and finally offers to push both.

Useful flags: `--no-cache` rebuilds dependencies only, `--rebuild-torch` forces
the CUDA tier, `--torch-only` stops after it, `--push[=HOST]` publishes.

---

## Development — `aegis_gym_toolbx`

Run it **from inside your clone**; that is how it finds the checkout to install.

```bash
cd ~/path/to/aegis_gym
aegis_gym_toolbx
```

It builds `localhost/aegis_gym_dev:<ver>`, creates the toolbx container
`aegis_gym_dev-<ver>`, installs your checkout editable and enters it. When a
container already exists, it offers `[J]oin / [r]ecreate / [c]leanup / [n]ew`;
if you have several, it asks which one to act on first.

Inside, your host edits are live — no reinstall, no rebuild:

```bash
python3 -c "import aegis_gym; print(aegis_gym.__file__)"
# -> /home/you/path/to/aegis_gym/aegis_gym/__init__.py   (your working copy)
python3 train.py -a rl --env reacher -B 16 --max-iterations 10 -e TEST_PLAYGROUND_dev
python3 -m pytest -v .                 # pytest is already in the image
```

Run tests with `python3 -m pytest`, not `uv run … pytest`: the dependencies are
installed system-wide in the image, whereas `uv run` would build a separate
`.venv` and download them all again. (The root README's `uv run` form is for
working on the host without a container.)

toolbx supplies the GPU, X11, `$HOME` and device nodes itself, which is why the
script passes no mount or `--device` flags.

### Dependency drift

The editable install is `--no-deps`, and the image's dependencies were frozen
when it was built — from the `uv.lock` of the branch cloned at build time
(`scripts/install_simulation.sh`), not from your working tree. So a bump in
your lock cannot reach an existing container on its own, and `require_base_image`
deliberately refuses to rebuild the base image for you.

That is why every create, recreate and **join** now checks:

```
>>> Verifying dependencies against /home/you/aegis_gym/uv.lock...
>>> 9 package(s) differ from the lock:
      numpy         lock=2.4.2      installed=2.2.6
      protobuf      lock=7.34.0     installed=3.20.3
      ...
>>> Install the lock-pinned versions now? [Y]es / [n]o:
```

Saying yes installs just those pins with `--no-deps`, so nothing else moves.
`torch`, `torchvision`, `triton`, the `nvidia-*` runtimes and `rsl-rl-lib` are
never touched — they come from the CUDA wheel index and from git, and the lock
pins the PyPI build of the same name, so reinstalling them from the lock is
exactly the `+cu128` over `+cu129` mismatch the image is built to avoid.
`--no-verify` skips the check; run it by hand with

```bash
bash container/scripts/verify_deps.sh ~/path/to/aegis_gym [--install]
```

It also runs `uv pip check`, which surfaces problems no lock comparison can
see. One is already known and structural: `install_hardware_control.sh`
apt-installs `python3-protobuf` and `python3-grpcio` in a **later layer** than
the pip install, and Ubuntu 22.04's protobuf 3.20.3 shadows the lock's version,
which `onnx` (requiring `>=4.25.1`) is unhappy about. Fixing that needs a
change to the base image, not to the container you are in.

Two more details that trip people up:

**The editable install needs `sudo`.** toolbx runs as your host user, who
cannot write to the image's root-owned `dist-packages`. The script handles it;
`--no-editable` skips the step entirely.

**Shared `$HOME` wants to shadow the image.** Because toolbx shares `$HOME`,
your host's `~/.local/lib/python3.*/site-packages` sits ahead of the image's
own packages, so `import torch` would find whatever the host has rather than
the cu129 build this image is built around. `PYTHONNOUSERSITE=1` in
`dev/Containerfile.toolbx` is what prevents that, and because it is baked into
the image it holds for `toolbox enter` and `toolbox run` alike — you do not
have to pass anything.

Setting it *around* toolbx does not work and is worth knowing about: toolbx
forwards only its own fixed list of variables (`COLORTERM`, `DISPLAY`, `LANG`,
`TERM`, `XDG_*`, …), so `env PYTHONNOUSERSITE=1 toolbox enter …` drops the
variable on the way in. If `torch.__version__` reports **cu128** inside the
container, the image is missing that `ENV` — rebuild it with
`aegis_gym_toolbx --no-cache` and recreate the container.

**The viewer works here, headless needs asking for.** The development image
sets `PYOPENGL_PLATFORM=glx` and clears `PYGLET_HEADLESS`, the same pair
`aegis_gym_run --gui` uses, so `train.py -v` opens a window. For a long run
over SSH with no `DISPLAY`, put it back per command:

```bash
PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=1 python3 train.py -a rl --env reacher
```

---

## Production — `aegis_gym_run`

```bash
aegis_gym_run train -a rl --env reacher -e TEST_PLAYGROUND_run1 -B 4096
aegis_gym_run eval --env reacher -e TEST_PLAYGROUND_run1
aegis_gym_run --gui eval --env reacher        # forward X11 for the viewer
aegis_gym_run shell                           # bare shell in the image
```

`--env` selects the environment (`reacher` or `push_t`), `-e/--exp-name` is the
ClearML experiment name, `-B/--num-envs` the parallel env count. Everything
after the command goes to `train.py` / `eval.py` untouched, so their own flags —
`--load-rl-model-id`, `--load-bc-model-id`, `--max-iterations`, `--control` —
work as usual. Use `-h` on those scripts for the full list.

The first run builds `ceai/aegis_gym_prod:<ver>`, showing you what it is about
to build and offering `[Y]es / [e]dit / [a]bort`. The branch comes from the
checkout you run the command in, falling back to `devel`. Before starting, it
prints the image's provenance — branch, commit, build date — and warns when that
branch has moved since then.

Runs are **headless by default**, which is what training wants. You get
`--network host` for the gRPC bridge, the NVIDIA GPU via CDI, and
`~/clearml.conf` mounted read-only when present.

| Flag                              | Effect                                                           |
| --------------------------------- | ---------------------------------------------------------------- |
| `--gui`                           | Forward X11 and switch Genesis to GLX so the viewer can open     |
| `--gpu nvidia\|none\|auto`        | Force or disable the GPU (default `auto`)                        |
| `--no-clearml` / `--clearml PATH` | Skip or relocate the config mount                                |
| `--shm-size SIZE`                 | Private IPC namespace with that `/dev/shm` instead of the host's |
| `-b` / `-B`                       | Build, or rebuild ignoring the cache, before running             |
| `--no-run`                        | Build and/or push only                                           |
| `--dry-run`                       | Print the podman command instead of running it                   |

`--control ros` talks to the real robot and needs the `aegis_ros` stack running
on this host — that is what `--network host` is for.

### Running on a ClearML queue

The production image has **no `ENTRYPOINT`** on purpose. A ClearML agent runs it
as

```
docker run --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 <image> bash -c '<agent script>'
```

and injects the task's code and arguments itself; an `ENTRYPOINT` would be
prepended to that command and break every queued task.
`CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1` is baked in as a default so the agent
uses the system Python with the dependencies already present. A task config can
still override it.

The manual ergonomics live in `/usr/local/bin/aegis-gym` instead — an ordinary
program on `$PATH` that `aegis_gym_run` passes as the container command. The
agent never sees it.

Publish an image for a queue with:

```bash
aegis_gym_run -b --no-run --push-as ghcr.io/agh-ceai/aegis_gym:v0.1.0
```

`--push` without `--push-as` uses the default registry
(`geonosis:5000/ceai/aegis_gym_prod:<ver>`).

---

## Check your setup

After building, these three confirm the parts that usually go wrong:

```bash
# GPU reaches the container, and torch is the CUDA build we intended
aegis_gym_run -y shell python3 -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.8.0+cu129 True

# the simulator runs headless on the GPU
aegis_gym_run -y shell python3 -c \
  "import genesis as gs; gs.init(backend=gs.gpu, logging_level='warning'); print('genesis ok')"

# the production image still satisfies the ClearML contract
podman inspect ceai/aegis_gym_prod:v0.1.0 --format '{{.Config.Entrypoint}}'
# -> []
```

A fuller pass over every script lives in [TESTING.md](TESTING.md).

---

## Cleanup

```bash
aegis_gym_clean              # dev containers + localhost/aegis_gym_dev images
aegis_gym_clean --all        # also offers the prod, base and torch images
aegis_gym_clean -y           # no prompts
```

Each group is confirmed separately, and images still in use are reported
without aborting the sweep. `-y --all` together removes the base tiers as well —
about 11 GB and a full rebuild to get back.

---

## Without the commands

Each command is a thin wrapper around podman. The equivalents:

```bash
# base tiers (build context: this directory)
podman build . -f Containerfile.torch -t ceai/aegis_gym_torch:v0.1.0
podman build . -f Containerfile \
    --build-arg TORCH_REF=ceai/aegis_gym_torch:v0.1.0 \
    --build-arg AEGIS_GYM_TAG=devel \
    --build-arg CEAI_RSL_RL_TAG=v3.3.2 \
    --build-arg AEGIS_ROS_TAG=humble-devel \
    -t ceai/aegis_gym:v0.1.0

# production (build context: run/)
podman build run -f run/Containerfile.prod \
    --build-arg BASE_REF=ceai/aegis_gym:v0.1.0 \
    --build-arg AEGIS_GYM_TAG=devel \
    -t ceai/aegis_gym_prod:v0.1.0
podman run --rm -it --network host --ipc host \
    --device nvidia.com/gpu=all --security-opt label=disable \
    -v ${HOME}/clearml.conf:/root/clearml.conf:ro \
    ceai/aegis_gym_prod:v0.1.0 aegis-gym train -a rl --env reacher -e TEST_PLAYGROUND_manual

# development (build context: dev/)
podman build dev -f dev/Containerfile.toolbx \
    --build-arg BASE_REF=ceai/aegis_gym:v0.1.0 \
    -t localhost/aegis_gym_dev:v0.1.0
toolbox create --image localhost/aegis_gym_dev:v0.1.0 aegis_gym_dev-v0.1.0
toolbox run --container aegis_gym_dev-v0.1.0 bash -lc \
    "sudo uv pip install --system --no-deps --editable /path/to/aegis_gym"
toolbox enter aegis_gym_dev-v0.1.0
```

Note that `--ipc host` and `--shm-size` are mutually exclusive in podman: host
IPC already gives the container the host's `/dev/shm`, which is what the
DataLoader workers need.

---

## Troubleshooting

**`crun: cannot stat '/usr/lib/.../libEGL_nvidia.so.<version>'`**
The CDI spec is stale — the NVIDIA driver was updated under it. Regenerate:
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Expect this after
every driver upgrade.

**`torch.__version__` reports `+cu128` inside the toolbx**
Your host's `~/.local` packages are shadowing the image, which means the image
is missing `ENV PYTHONNOUSERSITE=1`. Rebuild it and recreate the container:
`aegis_gym_toolbx --no-cache`, then `[r]ecreate`. Exporting the variable around
`toolbox enter` does not help — toolbx does not forward it.

**`Error: base image ceai/aegis_gym:<ver> not found`**
Run `aegis_gym_build_image -v <ver>` first. `aegis_gym_run` and
`aegis_gym_toolbx` both derive from it and will not build it for you.

**`no space left on device` during a build**
The image set needs ~11 GB and podman needs working room on top. Reclaim with
`podman image prune` (untagged layers only, safe), then retry — completed layers
are cached, so the build resumes rather than restarting.

**`Failed to download … operation timed out` while installing torch**
A transient PyPI hiccup, not a configuration problem. Rerun the same command;
everything already downloaded is cached.

**`cannot set shmsize when running in the {host } IPC Namespace`**
Only if you hand-roll the podman command: pass `--ipc host` *or* `--shm-size`,
never both.

**`unknown or unsupported mesh file: '….dae'` and "Falling back to legacy URDF parser"**
Benign. Genesis's newer URDF path goes through MuJoCo, which reads only
STL/OBJ/MSH, so COLLADA collision meshes send it to the legacy parser. Physics
defaults may differ slightly. Installing `pycollada` does *not* help — the limit
is MuJoCo's, and trimesh already reads `.dae` fine. The real fix is converting
those meshes to STL/OBJ upstream.

**`Error: container 'aegis_gym_prod' already exists`**
A previous run did not clean up. `podman rm -f aegis_gym_prod`, or pass
`--name` to run alongside it.

**`sudo: unable to resolve host toolbox`**
Add `127.0.0.1 toolbox` and `127.0.0.1 toolbx` to `/etc/hosts`.

**`user is not in the sudoers file` inside toolbx**
Install `crun` 1.8-1 or newer, then `newgrp sudo`.

---

## See also

- [Root README](../README.md) — environments, training CLI, project layout
- [TESTING.md](TESTING.md) — manual test pass for these scripts
- [CHANGELOG.md](CHANGELOG.md) — what changed in the container setup
- [`aegis_ros`](https://github.com/AGH-CEAI/aegis_ros) — the ROS 2 side of sim-to-real
- [`clearml_utils`](https://github.com/AGH-CEAI/clearml_utils) — working with ClearML tasks
