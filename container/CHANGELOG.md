# Changelog

All notable changes to this container file will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `fzf` in the development image, with a dpkg `path-include` drop-in so the shell integration under `/usr/share/doc/fzf/examples` survives the Ubuntu base image's `path-exclude=/usr/share/doc/*`.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `locales` in the development image. `dev/Containerfile.toolbx` takes a `LOCALE` build arg (default `en_US.UTF-8`) and `aegis_gym_toolbx` passes the host's `$LANG`.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `openssh-client` in the development image, so `git` can talk to `ssh://` and `git@host:` remotes. It is only a `Recommends` of `git`, which the torch tier drops with `--no-install-recommends`.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `install_links.sh`, which symlinks `aegis_gym_build_image`, `aegis_gym_run`, `aegis_gym_toolbx` and `aegis_gym_clean` into `~/.local/bin`.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `ceai/aegis_gym_prod` production image (`run/Containerfile.prod`) with `aegis_gym` installed at a fixed ref and provenance labels, plus the `/usr/local/bin/aegis-gym` dispatcher and `run/aegis_gym_run.sh` to build, push and run it.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `dev/Containerfile.toolbx` and `dev/aegis_gym_toolbx.sh`, replacing the previous enter/destroy script pair.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `aegis_gym_clean.sh`, with `--all` to also sweep the prod, base and torch images that the old destroy script left behind.

### Changed

- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - The in-image `aegis-gym` dispatcher now reports an unrecognised command itself instead of letting `exec` fail, and `aegis_gym_run` warns about one before it builds or starts anything. Running a real command verbatim is unchanged; only the failure path is different.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - Restructured to match the `aegis_docker` architecture: a base image with the dependencies pre-installed, a production image run through `podman run`, and a development image entered through toolbx.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - The base image is now built in two tiers, `ceai/aegis_gym_torch` and `ceai/aegis_gym`, so rebuilding the dependency tier does not redo the CUDA wheel install. `Containerfile.torch` moved up from `dev/`.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `Containerfile` now drives `scripts/install_simulation.sh` and `scripts/install_hardware_control.sh`, so the lock-pinned install is the only one left. The unlocked `pyproject.toml` install is gone.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - Image names unified on `ceai/*`; the default push registry is `geonosis:5000`.

### Deprecated

### Removed

- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `build.sh`, `dev/Containerfile.dev`, `dev/aegis_gym_enter_toolbox.sh` and
  `dev/aegis_gym_destroy_toolbox.sh`, superseded by the four new commands.

### Fixed

- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - Powerline prompts in the toolbx shell. `toolbox enter` forwards the host's `$LANG` but not its `$LC_ALL`, and the image had no matching locale, so zsh fell back to single-byte C; the agnoster theme then aborted on `$'\ue0b0'` with "character not in range" and left a bare prompt.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - The editable install of the local checkout ran against a hard-coded `$HOME/ceai_ws` and was commented out; it now finds the checkout from `$PWD` and runs by default (`--no-editable` opts out). It also needed `sudo` to work at all: toolbx runs as the host user, who cannot write to the image's root-owned `dist-packages`. `sudo` additionally selects the image's `/usr/local/bin/uv` over the host's `~/.local/bin/uv`, which `$HOME` sharing otherwise puts first on `PATH`.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - `uv export` emits the project itself as `-e .`, so the base image ended up with an editable install pointing at `/tmp/aegis_gym`, which the same layer then deleted. Added `--no-emit-project`.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - The install scripts assumed a `/ws` created by an upstream `WORKDIR`; they now create it themselves.
- [PR-155](https://github.com/AGH-CEAI/aegis_gym/pull/155) - The lock-pinned install replaced the CUDA wheel-index torch with the PyPI build, leaving `torch==2.8.0+cu128` beside `torchvision==0.23.0+cu129` while the image labels still claimed `cu129`. Packages owned by an earlier install -- torch, torchvision, triton, the `nvidia-*` runtime libraries and the `rsl_rl` fork -- are now excluded from the lock export, so the torch tier and the git fork keep their builds. The exclusion list is derived from the lock, not hard-coded. The `rsl_rl` fork had been surviving only because its version matched the lock's pin.

### Security


## [v202605061407]

### Added

- [PR-138](https://github.com/AGH-CEAI/aegis_gym/pull/138) - Scripts to autobuild and remove the development toolbx container.

## [v202603091815]

### Added

- [PR-70](https://github.com/AGH-CEAI/aegis_gym/pull/70) - Created first headless learning container.
