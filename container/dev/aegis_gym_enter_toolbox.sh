#!/usr/bin/env bash

set -euo pipefail

torch_image="localhost/aegis-torch:torch2.8-cu129"
dev_image="localhost/aegis-gym-dev:devel"
toolbox_name="aegis_gym-dev"

torch_containerfile="Containerfile.torch"
dev_containerfile="Containerfile.dev"

workspace_path="$HOME/ceai_ws"
aegis_gym_path="$workspace_path/src/aegis_gym"

ubuntu_version="22.04"
cuda_short_version="cu129"
torch_version="2.8.0"
torchvision_version="0.23.0"

aegis_gym_tag="devel"
ceai_rsl_rl_tag="ceai-devel"
aegis_ros_tag="humble-devel"

no_cache=false
skip_editable=false

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -w PATH    Workspace path on host (UNDER CONSTRUCTION)
             Default: $workspace_path

  -n         Build images with --no-cache

  -s         Skip editable installation of local aegis_gym

  -h         Show this help
EOF
}

confirm_N() {
  local prompt="$1"
  local choice

  while true; do
    read -r -p "$prompt [y/N] " choice

    case "$choice" in
      [yY])
        return 0
        ;;
      [nN]|"")
        return 1
        ;;
      *)
        echo "Response not valid. Please answer y or n." >&2
        ;;
    esac
  done
}

confirm_Y() {
  local prompt="$1"
  local choice

  while true; do
    read -r -p "$prompt [Y/n] " choice

    case "$choice" in
      [yY]|"")
        return 0
        ;;
      [nN])
        return 1
        ;;
      *)
        echo "Response not valid. Please answer y or n." >&2
        ;;
    esac
  done
}

build_torch_image() {
  local cache_args=()

  if [ "$no_cache" = true ]; then
    cache_args+=(--no-cache)
  fi

  echo
  echo "Building Torch image: $torch_image"

  podman build \
    "${cache_args[@]}" \
    --file "$torch_containerfile" \
    --tag "$torch_image" \
    --build-arg="UBUNTU_VERSION=$ubuntu_version" \
    --build-arg="CUDA_SHORT_VERSION=$cuda_short_version" \
    --build-arg="TORCH_VERSION=$torch_version" \
    --build-arg="TORCHVISION_VERSION=$torchvision_version" \
    .
}

build_dev_image() {
  local cache_args=()

  if [ "$no_cache" = true ]; then
    cache_args+=(--no-cache)
  fi

  echo
  echo "Building development image: $dev_image"

  podman build \
    "${cache_args[@]}" \
    --file "$dev_containerfile" \
    --tag "$dev_image" \
    --build-arg="TORCH_BASE_IMAGE=$torch_image" \
    --build-arg="AEGIS_GYM_TAG=$aegis_gym_tag" \
    --build-arg="CEAI_RSL_RL_TAG=$ceai_rsl_rl_tag" \
    --build-arg="AEGIS_ROS_TAG=$aegis_ros_tag" \
    .
}

create_toolbox() {
  if podman container exists "$toolbox_name"; then
    echo
    echo "Toolbox container already exists: $toolbox_name"
    return
  fi

  toolbox create \
    --container "$toolbox_name" \
    --image "$dev_image"
}

install_local_aegis_gym() {
    if [ "$skip_editable" = true ]; then
        echo
        echo "Skipping editable installation."
        return
    fi

    if [ ! -f "$aegis_gym_path/pyproject.toml" ]; then
        echo
        echo "Local aegis_gym repository was not found:"
        echo " $aegis_gym_path"
        echo
        echo "The Toolbox will still be started, but local Python files"
        echo "will not replace the package installed in the image."
        return
    fi

    echo
    echo "Installing local aegis_gym as editable without dependencies:"
    echo " $aegis_gym_path"

    toolbox run \
        --container "$toolbox_name" \
        bash -lc "
        uv pip install --system --no-deps --editable '$aegis_gym_path'
        "

    echo
    echo "Python will import aegis_gym from:"

    toolbox run \
        --container "$toolbox_name" \
        bash -lc "
        python3 -c 'import aegis_gym; print(aegis_gym.__file__)'
        "
}

# Parse command-line options
while getopts ":nsh" option; do
  case "$option" in
    n)
      no_cache=true
      ;;
    s)
      skip_editable=true
      ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

# Display configuration
echo "Torch image:          $torch_image"
echo "Development image:    $dev_image"
echo "Toolbox name:         $toolbox_name"
echo "Workspace:            $workspace_path"
echo "Local aegis_gym:      $aegis_gym_path"
echo "aegis_gym tag:        $aegis_gym_tag"
echo "rsl_rl tag:           $ceai_rsl_rl_tag"
echo "aegis_ros tag:        $aegis_ros_tag"
echo "Build without cache:  $no_cache"
echo "Editable install:     $([ "$skip_editable" = true ] && echo false || echo true)"

echo
if ! confirm_Y "continue with this configuration?"; then
  echo "Aborted."
  exit 0
fi

# Validate required files
if [ ! -f "$dev_containerfile" ]; then
  echo "Missing file: $dev_containerfile" >&2
  exit 1
fi

# Build the base image only when missing
if ! podman image exists "$torch_image"; then
    echo
    echo "Torch image does not exist: $torch_image"

    if confirm_Y "Build the Torch image and continue?"; then
        if [ ! -f "$torch_containerfile" ]; then
        echo "Missing file: $torch_containerfile" >&2
        exit 1
        fi

        build_torch_image
    else
        echo "Aborted."
        exit 0
    fi
else
    echo
    echo "Torch image already exists: $torch_image"
    if confirm_N "Do you want to rebuild image $torch_image?"; then
        build_torch_image
    fi
fi

build_dev_image

mkdir -p "$workspace_path/src"

create_toolbox
# install_local_aegis_gym

if confirm_Y "Enter toolbox?"; then
    echo
    echo "Entering Toolbox: $toolbox_name"
    echo "Use 'exit' to leave it."
    echo

    exec toolbox enter "$toolbox_name"
else
    echo
    echo "Created toolbox container: $toolbox_name"
    echo "To enter use:"
    echo "  env PYTHONNOUSERSITE=1 PYTHONPATH= toolbox enter $toolbox_name"
    echo
fi
