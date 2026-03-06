# Containerfile for the aegis_gym

## Usage
### Build and upload
```bash
# Setup the arguments
VERSION=v1.0.0 REGISTRY=localhost PUSH=false ./build.sh
VERSION=v1.0.0 REGISTRY=genesis.local PUSH=true ./build.sh
VERSION=v0.0.1 REGISTRY=ghcr.io PUSH=true ./build.sh
```
Will create
* `${REGISTRY}/ceai/aegis_gym:${VERSION}-simulator` for learning only
* `${REGISTRY}/ceai/aegis_gym:${VERSION}-hardware` for ROS-gRPC bridge
and optionally push these images to the given repository.

### Manual build
```bash
podman build . -t ceai/aegis_gym:test
```
