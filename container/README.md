# Containerfile for the aegis_gym

## Usage
### Build and upload
```bash
# Setup the arguments
./build.sh VERSION=v1.0.0 REGISTRY=192.168.0.100:5000 PUSH=true
```
Will create
* `${REGISTRY}/ceai/aegis_gym:${VERSION}-simulator` for learning only
* `${REGISTRY}/ceai/aegis_gym:${VERSION}-hardware` for ROS-gRPC bridge
and optionally push these images to the given repository.

### Manual build
```bash
podman build . -t ceai/aegis_gym:test
```
