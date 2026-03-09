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

### Run
```bash
podman run --rm -it --device nvidia.com/gpu=all -v ${HOME}/clearml.conf:/root/clearml.conf:ro localhost/agh-ceai/aegis_gym:v1.0.0 bash
# In container
git clone https://github.com/AGH-CEAI/aegis_gym.git
cd aegis_gym/aegis_gym/rsl
python3 grasp_train.py --stage=rl --control=sim
python3 grasp_eval.py --stage=rl --control=sim --no_vis
python3 grasp_eval.py --stage=rl --control=ros
```

## Manual build
```bash
podman build . -t ceai/aegis_gym:test
```

---

## Development notes
Ensure that the Nvidia container toolkit genreted proper configuration for the podman:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```
