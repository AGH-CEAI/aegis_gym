# aegis_gym

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Licence](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)

The collection of [RSL-RL](https://github.com/leggedrobotics/rsl_rl) reinforcement learning environments for the [Aegis UR5e station](https://github.com/AGH-CEAI/pprai2026_drl_robotic_station).
Featuring real-time control for the real robot via [gRPC<->ROS 2 bridge](https://github.com/AGH-CEAI/aegis_ros/tree/humble-devel/aegis_grpc) and [Genesis-World](https://genesis-world.readthedocs.io/) scenes.

<p align="center">
<img src="./docs/4096_units_are_ready.png" alt="Static image of 4096 parallel instances of the Aegis station in Genesis simulator." width="768"/>
</p>

---

## Containers
Check the corresponding [README.md](./container/README.md).

---

## Run

### Build & install
```bash
uv build
pip3 install ./dist/aegis_gym-*.whl
# Combined command:
uv build && pip3 uninstall aegis_gym -y && pip3 install "./dist/aegis_gym-0.0.1-py3-none-any.whl[sim-genesis]"
```

### Train
# TODO

### Eval
# TODO

---
## Utilities

Check out the [utils README](./utils/README.md).

---
## Development notes

### Varia

* All poses consist of position `x,y,z` and orientation in quaterion form `qx,qy,qz,qw`, i.e.: `pose=[x,y,z,qx,qy,qz,qw]` (This is the Genesis notation, **it differs from ROS 2**).
* For simplicity, the project uses Pytorch tensors instead of numpy ones.

### Automatic tests
# TODO

### pre-commit / prek

This project uses various tools for aiding the quality of the source code. Currently most of them are executed by the `pre-commit`. As a faster alternative it is suggested to use `prek`. Please make sure to enable its hooks:

```bash
# In case of pre-commit
pre-commit install
# In case of prek
prek install
```

---
## License
This repository is licensed under the Apache 2.0, see LICENSE for details.
