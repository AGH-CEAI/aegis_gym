# aegis_gym

The collection of [Gymnasium](https://gymnasium.farama.org/) environments for the Aegis UR5e station.

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Licence](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<p align="center">
    <img src="./docs/aegis_reacher.png" alt="Static image of the AegisReacher environment in Genesis simulator." width="640"/>
</p>


---

## Notes
* All poses consist of position `x,y,z` and orientation in quaterion form `qx,qy,qz,qw`, i.e.: `pose=[x,y,z,qx,qy,qz,qw]`
* For simplicity, the project uses Pytorch tensors instead of numpy ones.

---

## Build & install
```bash
poetry build
pip3 install ./dist/aegis_gym-*.whl
# Combined command:
poetry build && pip3 uninstall aegis_gym -y && pip3 install "./dist/aegis_gym-0.0.1-py3-none-any.whl[sim-genesis]"
```

## Run tests
```bash
poetry run pytest -v -s
```
## Run test training
```bash
python3 ./test/sb3_run_train.py
```
---
## Development notes

This project uses various tools for aiding the quality of the source code. Currently most of them are executed by the `pre-commit`. Please make sure to enable its hooks:

```bash
pre-commit install
```

---
## License
This repository is licensed under the Apache 2.0, see LICENSE for details.
