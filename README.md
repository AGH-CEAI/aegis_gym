# aegis_gym

The collection of [Gymnasium](https://gymnasium.farama.org/) environments for the Aegis UR5e station.

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Licence](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)

<p align="center">
    <img src="./docs/aegis_reacher.png" alt="Static image of the AegisReacher environment in Genesis simulator." width="640"/>
</p>

---

## Notes
* All poses consist of position `x,y,z` and orientation in quaterion form `qx,qy,qz,qw`, i.e.: `pose=[x,y,z,qx,qy,qz,qw]`
* For simplicity, the project uses Pytorch tensors instead of numpy ones.

---

## Containerfile
Check the corresponding [README.md](./container/README.md).

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
## URDF model for simulator

### Uploading
1. Generate standalone URDF model with [aegis_ros/aegis_descrption]() launch command:
```bash
ros2 launch aegis_description generate_standalone_urdf.launch.py disable_cell:=true
```
Which will generate the whole URDF file with 3D models in a default `~/ceai_ws/aegis_urdf` directory.

2. Run the [`utils/upload_urdf_to_clearml.py`](./utils/upload_urdf_to_clearml.py) script with the following options:
```bash
python3 utils/upload_urdf_to_clearml.py ~/ceai_ws/aegis_urdf --name AegisURDFModel --project AEGIS_GRASP --desc "Aegis simulator assets"
```
> [!WARNING]
> **To update the dataset** make sure to add an additional option: `--parent "PREVIOUS_DATASET_ID"`

3. Check the ClearML server's datasets.

### Usage

In the robot's config set the `urdf_model_id` param to the ClearML's dataset ID.

> [!IMPORTANT]
> In case of failure to obtain the model, the code will try to load URDF model from `~/ceai_ws/aegid_urdf` directory.



---
## Development notes

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
