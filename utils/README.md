# Utility scripts

[Back to main README](../README.md)

---
## Tools for ClearML tasks management

> [!NOTE]
> Be sure to get the ClearML project structure to this:
```
PROJECTS/.../
└── YOUR_PROJECT/
    ├── EXPERIMENT_1/
    │   ├── SUMMARY
    │   ├── trial_run_1
    │   ├── trial_run_2
    │   ├── ...
    │   └── trial_run_N
    ├── EXPERIMENT_2/
    │   ├── SUMMARY
    │   ├── trial_run_1
    │   ├── trial_run_2
    │   ├── ...
    │   └── trial_run_N
    ├── EXPERIMENT_3/
    │   ├── SUMMARY
    │   ├── trial_run_1
    │   ├── trial_run_2
    │   ├── ...
    │   └── trial_run_N
    └── EXPERIMENTS_SUMMARY
```
where `SUMMARY` is the result of the `clearml_summarizer.py` run in the `YOUR_PROJECT/EXPERIMENT_X` project name path, and the `EXPERIMENTS_SUMMARY` is the result of the `clearml_exp_plotter.py` run in the `YOUR_PROJECT` project name path.

---

### [`clearml_enqueue_tasks.py`](./clearml_enqueue_tasks.py)
Clone task multiple times into a selected queue.

---

### [`clearml_exp_plotter.py`](./clearml_exp_plotter.py)
Aggregate summaries of multiple ClearML experiments into comparison plots!

#### Usage

To use it properly, you need to run the `clearml_summarizer.py` first.

```bash
uv run --script ./clearml_exp_plotter.py --cleanup-previous-tags --project-name PROJECT/PATH
uv run --script ./clearml_exp_plotter.py -h
```

---
## [`clearml_summarizer.py`](./clearml_summarizer.py)
Summarize ClearML experiments.

### Usage
```bash
uv run --script ./clearml_summarizer.py --cleanup-previous-tags --project-name PROJECT/PATH --tags bc
uv run --script ./clearml_summarizer.py -h
```

---

## Dataset manipulation

---
### [`upload_urdf_to_clearml.py`](./upload_urdf_to_clearml.py)
Upload URDF model, required by simulators, into ClearML.

#### Uploading
1. Generate standalone URDF model with [aegis_ros/aegis_description](https://github.com/AGH-CEAI/aegis_ros/tree/humble-devel/aegis_description) launch command:
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

#### Usage

In the robot's config set the `urdf_model_id` param to the ClearML's dataset ID.

> [!IMPORTANT]
> In case of failure to obtain the model, the code will try to load URDF model from `~/ceai_ws/aegis_urdf` directory.

---

[Back to main README](../README.md)
