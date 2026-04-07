# Utility scfipts

[Back to main README](../README.md)

---
## [`clearml_enquee_tasks.py`](./clearml_enquee_tasks.py)
Clone task multiple times into a selected queue.

---
## [`clearml_summarizer.py`](./clearml_summarizer.py)
Summarize ClearML experiments.

### Usage
```bash
uv run ./clearml_summarizer.py --cleanup-previous-tags --project-name PROJECT/PATH --tags bc
uv run ./clearml_summarizer.py -h
```


---
## [`upload_urdf_to_clearml.py`](./upload_urdf_to_clearml.py)
Upload URDF model, required by simulators, into ClearML.

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

[Back to main README](../README.md)
