# Two-Stage Training

1. **Stage 1 – Teacher Reinforcement Learning Policy:**
   A privileged teacher policy is trained using full state information. This allows fast and efficient learning in simulation with access to all relevant data.

2. **Stage 2 – Vision-Based Student Policy:**
   Knowledge from the teacher policy is distilled into a student policy that uses only camera observations (and optionally robot proprioception). This step bridges the gap toward real-world deployment, where full state information is unavailable.


## Training

- Train the RL teacher policy:
```bash
python3 grasp_train.py --stage=rl
python3 grasp_train.py --stage=rl --num_envs=1
python3 grasp_train.py --stage=rl --num_envs=1 --control=ros
```
- Train the student BC policy (requires a trained RL teacher):
```bash
python3 grasp_train.py --stage=bc
```

## Evaluation
- Evaluate a trained RL policy:
```bash
python3 grasp_eval.py --stage=rl
python3 grasp_eval.py --stage=rl --load-rl-task-id aa7a4ad4079a4964b7c33a46651c9eab
```
- Evaluate a trained BC policy:
```bash
python3 grasp_eval.py --stage=bc
```
- Record evaluation videos:
```bash
python3 grasp_eval.py --stage=bc --record
```

## Camera Setups
The environment supports configurable camera layouts, allowing different perception setups to be selected at initialization time.
Currently, two camera setups are available:
* **default** – matches the real robot setup (without the front tool camera): one static top-down scene camera plus two eye-in-hand cameras moving with the end-effector
* **dual_scene** – legacy configuration: two static scene cameras forming a stereo view from the front

## Hyperparameter Optimization
This repository includes a template script `hpo.py` for hyperparameter optimization using ClearML.
HPO automatically searches for the best training configuration by running multiple experiments with different hyperparameters.

The script requires a base task, which is a previously executed training task stored in ClearML.
This task is used as a template and will be cloned during optimization.

When running `hpo.py`, ClearML creates one optimizer task (manages the optimization process) and multiple child tasks, each representing a single experiment with different hyperparameters.

The script contains a user configuration section where you define settings such as the base task ID, optimization metric, optimizer type, execution queue, concurrency limits, total number of jobs, maximum iterations, pool period, report period, and the hyperparameters to optimize.
Comments in the template explain what each setting does and how to adapt it for your experiments.
